"""Experiment: can the classifier say "unknown" for signal types it was never trained on?

The trained CNN (models/cnn_baseline.pt, unchanged) is scored by five open-set methods
(spectrum_intel/openset.py). Everything is fitted and calibrated on KNOWN signals only:
  - class centres / covariance / kNN bank: known training-distribution data (seed 0, the training seed)
  - thresholds: known validation data (seed 12), set so that 95% of known signals are accepted
  - test: known signals (seed 11) and six unknown types (seed 13), never used for fitting

An exploratory run on other seeds (0-3) came first. The hypotheses below record what it suggested;
this script tests them on fresh data.
  H1  Softmax-based scores (msp, max logit, energy) barely separate unknowns: the CNN is confident
      about signals it never saw.
  H2  Feature-distance scores (Mahalanobis, kNN) reject the far unknowns (GFSK, AM, FM) almost
      completely at SNR >= 0 dB, while rejecting 5% of known signals.
  H3  The near unknowns 8-PSK and 64-QAM are accepted as QPSK and 16-QAM by every score, because
      the CNN's features do not separate them (it never needed to).
  H4  Neither earlier CNN layers nor classic higher-order statistics (cumulants) fix H3 here.
  H5  Below about 0 dB unknown signals look like noise and are accepted as "noise"; most rejected
      known signals are low-SNR ones.

    python scripts/exp_openset.py      # ~2 min
Output: results/experiments/openset/<run>/  and models/openset_baseline.npz (detector statistics)
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import BASELINE_MODEL

from spectrum_intel import classifier as cl
from spectrum_intel import experiments as ex
from spectrum_intel import openset as osr
from spectrum_intel import spectrum as sp
from spectrum_intel.dataset import DEFAULT_SNRS, make_dataset
from spectrum_intel.dsp import awgn
from spectrum_intel.plots import AQUA, BLUE, GRID, INK, MUTED, ORANGE, YELLOW, save
from spectrum_intel.signals import CLASSES, clean_signal

SCORECOL = {"msp": MUTED, "max_logit": YELLOW, "energy": ORANGE, "mahalanobis": BLUE, "knn": AQUA}
SCORENAME = {"msp": "softmax confidence", "max_logit": "max logit", "energy": "energy",
             "mahalanobis": "Mahalanobis", "knn": "kNN distance"}


# ---- H4: other features (kept here so the negative result can be reproduced) ---------------------
@torch.no_grad()
def layer_features(model, X):
    """Mean and std over time of each convolution block's output."""
    outs = [[] for _ in model.features]
    for i in range(0, len(X), 512):
        h = torch.from_numpy(np.ascontiguousarray(X[i:i + 512]))
        for j, blk in enumerate(model.features):
            h = blk(h)
            outs[j].append(torch.cat([h.mean(-1), h.std(-1)], 1).numpy())
    return [np.concatenate(o) for o in outs]


def cumulants(X):
    """Classic modulation statistics: higher-order cumulants, envelope and phase-step spread."""
    x = X[:, 0] + 1j * X[:, 1]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2, 1, keepdims=True))
    m20, m21 = np.mean(x ** 2, 1), np.mean(np.abs(x) ** 2, 1)
    m40, m41, m42 = np.mean(x ** 4, 1), np.mean(x ** 3 * np.conj(x), 1), np.mean(np.abs(x) ** 4, 1)
    m63, m80 = np.mean(np.abs(x) ** 6, 1), np.mean(x ** 8, 1)
    c40, c41 = m40 - 3 * m20 ** 2, m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
    c63 = m63 - 9 * m42 * m21 + 12 * m21 ** 3
    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1]))
    return np.stack([np.abs(c40), np.abs(c41), c42.real, c63.real, np.abs(m80), np.std(np.abs(x), 1),
                     np.std(dphi, 1), np.abs(np.mean(np.exp(1j * dphi), 1))], 1)


def shared_maha(Ftr, ytr):
    means = np.stack([Ftr[ytr == c].mean(0) for c in range(len(CLASSES))])
    R = Ftr - means[ytr]
    cov = R.T @ R / len(R)
    P = np.linalg.inv(cov + 1e-3 * np.trace(cov) / len(cov) * np.eye(len(cov)))
    return lambda F: -np.einsum("nci,ij,ncj->nc", F[:, None] - means[None], P, F[:, None] - means[None]).min(1)


def class_gauss(Ftr, ytr):
    par = []
    for c in range(len(CLASSES)):
        A = Ftr[ytr == c]
        C = np.cov(A.T)
        C = C + 1e-3 * np.trace(C) / len(C) * np.eye(len(C))
        par.append((A.mean(0), np.linalg.inv(C), np.linalg.slogdet(C)[1]))
    return lambda F: -np.min(np.stack([np.einsum("ni,ij,nj->n", F - mu, P, F - mu) + ld for mu, P, ld in par], 1), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-per-class", type=int, default=300)
    ap.add_argument("--per-class", type=int, default=300, help="known test examples per class")
    ap.add_argument("--per-kind", type=int, default=300, help="unknown test examples per type")
    a = ap.parse_args()
    run = ex.new_run("openset", {"model": "models/cnn_baseline.pt", "fit": f"{a.fit_per_class}/class, seed 0",
                                 "validation": "200/class, seed 12", "test_known": f"{a.per_class}/class, seed 11",
                                 "test_unknown": f"{a.per_kind}/type, seed 13", "target_tpr": osr.TARGET_TPR,
                                 "unknown_near": osr.UNKNOWN_NEAR, "unknown_far": osr.UNKNOWN_FAR,
                                 "snr_db": list(DEFAULT_SNRS)}, 13)
    model = cl.load_model(BASELINE_MODEL)
    Xf, yf, _ = make_dataset(a.fit_per_class, seed=0)
    Xv, yv, sv = make_dataset(200, seed=12)
    Xk, yk, sk = make_dataset(a.per_class, seed=11)
    Xu, ku, su = osr.make_unknown_set(a.per_kind, DEFAULT_SNRS, seed=13)
    det = osr.OpenSetDetector.fit(model, Xf, yf)
    th = det.calibrate(model, Xv)
    det.save(run / "openset_detector.npz")
    det.save(ex.ROOT / "models" / "openset_baseline.npz")
    S_k, S_u = det.scores(model, Xk), det.scores(model, Xu)
    hk, hu = sk >= 0, su >= 0
    rows, summary = [], {"thresholds": th}
    kinds = osr.UNKNOWN_KINDS

    # ---- H1-H3: scores compared --------------------------------------------------------------------
    res = {}
    for s in osr.SCORES:
        res[s] = {"auroc_all": osr.auroc(S_k[s], S_u[s]), "auroc_snr_ge_0": osr.auroc(S_k[s][hk], S_u[s][hu]),
                  "known_accepted_test": float(np.mean(S_k[s] >= th[s])),
                  "unknown_accepted_snr_ge_0": osr.unknown_accepted(S_u[s][hu], th[s]),
                  "accepted_by_kind_snr_ge_0": {k: osr.unknown_accepted(S_u[s][hu & (ku == i)], th[s])
                                                for i, k in enumerate(kinds)}}
        rows += [{"part": "scores", "score": s, "kind": k, "accepted_snr_ge_0": v}
                 for k, v in res[s]["accepted_by_kind_snr_ge_0"].items()]
        print(f"  {s:12s} AUROC {res[s]['auroc_snr_ge_0']:.3f} (>=0 dB)  known kept {res[s]['known_accepted_test']:.3f}  "
              "unknown accepted: " + " ".join(f"{k} {v:.2f}" for k, v in res[s]["accepted_by_kind_snr_ge_0"].items()))
    summary["scores"] = res

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    w = 0.16
    for j, s in enumerate(osr.SCORES):
        v = [100 * res[s]["accepted_by_kind_snr_ge_0"][k] for k in kinds]
        ax.bar(np.arange(len(kinds)) + (j - 2) * w, v, w, color=SCORECOL[s], label=SCORENAME[s])
    ax.axvline(2.5, color=INK, lw=0.8, ls=":")
    ax.text(1, 104, "near (look like known classes)", ha="center", fontsize=8, color=INK)
    ax.text(4, 104, "far (unlike anything known)", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(kinds)), ["8-PSK", "64-QAM", "other OFDM", "GFSK", "AM", "FM"])
    ax.set_ylabel("unknown signals wrongly accepted (%)")
    ax.set_ylim(0, 112)
    ax.set_title("SNR >= 0 dB; every score keeps 95% of known signals", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    ax = axes[1]
    ax.barh([SCORENAME[s] for s in osr.SCORES], [res[s]["auroc_snr_ge_0"] for s in osr.SCORES],
            color=[SCORECOL[s] for s in osr.SCORES])
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(0.505, -0.6, "chance", fontsize=8, color=MUTED)
    for i, s in enumerate(osr.SCORES):
        ax.text(res[s]["auroc_snr_ge_0"] + 0.01, i, f"{res[s]['auroc_snr_ge_0']:.2f}", va="center", fontsize=8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC known vs unknown (1 = perfect)")
    ax.set_title("Separation, SNR >= 0 dB", fontsize=10)
    save(fig, run / "scores.png")

    # ---- what the closed-set CNN calls each unknown, and what the open-set version says -------------
    best = "mahalanobis"
    closed = np.stack([np.bincount(S_u["pred"][hu & (ku == i)], minlength=6) for i in range(len(kinds))]).astype(float)
    opened = np.stack([np.bincount(np.where(S_u[best][hu & (ku == i)] >= th[best], S_u["pred"][hu & (ku == i)], 6),
                                   minlength=7) for i in range(len(kinds))]).astype(float)
    closed /= closed.sum(1, keepdims=True)
    opened /= opened.sum(1, keepdims=True)
    summary["closed_set_labels_of_unknowns"] = {k: dict(zip(CLASSES, closed[i].round(3).tolist())) for i, k in enumerate(kinds)}
    summary["open_set_labels_of_unknowns"] = {k: dict(zip(CLASSES + ("unknown",), opened[i].round(3).tolist()))
                                              for i, k in enumerate(kinds)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [6, 7]})
    for ax, M, cols, title in [(axes[0], closed, CLASSES, "Closed set: the CNN must pick a known class"),
                               (axes[1], opened, CLASSES + ("UNKNOWN",), "Open set (Mahalanobis): may answer unknown")]:
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.01:
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(cols)), cols, fontsize=8, rotation=20)
        ax.set_yticks(range(len(kinds)), kinds, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("true (unknown) signal, SNR >= 0 dB")
    save(fig, run / "labels.png")

    # ---- H5: SNR dependence ------------------------------------------------------------------------
    snrs = np.array(DEFAULT_SNRS, float)
    near = np.isin(ku, [kinds.index(k) for k in osr.UNKNOWN_NEAR])
    acc_k, rej_far, rej_near, kept_k = [], [], [], []
    for s in snrs:
        ok = (S_k[best] >= th[best]) & (sk == s)
        kept_k.append(float(np.mean(S_k[best][sk == s] >= th[best])))
        acc_k.append(float(np.mean(S_k["pred"][ok] == yk[ok])) if ok.any() else float("nan"))
        rej_far.append(float(np.mean(S_u[best][(su == s) & ~near] < th[best])))
        rej_near.append(float(np.mean(S_u[best][(su == s) & near] < th[best])))
        rows.append({"part": "snr", "snr_db": s, "known_kept": kept_k[-1], "known_accuracy_if_kept": acc_k[-1],
                     "far_unknown_rejected": rej_far[-1], "near_unknown_rejected": rej_near[-1]})
    noise_share = float(np.mean(S_u["pred"][(su <= -6) & ~near] == CLASSES.index("noise")))
    rejected_low = float(np.mean(sk[S_k[best] < th[best]] < 0))
    low = su <= -6
    low_labels = {k: dict(zip(CLASSES, np.bincount(S_u["pred"][low & (ku == i) & (S_u[best] >= th[best])], minlength=6).tolist()))
                  for i, k in enumerate(kinds)}
    rej_k = S_k[best] < th[best]
    rejected_known_by_class = dict(zip(CLASSES, np.bincount(yk[rej_k], minlength=6).tolist()))
    print(f"  rejected known signals by class: {rejected_known_by_class}")
    summary["snr"] = {"unknowns_accepted_at_le_minus6dB_labels": low_labels,
                      "rejected_known_by_class": rejected_known_by_class,"snr_db": snrs.tolist(), "known_kept": kept_k, "known_accuracy_if_kept": acc_k,
                      "far_unknown_rejected": rej_far, "near_unknown_rejected": rej_near,
                      "far_unknowns_at_le_minus6dB_called_noise": noise_share,
                      "share_of_rejected_known_below_0dB": rejected_low}
    print(f"  far unknowns at <= -6 dB called noise: {noise_share:.2f};  rejected knowns below 0 dB: {rejected_low:.2f}")
    print("  by SNR: " + "  ".join(f"{s:g}:{100 * f:.0f}/{100 * n:.0f}" for s, f, n in zip(snrs, rej_far, rej_near)))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(snrs, 100 * np.array(rej_far), "o-", color=BLUE, label="far unknowns rejected (GFSK, AM, FM)")
    ax.plot(snrs, 100 * np.array(rej_near), "s-", color=ORANGE, label="near unknowns rejected (8-PSK, 64-QAM, other OFDM)")
    ax.plot(snrs, 100 * np.array(kept_k), "^--", color=INK, label="known signals kept")
    ax.plot(snrs, 100 * np.array(acc_k), ":", color=MUTED, label="accuracy of the known signals kept")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("%")
    ax.set_ylim(-3, 103)
    ax.set_title("Mahalanobis open-set detector vs SNR (thresholds fixed, 95% of known kept overall)", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    save(fig, run / "vs_snr.png")

    # ---- H4: other features --------------------------------------------------------------------------
    other = {}
    LF = [layer_features(model, X) for X in (Xf, Xv, Xk, Xu)]
    for j in range(len(model.features)):
        f = shared_maha(LF[0][j], yf)
        other[f"conv block {j + 1}, Mahalanobis"] = tuple(f(L[j]) for L in LF[1:])
    CF = [cumulants(X) for X in (Xf, Xv, Xk, Xu)]
    g = class_gauss(CF[0], yf)
    cum = tuple(g(F) for F in CF[1:])
    other["cumulant statistics, per-class Gaussian"] = cum
    cv, ck, cu = det.scores(model, Xv)[best], S_k[best], S_u[best]
    z = lambda x, ref: (x - np.median(ref)) / np.std(ref)
    other["CNN Mahalanobis AND cumulants (both must accept)"] = (np.minimum(z(cv, cv), z(cum[0], cum[0])),
                                                               np.minimum(z(ck, cv), z(cum[1], cum[0])),
                                                               np.minimum(z(cu, cv), z(cum[2], cum[0])))
    summary["other_features"] = {}
    for name, (a_v, a_k, a_u) in other.items():
        t = np.quantile(a_v, 1 - osr.TARGET_TPR)
        r = {"auroc_snr_ge_0": osr.auroc(a_k[hk], a_u[hu]),
             "accepted_by_kind_snr_ge_0": {k: float(np.mean(a_u[hu & (ku == i)] >= t)) for i, k in enumerate(kinds)}}
        summary["other_features"][name] = r
        rows += [{"part": "other_features", "score": name, "kind": k, "accepted_snr_ge_0": v}
                 for k, v in r["accepted_by_kind_snr_ge_0"].items()]
        print(f"  {name:48s} AUROC {r['auroc_snr_ge_0']:.3f}  " +
              " ".join(f"{k} {v:.2f}" for k, v in r["accepted_by_kind_snr_ge_0"].items()))

    # ---- a band with unknown emitters (one illustrative example, not a measurement) ------------------
    rng = np.random.default_rng(21)
    band = [("qpsk", 12), ("gfsk", 12), ("ofdm", 12), ("fm", 12), ("8psk", 15), ("qam16", 15), ("am", 12), (None, 0)]
    n_nat, n_ch = 1024, len(band)
    slot = awgn(n_nat * n_ch, 1.0, rng)
    for c, (kind, snr) in enumerate(band):
        if kind is None:
            continue
        x = clean_signal(kind, n_nat, rng) if kind in CLASSES else osr.unknown_clean(kind, n_nat, rng)
        x = x * np.exp(1j * rng.uniform(0, 2 * np.pi))
        slot += sp.to_wideband(np.sqrt(10 ** (snr / 10) / n_ch) * x, c, n_ch)
    win = cl.complex_to_input(sp.channelize_all(slot, n_ch)[:, :n_nat])
    Sb = det.scores(model, win)
    closed_lab = [CLASSES[p] for p in Sb["pred"]]
    open_lab = [CLASSES[p] if Sb[best][i] >= th[best] else "UNKNOWN" for i, p in enumerate(Sb["pred"])]
    summary["band_example"] = [{"channel": c, "true": kind or "noise", "closed_set": cs, "open_set": os_}
                               for c, ((kind, _), cs, os_) in enumerate(zip(band, closed_lab, open_lab))]
    f, p = sp.psd(slot, nperseg=1024)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(p) + 1e-12), color=INK, lw=0.8)
    centres = sp.channel_centres(n_ch)
    top = 10 * np.log10(p.max()) + 4
    for c, ((kind, _), cs, os_) in enumerate(zip(band, closed_lab, open_lab)):
        true = kind or "noise"
        unknown_true = true in osr.UNKNOWN_KINDS
        good = (os_ == "UNKNOWN") if unknown_true else (os_ == true)
        ax.text(centres[c], top + 9, f"true: {true}", ha="center", fontsize=8, color=INK)
        ax.text(centres[c], top + 5, f"closed: {cs}", ha="center", fontsize=8, color=MUTED)
        ax.text(centres[c], top + 1, f"open: {os_}", ha="center", fontsize=8, fontweight="bold",
                color=BLUE if good else ORANGE)
        ax.axvline(centres[c] - 0.5 / n_ch, color=GRID, lw=0.8)
    ax.set_ylim(top - 45, top + 13)
    ax.set_xlabel("frequency (cycles/sample of the wideband recording)")
    ax.set_ylabel("power (dB)")
    ax.set_title("One band, three unknown emitters: blue = open-set answer right, orange = wrong (illustration, seed 21)",
                 fontsize=10)
    save(fig, run / "band_example.png")
    print("  band: " + ", ".join(f"{d['true']}->{d['open_set']}" for d in summary["band_example"]))

    for part in dict.fromkeys(r["part"] for r in rows):
        ex.save_csv(run, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run, summary)
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
