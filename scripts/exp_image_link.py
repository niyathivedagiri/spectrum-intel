"""Experiment: how does image quality degrade with SNR, and how should a receiver treat damaged packets?

Hypotheses
  H1  Raw-pixel images degrade gracefully with SNR, while a compressed PNG of the
      same image is all-or-nothing (needs every packet intact).
  H2  For images, keeping CRC-failed packets (error-tolerant mode) gives higher
      quality than discarding them, because a few wrong bits damage a few pixels,
      whereas a discarded packet loses 64 bytes entirely.
Controlled comparison: strict and tolerant receivers see the identical channel
realisation (same seed); only the treatment of damaged packets differs.

    python scripts/exp_image_link.py        # ~3 min
Output: results/experiments/image_link/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel import link
from spectrum_intel import metrics as m
from spectrum_intel import payload as pl
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, save

COL = {"bpsk": BLUE, "qpsk": ORANGE, "qam16": AQUA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    snrs = np.arange(-4, 15, 2.0)
    img = pl.load_image(None, a.size)
    data, png = pl.image_to_bytes(img), pl.png_bytes(img)
    run = ex.new_run("image_link", {"image": "grace_hopper (matplotlib sample)", "size": a.size,
                                    "raw_bytes": len(data), "png_bytes": len(png), "trials": a.trials,
                                    "snr_db": snrs.tolist(), "packet_payload_bytes": 64}, a.seed)
    rows, gallery = [], {}
    for mod in LINK_MODULATIONS:
        for snr in snrs:
            for t in range(a.trials):
                seed = a.seed * 100_000 + int(snr * 10) * 100 + t * 10 + LINK_MODULATIONS.index(mod)
                res = {}
                for mode, keep in [("strict", False), ("tolerant", True)]:
                    r = link.send(data, mod, snr, np.random.default_rng(seed), keep_corrupted=keep, fill=128)
                    im, _ = pl.bytes_to_image(r.received, img.shape)
                    res[mode] = (r, im)
                    if mod == "qpsk" and t == 0:
                        gallery[(mode, float(snr))] = im
                rp = link.send(png, mod, snr, np.random.default_rng(seed + 1))
                dec = pl.decode_png(rp.received)
                r_s, im_s = res["strict"]
                r_t, im_t = res["tolerant"]
                rows.append({
                    "mod": mod, "snr_db": float(snr), "trial": t, "seed": seed,
                    "raw_ber": r_s.raw_ber(), "packet_success": r_s.packet_success_rate,
                    "strict_psnr": m.psnr(img, im_s), "strict_ssim": m.ssim(img, im_s),
                    "strict_pixels_wrong": m.pixel_error_rate(img, im_s), "strict_lost": len(r_s.missing),
                    "tolerant_psnr": m.psnr(img, im_t), "tolerant_ssim": m.ssim(img, im_t),
                    "tolerant_pixels_wrong": m.pixel_error_rate(img, im_t), "tolerant_lost": len(r_t.missing),
                    "tolerant_damaged_kept": len(r_t.corrupted),
                    "png_exact": bool(dec is not None and np.array_equal(dec, img)), "png_lost": len(rp.missing),
                })
            last = [r for r in rows if r["mod"] == mod and r["snr_db"] == snr]
            print(f"  {mod:5s} SNR {snr:5.1f}  PSNR strict {np.mean([min(r['strict_psnr'], 60) for r in last]):5.1f}"
                  f"  tolerant {np.mean([min(r['tolerant_psnr'], 60) for r in last]):5.1f} dB"
                  f"  PNG ok {np.mean([r['png_exact'] for r in last]):.0%}")
    ex.save_csv(run, "image_quality.csv", rows)

    def agg(mod, key, cap=None):
        out = []
        for s in snrs:
            v = [r[key] for r in rows if r["mod"] == mod and r["snr_db"] == s]
            v = [min(x, cap) for x in v] if cap else v
            out.append(float(np.mean(v)))
        return np.array(out)

    # ---- PSNR and SSIM vs SNR --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for mod in LINK_MODULATIONS:
        axes[0].plot(snrs, agg(mod, "tolerant_psnr", 60), "o-", color=COL[mod], ms=4, label=f"{mod.upper()} tolerant")
        axes[0].plot(snrs, agg(mod, "strict_psnr", 60), "s--", color=COL[mod], ms=3, alpha=0.7,
                     label=f"{mod.upper()} strict")
        axes[1].plot(snrs, agg(mod, "tolerant_ssim"), "o-", color=COL[mod], ms=4)
        axes[1].plot(snrs, agg(mod, "strict_ssim"), "s--", color=COL[mod], ms=3, alpha=0.7)
    axes[0].axhline(30, color=MUTED, lw=0.8, ls=":")
    axes[0].text(snrs[0], 31, "~30 dB: hard to see damage", fontsize=8, color=MUTED)
    axes[0].set_ylabel("PSNR (dB, capped at 60 = perfect)")
    axes[1].set_ylabel("SSIM (1 = identical structure)")
    for ax in axes:
        ax.set_xlabel("SNR per sample (dB)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("Image fidelity vs SNR (solid: keep damaged packets, dashed: drop them)", fontsize=10)
    axes[1].set_title("Structural similarity vs SNR", fontsize=10)
    save(fig, run / "image_quality_vs_snr.png")

    # ---- PNG all-or-nothing vs raw graceful ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for mod in LINK_MODULATIONS:
        ax.plot(snrs, 100 * agg(mod, "png_exact"), "o-", color=COL[mod], ms=4, label=f"{mod.upper()}: PNG decoded")
        ax.plot(snrs, 100 * agg(mod, "packet_success"), ":", color=COL[mod], lw=1.2,
                label=f"{mod.upper()}: packets passing CRC")
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("%")
    ax.set_title("A compressed PNG needs every packet: it fails until packet loss is ~0")
    ax.legend(fontsize=7, ncol=2)
    save(fig, run / "png_all_or_nothing.png")

    # ---- gallery: QPSK, strict vs tolerant ------------------------------------------------
    show = [s for s in [-2.0, 0.0, 2.0, 4.0, 6.0] if ("strict", s) in gallery]
    fig, axes = plt.subplots(2, len(show) + 1, figsize=(2.3 * (len(show) + 1), 5.6),
                             gridspec_kw={"hspace": 0.35})
    for row, mode in enumerate(["strict", "tolerant"]):
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("original" if row == 0 else "", fontsize=9)
        axes[row, 0].set_ylabel("drop damaged" if mode == "strict" else "keep damaged", fontsize=9)
        for col, s in enumerate(show, start=1):
            im = gallery[(mode, s)]
            axes[row, col].imshow(im)
            axes[row, col].set_title(f"{s:g} dB  PSNR {min(m.psnr(img, im), 60):.0f}", fontsize=8, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("QPSK image link: what arrives at each SNR", fontsize=11)
    save(fig, run / "qpsk_gallery.png")

    def first_snr(mod, key, thr):
        v = agg(mod, key, 60)
        ok = np.where(v >= thr)[0]
        return float(snrs[ok[0]]) if len(ok) else None

    summary = {mod: {"snr_for_psnr30_tolerant": first_snr(mod, "tolerant_psnr", 30),
                     "snr_for_psnr30_strict": first_snr(mod, "strict_psnr", 30),
                     "snr_for_png_always_ok": first_snr(mod, "png_exact", 1.0),
                     "mean_psnr_gain_tolerant_db_where_lossy": float(np.mean(
                         [min(r["tolerant_psnr"], 60) - min(r["strict_psnr"], 60) for r in rows
                          if r["mod"] == mod and r["strict_lost"] > 0]) if any(
                         r["strict_lost"] > 0 for r in rows if r["mod"] == mod) else 0.0)}
               for mod in LINK_MODULATIONS}
    ex.save_metrics(run, {"summary": summary})
    for mod, v in summary.items():
        print(f"  {mod}: {v}")
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
