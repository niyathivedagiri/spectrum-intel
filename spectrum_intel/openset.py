"""Open-set recognition: let the classifier answer "unknown" instead of forcing one of its 6 classes.

A classifier trained on 6 classes (closed set) must put every input into one of them, even a
signal type it has never seen. In real spectrum monitoring new emitters appear all the time, so
the system needs a reject option: "this does not look like anything I was trained on".

Scores (higher = more like the known classes), all computed from the trained CNN, no retraining:
  msp          maximum softmax probability (the usual "confidence"), the standard baseline
  max_logit    largest raw class score before the softmax (the softmax hides how large it is)
  energy       logsumexp of the logits (Liu et al. 2020)
  mahalanobis  minus the distance of the CNN's penultimate features (64 values) to the nearest
               class centre, with one covariance shared by all classes (Lee et al. 2018). The
               centres and covariance are fitted on known TRAINING data only.
  knn          minus the distance to the 10th nearest known TRAINING example, features scaled to
               unit length (Sun et al. 2022).

Threshold: chosen on known validation data only so that 95% of known signals are accepted. The
unknown signals are never used to fit or tune anything.

Unknown signal types (never in training), in two groups:
  near: 8psk, 64qam, ofdm_wide (a different OFDM: 52 of 64 subcarriers, short cyclic prefix)
  far:  gfsk (frequency shift keying), am, fm (analog broadcast-style signals)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from spectrum_intel.classifier import complex_to_input
from spectrum_intel.dsp import add_awgn, normalize_power
from spectrum_intel.signals import CLASSES, ofdm, pulse_shape

UNKNOWN = -1
UNKNOWN_NEAR = ("8psk", "64qam", "ofdm_wide")
UNKNOWN_FAR = ("gfsk", "am", "fm")
UNKNOWN_KINDS = UNKNOWN_NEAR + UNKNOWN_FAR
SCORES = ("msp", "max_logit", "energy", "mahalanobis", "knn")
KNN_K = 10
TARGET_TPR = 0.95


# --------------------------------------------------------------------------
# Unknown signal types
# --------------------------------------------------------------------------
def _psk8(n: int, rng) -> np.ndarray:
    return np.exp(2j * np.pi * rng.integers(0, 8, n) / 8)


def _qam64(n: int, rng) -> np.ndarray:
    lv = np.arange(-7, 8, 2)
    s = rng.choice(lv, n) + 1j * rng.choice(lv, n)
    return s / np.sqrt(42.0)                                  # unit average power


def _single_carrier(symbols_fn, n_samples: int, rng, sps: int = 8, beta: float = 0.35, span: int = 8):
    n_sym = n_samples // sps + 2 * span + 2
    y = pulse_shape(symbols_fn(n_sym, rng), sps, beta, span)
    start = span * sps + int(rng.integers(0, sps))
    return y[start:start + n_samples]


def _gfsk(n_samples: int, rng, sps: int = 8, h: float = 0.5, bt: float = 0.5) -> np.ndarray:
    """Gaussian-filtered continuous-phase FSK (as in Bluetooth): constant envelope."""
    n_sym = n_samples // sps + 4
    nrz = np.repeat(2 * rng.integers(0, 2, n_sym) - 1.0, sps)
    t = np.arange(-2 * sps, 2 * sps + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    freq = np.convolve(nrz, g / g.sum(), mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps
    start = int(rng.integers(0, 2 * sps))
    return np.exp(1j * phase[start:start + n_samples])


def _message(n_samples: int, rng, n_tones: int = 3) -> np.ndarray:
    """A band-limited 'audio' message: a few low-frequency tones, peak about 1."""
    n = np.arange(n_samples)
    m = sum(rng.uniform(0.3, 1.0) * np.cos(2 * np.pi * rng.uniform(0.002, 0.02) * n + rng.uniform(0, 2 * np.pi))
            for _ in range(n_tones))
    return m / np.max(np.abs(m))


def _am(n_samples: int, rng) -> np.ndarray:
    """AM with carrier (broadcast AM), modulation depth 0.5-0.9."""
    return (1 + rng.uniform(0.5, 0.9) * _message(n_samples, rng)).astype(complex)


def _fm(n_samples: int, rng) -> np.ndarray:
    """Analog FM, peak deviation 0.03-0.06 cycles/sample: constant envelope."""
    dev = rng.uniform(0.03, 0.06)
    return np.exp(2j * np.pi * dev * np.cumsum(_message(n_samples, rng)))


def unknown_clean(kind: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "8psk":
        x = _single_carrier(_psk8, n_samples, rng)
    elif kind == "64qam":
        x = _single_carrier(_qam64, n_samples, rng)
    elif kind == "ofdm_wide":
        x = ofdm(n_samples, rng, n_fft=64, n_active=52, cp_len=16)
    elif kind == "gfsk":
        x = _gfsk(n_samples, rng)
    elif kind == "am":
        x = _am(n_samples, rng)
    elif kind == "fm":
        x = _fm(n_samples, rng)
    else:
        raise ValueError(f"unknown kind must be one of {UNKNOWN_KINDS}")
    return normalize_power(x)


def generate_unknown(kind: str, n_samples: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Same channel as signals.generate: random carrier phase, then AWGN at the given SNR."""
    x = unknown_clean(kind, n_samples, rng) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return add_awgn(x, snr_db, rng)


def make_unknown_set(per_kind: int, snrs, seed: int, n_samples: int = 1024, kinds=UNKNOWN_KINDS):
    """Returns (X network input, kind index, snr) for the unknown types."""
    rng = np.random.default_rng(seed)
    X, k, s = [], [], []
    for ki, kind in enumerate(kinds):
        for _ in range(per_kind):
            snr = float(rng.choice(snrs))
            X.append(generate_unknown(kind, n_samples, snr, rng))
            k.append(ki)
            s.append(snr)
    return complex_to_input(np.array(X)), np.array(k), np.array(s, float)


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
@torch.no_grad()
def logits_and_features(model, X: np.ndarray, batch_size: int = 512):
    """Logits (n, 6) and penultimate features (n, 64): after the first dense layer + ReLU."""
    model.eval()
    dev = next(model.parameters()).device
    L, F = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
        h = model.features(xb).mean(dim=-1, keepdim=True)
        f = model.head[2](model.head[1](model.head[0](h)))       # Flatten -> Linear -> ReLU
        L.append(model.head[4](f).cpu().numpy())                  # (dropout is off in eval)
        F.append(f.cpu().numpy())
    return np.concatenate(L), np.concatenate(F)


def _unit(F):
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)


def _softmax(L):
    z = L - L.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class OpenSetDetector:
    """Known-vs-unknown scoring on top of a trained SpectrumCNN."""
    means: np.ndarray                     # (6, 64) class centres of the penultimate features
    precision: np.ndarray                 # (64, 64) inverse of the shared covariance
    bank: np.ndarray | None = None        # unit-length training features for the kNN score
    thresholds: dict = field(default_factory=dict)   # score name -> accept if score >= threshold

    @classmethod
    def fit(cls, model, X_train: np.ndarray, y_train: np.ndarray, ridge: float = 1e-3) -> "OpenSetDetector":
        _, F = logits_and_features(model, X_train)
        means = np.stack([F[y_train == c].mean(axis=0) for c in range(len(CLASSES))])
        R = F - means[y_train]
        cov = R.T @ R / len(R)
        cov += ridge * np.trace(cov) / len(cov) * np.eye(len(cov))        # keeps the inverse stable
        return cls(means, np.linalg.inv(cov), _unit(F).astype(np.float32))

    def scores(self, model, X: np.ndarray) -> dict:
        L, F = logits_and_features(model, X)
        P = _softmax(L)
        d = F[:, None, :] - self.means[None]                              # (n, 6, 64)
        maha = np.einsum("nci,ij,ncj->nc", d, self.precision, d)
        m = L.max(axis=1)
        dk = np.sqrt(np.maximum(2 - 2 * _unit(F) @ self.bank.T, 0))       # distances on the unit sphere
        knn = -np.partition(dk, KNN_K - 1, axis=1)[:, KNN_K - 1]
        return {"knn": knn, "msp": P.max(axis=1), "max_logit": m,
                "energy": m + np.log(np.exp(L - m[:, None]).sum(axis=1)),
                "mahalanobis": -maha.min(axis=1),
                "pred": L.argmax(axis=1), "pred_maha": maha.argmin(axis=1)}

    def calibrate(self, model, X_val: np.ndarray, target_tpr: float = TARGET_TPR) -> dict:
        """Thresholds that accept `target_tpr` of KNOWN validation signals (unknowns are not used)."""
        s = self.scores(model, X_val)
        self.thresholds = {k: float(np.quantile(s[k], 1 - target_tpr)) for k in SCORES}
        return self.thresholds

    def predict(self, model, X: np.ndarray, score: str = "mahalanobis") -> np.ndarray:
        """Class index, or UNKNOWN (-1) when the score is below its threshold."""
        s = self.scores(model, X)
        return np.where(s[score] >= self.thresholds[score], s["pred"], UNKNOWN)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, means=self.means, precision=self.precision, bank=self.bank,
                            names=np.array(list(self.thresholds)), values=np.array(list(self.thresholds.values())))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OpenSetDetector":
        d = np.load(path)
        return cls(d["means"], d["precision"], d["bank"], dict(zip([str(n) for n in d["names"]], d["values"].tolist())))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Probability that a random known signal scores higher than a random unknown one (ties = 1/2)."""
    k = np.asarray(known_scores, float)
    u = np.asarray(unknown_scores, float)
    allv = np.concatenate([k, u])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv[np.r_[False, np.diff(allv[order]) == 0]] if len(allv) > 1 else []):
        tie = allv == v
        ranks[tie] = ranks[tie].mean()                                  # average ranks for ties
    return float((ranks[:len(k)].sum() - len(k) * (len(k) + 1) / 2) / (len(k) * len(u)))


def unknown_accepted(unknown_scores: np.ndarray, threshold: float) -> float:
    """Fraction of unknown signals wrongly accepted as known (FPR at the chosen threshold)."""
    return float(np.mean(np.asarray(unknown_scores) >= threshold))
