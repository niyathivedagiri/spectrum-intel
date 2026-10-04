"""Energy detector: the classical baseline for "is this channel in use?".

Test statistic: average power of N complex samples,  T = mean(|x|^2).

With noise only (complex Gaussian, power s2), N*T/s2 follows a Gamma(N, 1)
distribution. So for a chosen false-alarm probability Pfa the threshold is

    lambda = s2 * gammaincinv(N, 1 - Pfa) / N

This is the Neyman-Pearson-style rule: fix how often you may cry wolf on an
empty channel, then catch as many real signals as possible.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaincc, gammaincinv


def energy(x: np.ndarray) -> np.ndarray:
    """Average power along the last axis (works for one window or a batch)."""
    return np.mean(np.abs(x) ** 2, axis=-1)


def threshold(noise_power: float, n_samples: int, pfa: float) -> float:
    """Detection threshold for the given noise power, window length and false-alarm rate."""
    return noise_power * gammaincinv(n_samples, 1 - pfa) / n_samples


def theoretical_pd(snr_db, n_samples: int, pfa: float) -> np.ndarray:
    """Approximate probability of detection for a signal at snr_db (Gaussian-like signal model)."""
    snr = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    lam = gammaincinv(n_samples, 1 - pfa)            # threshold in units of noise power x N
    return gammaincc(n_samples, lam / (1 + snr))


class EnergyDetector:
    """Decides 'occupied' when a window's average power exceeds the threshold."""

    def __init__(self, noise_power: float, n_samples: int, pfa: float = 0.01):
        self.noise_power = noise_power
        self.n_samples = n_samples
        self.pfa = pfa
        self.threshold = threshold(noise_power, n_samples, pfa)

    def statistic(self, x: np.ndarray) -> np.ndarray:
        return energy(x)

    def detect(self, x: np.ndarray) -> np.ndarray:
        """True where the window is judged occupied. x: (..., n_samples)."""
        return energy(x) > self.threshold
