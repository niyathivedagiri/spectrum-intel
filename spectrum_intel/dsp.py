"""Core DSP utilities used across the spectrum-intelligence pipeline.

All signals are complex baseband I/Q arrays (numpy complex128) sampled at fs Hz.
"""
from __future__ import annotations

import numpy as np


def db(power_ratio):
    """Convert a power ratio to decibels."""
    return 10 * np.log10(power_ratio)


def from_db(value_db):
    """Convert decibels back to a power ratio."""
    return 10 ** (np.asarray(value_db) / 10)


def signal_power(x: np.ndarray) -> float:
    """Average power of a real or complex signal: mean(|x|^2)."""
    return float(np.mean(np.abs(x) ** 2))


def fft_spectrum(x: np.ndarray, fs: float, window: str | None = None):
    """Return (frequencies in Hz, magnitude in dB) centred on 0 Hz.

    window: None (rectangular) or "hann".
    """
    n = len(x)
    w = np.hanning(n) if window == "hann" else np.ones(n)
    X = np.fft.fftshift(np.fft.fft(x * w)) / np.sum(w)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    return f, 20 * np.log10(np.abs(X) + 1e-12)


def tone(freq_hz: float, fs: float, n: int, amplitude: float = 1.0) -> np.ndarray:
    """Complex tone amplitude * exp(j 2π f n / fs)."""
    t = np.arange(n) / fs
    return amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)


def frequency_shift(x: np.ndarray, shift_hz: float, fs: float) -> np.ndarray:
    """Shift a baseband signal by shift_hz (mixing, channel placement, Doppler)."""
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * shift_hz * n / fs)


def awgn(n: int, noise_power: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Complex white Gaussian noise with the given total power (split equally over I and Q)."""
    rng = rng or np.random.default_rng()
    return np.sqrt(noise_power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex AWGN so that signal_power(x) / noise_power = snr_db."""
    noise_power = signal_power(x) / from_db(snr_db)
    return x + awgn(len(x), noise_power, rng)
