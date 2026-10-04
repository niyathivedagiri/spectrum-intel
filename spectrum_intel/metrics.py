"""Communication metrics and their textbook references.

Error accounting is kept separate on purpose:
  raw bit errors      errors in demodulated bits (before any correction)
  corrected errors    errors fixed by forward error correction (later phase)
  uncorrectable       errors left after correction
  packet failures     packets rejected by the CRC or never detected
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from spectrum_intel.transmitter import SPS


def q_function(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    n = min(len(tx_bits), len(rx_bits))
    return int(np.sum(np.asarray(tx_bits[:n]) != np.asarray(rx_bits[:n])) + abs(len(tx_bits) - len(rx_bits)))


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return bit_errors(tx_bits, rx_bits) / max(len(tx_bits), 1)


def ser(tx_idx: np.ndarray, rx_idx: np.ndarray) -> float:
    n = min(len(tx_idx), len(rx_idx))
    return float(np.mean(np.asarray(tx_idx[:n]) != np.asarray(rx_idx[:n]))) if n else float("nan")


def esn0_from_snr(snr_db, sps: int = SPS):
    """Es/N0 at the matched-filter output from the per-sample SNR."""
    return np.asarray(snr_db) + 10 * np.log10(sps)


def ebn0_from_snr(snr_db, bits_per_symbol: int, sps: int = SPS):
    return esn0_from_snr(snr_db, sps) - 10 * np.log10(bits_per_symbol)


def theory_ber(mod: str, ebn0_db):
    """Gray-coded BER in AWGN (exact for BPSK/QPSK, standard approximation for 16-QAM)."""
    eb = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    if mod in ("bpsk", "qpsk"):
        return q_function(np.sqrt(2 * eb))
    if mod == "qam16":
        return 0.75 * q_function(np.sqrt(0.8 * eb))
    raise ValueError(mod)


def theory_ser(mod: str, esn0_db):
    es = 10 ** (np.asarray(esn0_db, dtype=float) / 10)
    if mod == "bpsk":
        return q_function(np.sqrt(2 * es))
    if mod == "qpsk":
        p = q_function(np.sqrt(es))
        return 2 * p - p ** 2
    if mod == "qam16":
        p = 1.5 * q_function(np.sqrt(es / 5))
        return 1 - (1 - p) ** 2
    raise ValueError(mod)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an error rate k/n (works even when k = 0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))
