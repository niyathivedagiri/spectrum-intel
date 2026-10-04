"""Symbol mapping and demapping for the data link.

Mapping reuses the Gray-coded, unit-power constellations of signals.py, so the
link transmits exactly the waveforms the classifier was trained on.
Demapping picks the nearest constellation point (maximum-likelihood decision
for AWGN) and returns its bits.
"""
from __future__ import annotations

import numpy as np

from spectrum_intel.signals import BITS_PER_SYMBOL, bits_to_symbols, constellation

LINK_MODULATIONS = ("bpsk", "qpsk", "qam16")


def bits_per_symbol(mod: str) -> int:
    return BITS_PER_SYMBOL[mod]


def modulate(bits: np.ndarray, mod: str) -> np.ndarray:
    """Bits -> symbols. Pads with zeros to a whole number of symbols."""
    k = bits_per_symbol(mod)
    bits = np.asarray(bits, dtype=int)
    pad = (-len(bits)) % k
    return bits_to_symbols(np.concatenate([bits, np.zeros(pad, dtype=int)]), mod)


def _pattern_bits(mod: str) -> np.ndarray:
    k = bits_per_symbol(mod)
    return ((np.arange(2 ** k)[:, None] >> np.arange(k)[::-1]) & 1).astype(np.uint8)


def nearest_index(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Index of the closest constellation point for each received symbol."""
    pts = constellation(mod)
    return np.argmin(np.abs(np.asarray(symbols)[:, None] - pts[None, :]), axis=1)


def demodulate(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Hard-decision symbols -> bits (MSB first per symbol)."""
    return _pattern_bits(mod)[nearest_index(symbols, mod)].ravel()


def symbol_indices(bits: np.ndarray, mod: str) -> np.ndarray:
    """Constellation index sent for each group of bits (for symbol error rate)."""
    k = bits_per_symbol(mod)
    bits = np.asarray(bits, dtype=int)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % k, dtype=int)]).reshape(-1, k)
    return bits @ (1 << np.arange(k)[::-1])
