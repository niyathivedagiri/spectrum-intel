"""RF channel models applied to complex baseband I/Q samples.

Every effect acts on the actual samples, so it changes what the receiver
recovers. All randomness comes from the caller's numpy Generator (reproducible).

SNR convention (same as the rest of the project): SNR per sample,
    SNR = (average power of the transmitted burst) / (noise power per sample).
With 8 samples per symbol, Es/N0 at the matched-filter output = SNR + 9.03 dB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectrum_intel.dsp import awgn, frequency_shift, signal_power


@dataclass
class ChannelInfo:
    """What the channel actually did (ground truth, for checking the receiver)."""
    snr_db: float
    noise_power: float
    signal_power: float
    phase: float
    delay: int
    cfo: float


def awgn_link(tx: np.ndarray, snr_db: float, rng: np.random.Generator, *,
              phase: float | None = None, delay: int | None = None, cfo: float = 0.0,
              tail: int = 256) -> tuple[np.ndarray, ChannelInfo]:
    """Burst over an AWGN channel with unknown carrier phase, arrival time and frequency offset.

    phase: carrier phase (rad), random if None.
    delay: number of noise-only samples before the burst arrives, random 32..287 if None.
    cfo:   carrier frequency offset in cycles/sample (e.g. Doppler or oscillator error).
    """
    phase = rng.uniform(0, 2 * np.pi) if phase is None else phase
    delay = int(rng.integers(32, 288)) if delay is None else int(delay)
    p_sig = signal_power(tx)
    x = tx * np.exp(1j * phase)
    if cfo:
        x = frequency_shift(x, cfo, 1.0)
    padded = np.concatenate([np.zeros(delay, complex), x, np.zeros(tail, complex)])
    noise_power = p_sig / 10 ** (snr_db / 10)
    rx = padded + awgn(len(padded), noise_power, rng)
    return rx, ChannelInfo(snr_db, noise_power, p_sig, phase, delay, cfo)
