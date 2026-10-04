"""Signal generator: the radio traffic the system has to sense and classify.

Six classes of complex baseband I/Q signal:

    bpsk, qpsk, qam16   single-carrier digital modulations with root-raised-cosine pulses
    ofdm                multicarrier signal (QPSK on each subcarrier, with cyclic prefix)
    noise               receiver noise only (an empty channel)
    interference        non-communication emission: a CW tone or a chirp sweep

Frequencies are *normalised*: cycles per sample, i.e. a fraction of the sample
rate fs. The usable complex band is therefore -0.5 ... +0.5.
Every clean signal is scaled to unit average power.
"""
from __future__ import annotations

import numpy as np

from spectrum_intel.dsp import add_awgn, awgn, frequency_shift, normalize_power

CLASSES = ("bpsk", "qpsk", "qam16", "ofdm", "noise", "interference")
MODULATIONS = ("bpsk", "qpsk", "qam16")
BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2, "qam16": 4}

# Gray-coded amplitude levels for one axis of 16-QAM: neighbours differ by one bit.
_QAM16_LEVELS = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}


# --------------------------------------------------------------------------
# Bits -> symbols
# --------------------------------------------------------------------------
def bits_to_symbols(bits: np.ndarray, mod: str) -> np.ndarray:
    """Map a 0/1 bit array to unit-average-power, Gray-coded constellation symbols."""
    bits = np.asarray(bits, dtype=int)
    k = BITS_PER_SYMBOL[mod]
    if len(bits) % k:
        raise ValueError(f"{mod} needs a multiple of {k} bits, got {len(bits)}")
    b = bits.reshape(-1, k)
    if mod == "bpsk":
        return (1 - 2 * b[:, 0]).astype(complex)
    if mod == "qpsk":
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)
    if mod == "qam16":
        i = np.array([_QAM16_LEVELS[(p, q)] for p, q in b[:, :2]])
        q = np.array([_QAM16_LEVELS[(p, q)] for p, q in b[:, 2:]])
        return (i + 1j * q) / np.sqrt(10)
    raise ValueError(f"unknown modulation {mod!r}")


def constellation(mod: str) -> np.ndarray:
    """All ideal constellation points of a modulation."""
    k = BITS_PER_SYMBOL[mod]
    all_bits = ((np.arange(2**k)[:, None] >> np.arange(k)[::-1]) & 1).ravel()
    return bits_to_symbols(all_bits, mod)


def random_symbols(mod: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random symbols of the given modulation."""
    bits = rng.integers(0, 2, n * BITS_PER_SYMBOL[mod])
    return bits_to_symbols(bits, mod)


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------
def rrc_taps(beta: float = 0.35, sps: int = 8, span: int = 8) -> np.ndarray:
    """Root-raised-cosine filter taps, unit energy.

    beta: roll-off (0..1), sps: samples per symbol, span: filter length in symbols.
    """
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps  # time in symbol periods
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))


def pulse_shape(symbols: np.ndarray, sps: int = 8, beta: float = 0.35, span: int = 8) -> np.ndarray:
    """Upsample symbols by sps (insert zeros) and filter with an RRC pulse."""
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    return np.convolve(up, rrc_taps(beta, sps, span))


def single_carrier(mod: str, n_samples: int, rng: np.random.Generator,
                   sps: int = 8, beta: float = 0.35, span: int = 8) -> np.ndarray:
    """A random stretch of a pulse-shaped BPSK / QPSK / 16-QAM transmission.

    Occupied bandwidth = (1 + beta) / sps  (as a fraction of fs).
    The window starts at a random point inside a symbol, like a real receiver.
    """
    n_sym = n_samples // sps + 2 * span + 2
    y = pulse_shape(random_symbols(mod, n_sym, rng), sps, beta, span)
    start = span * sps + int(rng.integers(0, sps))
    return normalize_power(y[start:start + n_samples])


# --------------------------------------------------------------------------
# OFDM
# --------------------------------------------------------------------------
def active_subcarriers(n_fft: int = 128, n_active: int = 20) -> np.ndarray:
    """FFT bin indices used for data: n_active/2 each side of DC, DC left empty."""
    half = n_active // 2
    ks = np.r_[-half:0, 1:half + 1]
    return ks % n_fft


def ofdm_symbol(data: np.ndarray, n_fft: int = 128, cp_len: int = 32,
                bins: np.ndarray | None = None) -> np.ndarray:
    """One OFDM symbol: place data on subcarriers, IFFT, prepend cyclic prefix."""
    bins = active_subcarriers(n_fft, len(data)) if bins is None else bins
    X = np.zeros(n_fft, dtype=complex)
    X[bins] = data
    x = np.fft.ifft(X)
    return np.concatenate([x[-cp_len:], x])


def ofdm(n_samples: int, rng: np.random.Generator, n_fft: int = 128, n_active: int = 20,
         cp_len: int = 32, subcarrier_mod: str = "qpsk") -> np.ndarray:
    """A random stretch of an OFDM transmission.

    Subcarrier spacing = 1/n_fft, occupied bandwidth ~ (n_active + 1)/n_fft.
    """
    bins = active_subcarriers(n_fft, n_active)
    sym_len = n_fft + cp_len
    n_sym = n_samples // sym_len + 2
    stream = np.concatenate([
        ofdm_symbol(random_symbols(subcarrier_mod, n_active, rng), n_fft, cp_len, bins)
        for _ in range(n_sym)
    ])
    start = int(rng.integers(0, sym_len))
    return normalize_power(stream[start:start + n_samples])


# --------------------------------------------------------------------------
# Interference
# --------------------------------------------------------------------------
def interference(n_samples: int, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
    """A continuous-wave tone or a linear chirp (frequency sweep), unit power."""
    kind = kind or str(rng.choice(["tone", "chirp"]))
    n = np.arange(n_samples)
    if kind == "tone":
        f = rng.uniform(-0.4, 0.4)
        phase = 2 * np.pi * f * n
    elif kind == "chirp":
        f0 = rng.uniform(-0.4, 0.0)
        f1 = rng.uniform(0.0, 0.4)
        if rng.random() < 0.5:
            f0, f1 = f1, f0                      # sweep up or down
        phase = 2 * np.pi * (f0 * n + (f1 - f0) * n**2 / (2 * n_samples))
    else:
        raise ValueError(f"unknown interference kind {kind!r}")
    return np.exp(1j * phase)


# --------------------------------------------------------------------------
# One entry point for every class
# --------------------------------------------------------------------------
def clean_signal(label: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Noise-free signal of the given class (unit power). 'noise' returns zeros."""
    if label in MODULATIONS:
        return single_carrier(label, n_samples, rng)
    if label == "ofdm":
        return ofdm(n_samples, rng)
    if label == "interference":
        return interference(n_samples, rng)
    if label == "noise":
        return np.zeros(n_samples, dtype=complex)
    raise ValueError(f"unknown class {label!r}; choose from {CLASSES}")


def generate(label: str, n_samples: int, snr_db: float,
             rng: np.random.Generator | None = None, cfo: float = 0.0) -> np.ndarray:
    """Received signal of one class at the given SNR.

    A random carrier phase is applied (the receiver doesn't know it),
    optionally a carrier frequency offset `cfo` (cycles/sample, e.g. Doppler),
    then AWGN is added. The 'noise' class is noise only.
    """
    rng = rng or np.random.default_rng()
    if label == "noise":
        return awgn(n_samples, 1.0, rng)
    x = clean_signal(label, n_samples, rng) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    if cfo:
        x = frequency_shift(x, cfo, 1.0)
    return add_awgn(x, snr_db, rng)
