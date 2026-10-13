"""Receiver: complex baseband samples -> bytes.

Steps
 1. Matched filter (the same root-raised-cosine pulse as the transmitter):
    maximises SNR at the symbol instants and, with the transmit filter,
    gives zero intersymbol interference.
 2. Frame + timing synchronisation: correlate with the known preamble at
    symbol spacing; the correlation peak gives the frame start AND the best
    sampling phase in one step.
 3. Frequency offset: under a frequency offset the preamble's phase rotates
    linearly with time. The rotation is measured at lags 1, 4, 16 and 32
    symbols: each longer lag is more precise and the shorter one resolves its
    2*pi ambiguity. After the header is decoded, the slope of the pilot phases
    across the whole frame refines it further.
 4. Channel estimate: the known pilots give the complex gain (amplitude +
    phase) every 17 symbols; it is interpolated linearly in between and
    divided out (one-tap equalisation).
 5. SNR estimate from the preamble residual (used later by adaptive control).
 Optional (equalizer="ls"): before step 4, a short linear equaliser is fitted by least
    squares so that it turns the received preamble back into the known preamble. It
    undoes multipath echoes (intersymbol interference), which the one-tap pilot
    correction cannot. Off by default.
 6. Header (BPSK) -> modulation + length; payload demapping -> bits -> bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate

from spectrum_intel.modulation import bits_per_symbol, demodulate
from spectrum_intel.payload import bits_to_bytes
from spectrum_intel.signals import rrc_taps
from spectrum_intel.transmitter import (BETA, HEADER_BITS, PILOT, PILOT_SPACING, PREAMBLE,
                                        SPAN, SPS, parse_header, stream_length)

# Normalised preamble score needed to declare a frame. Set from measurements:
# noise-only 3,000-sample windows: max over 400 trials 0.549 (99.9th percentile 0.546);
# real frames: minimum 0.69 at -8 dB SNR, 0.57 at -10 dB (where the header fails anyway).
DETECTION_THRESHOLD = 0.58
SYNC_SEGMENTS = 7              # preamble split into 7 x 9 symbols for frequency-robust detection
PILOT_WINDOW = 7               # pilots averaged when estimating the channel (noise vs tracking speed)
EQ_TAPS = 7                    # optional preamble-trained equaliser: +-3 symbols of echo


@dataclass
class RxResult:
    detected: bool
    header_ok: bool = False
    mod: str | None = None
    payload: bytes | None = None
    payload_bits: np.ndarray | None = None
    data_symbols: np.ndarray | None = None       # equalised header+payload symbols
    raw_symbols: np.ndarray | None = None        # matched-filter samples before equalisation
    snr_est_db: float = float("nan")
    esn0_est_db: float = float("nan")
    cfo_est: float = 0.0                         # cycles/sample
    start: int = -1
    sync_metric: float = 0.0
    info: dict = field(default_factory=dict)


def _multi_lag_freq(z: np.ndarray, lags=(1, 4, 16, 32)) -> float:
    """Frequency (rad/symbol) of z = preamble * conj(known preamble), multi-lag estimator."""
    w = 0.0
    for lag in lags:
        if lag >= len(z):
            break
        r = np.sum(z[lag:] * np.conj(z[:-lag]))
        meas = np.angle(r * np.exp(-1j * w * lag)) / lag      # residual, unambiguous given w
        w = w + meas
    return float(w)


def _pilot_slope(stream: np.ndarray, n_data: int) -> float:
    """Residual frequency (rad/symbol) from a straight-line fit to the unwrapped pilot phases."""
    n_blocks = int(np.ceil(n_data / PILOT_SPACING))
    pos = np.arange(n_blocks + 1) * (PILOT_SPACING + 1)
    if len(pos) < 3:
        return 0.0
    g = stream[pos] / PILOT
    ph = np.unwrap(np.angle(g))
    wts = np.abs(g) ** 2
    A = np.vstack([pos, np.ones_like(pos)]).T * np.sqrt(wts)[:, None]
    slope, _ = np.linalg.lstsq(A, ph * np.sqrt(wts), rcond=None)[0]
    return float(slope)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average (shorter at the ends) of complex values."""
    if w <= 1 or len(x) < 2:
        return x
    k = np.ones(min(w, len(x)))
    return np.convolve(x, k, mode="same") / np.convolve(np.ones(len(x)), k, mode="same")


def _gain_track(stream: np.ndarray, n_data: int, window: int = PILOT_WINDOW):
    """Equalise data symbols with the pilot gains: averaged over `window` pilots
    (reduces estimation noise), then interpolated linearly between pilots."""
    n_blocks = int(np.ceil(n_data / PILOT_SPACING))
    step = PILOT_SPACING + 1
    pilot_pos = np.arange(n_blocks + 1) * step
    gains = _smooth(stream[pilot_pos] / PILOT, window)
    data_pos = (np.arange(n_blocks)[:, None] * step + 1 + np.arange(PILOT_SPACING)[None, :]).ravel()[:n_data]
    g = np.interp(data_pos, pilot_pos, gains.real) + 1j * np.interp(data_pos, pilot_pos, gains.imag)
    return stream[data_pos] / g, g


def _apply_fir(sym: np.ndarray, w: np.ndarray) -> np.ndarray:
    """e[n] = sum_k w[k] * sym[n + k - c], c = centre tap (zeros outside the burst)."""
    c = len(w) // 2
    out = np.zeros(len(sym), complex)
    for k, wk in enumerate(w):
        j = k - c
        if j >= 0:
            out[:len(sym) - j] += wk * sym[j:]
        else:
            out[-j:] += wk * sym[:j]
    return out


def ls_equalizer(sym: np.ndarray, n_taps: int = EQ_TAPS, ridge: float = 1e-2) -> np.ndarray:
    """Symbol-spaced FIR equaliser trained on the preamble (regularised least squares).

    Finds w so that sum_k w[k] sym[n+k-c] ~= PREAMBLE[n]. Preamble symbols are known,
    so this is data-aided: no decisions are needed. ridge keeps it stable when the
    channel has a deep notch (the LS solution would otherwise amplify noise).
    """
    c = n_taps // 2
    rows = np.arange(c, len(PREAMBLE))
    A = np.stack([sym[rows + k - c] for k in range(n_taps)], axis=1)
    target = PREAMBLE[rows]
    R = A.conj().T @ A
    lam = ridge * np.real(np.trace(R)) / n_taps
    return np.linalg.solve(R + lam * np.eye(n_taps), A.conj().T @ target)


def _sync(y: np.ndarray):
    """Segmented (non-coherent) preamble correlation.

    Each 9-symbol segment is correlated coherently and the magnitudes are added,
    so a frequency offset that rotates the phase across the whole preamble does
    not destroy the peak. Returns (start index, normalised metric).
    """
    seg = len(PREAMBLE) // SYNC_SEGMENTS
    span = (len(PREAMBLE) - 1) * SPS + 1
    if len(y) < span:
        return -1, 0.0
    total = np.zeros(len(y) - span + 1)
    for s in range(SYNC_SEGMENTS):
        ref = np.zeros((seg - 1) * SPS + 1, complex)
        ref[::SPS] = PREAMBLE[s * seg:(s + 1) * seg]
        c = correlate(y[s * seg * SPS:], ref, mode="valid", method="fft")
        total += np.abs(c[:len(total)])
    mask = np.zeros(span)
    mask[::SPS] = 1
    energy = np.convolve(np.abs(y) ** 2, mask[::-1], mode="valid")[:len(total)]
    metric = total / np.sqrt(np.maximum(energy, 1e-12) * SYNC_SEGMENTS * seg)   # 1.0 = perfect match
    n0 = int(np.argmax(total))
    if metric[n0] < DETECTION_THRESHOLD:
        # The strongest raw peak can sit in the DATA when the preamble arrived in a fade and the
        # signal grew stronger later in the frame (time-varying fading). The normalised metric is
        # level-independent, so its best position is checked as well. This only changes bursts
        # that would otherwise be rejected (AWGN results that passed before are unchanged).
        n1 = int(np.argmax(metric))
        if metric[n1] >= DETECTION_THRESHOLD:
            return n1, float(metric[n1])
    return n0, float(metric[n0])


def receive(rx: np.ndarray, cfo_search: float = 0.0, cfo_step: float = 0.004,
            equalizer: str = "none", eq_taps: int = EQ_TAPS, pilot_window: int = PILOT_WINDOW) -> RxResult:
    """Demodulate one burst.

    cfo_search: if > 0, also try coarse frequency offsets in [-cfo_search, +cfo_search]
    (cycles/sample, e.g. LEO Doppler) and keep the one with the strongest preamble.
    equalizer: "none" (one-tap pilot correction only) or "ls" (preamble-trained FIR first).
    eq_taps: equaliser length in symbols (odd). Longer handles longer echoes but is estimated
    from the same 63 preamble symbols, so it adds more noise on a clean channel.
    pilot_window: pilots averaged per gain estimate. Large = less noise (best for slowly
    changing channels); small = follows fast fading (Doppler) but noisier.
    """
    if equalizer not in ("none", "ls"):
        raise ValueError("equalizer must be 'none' or 'ls'")
    coarse = 0.0
    if cfo_search > 0:
        best = (-1.0, 0.0)
        n = np.arange(len(rx))
        for f in np.arange(-cfo_search, cfo_search + 1e-12, cfo_step):
            _, m = _sync(np.convolve(rx * np.exp(-2j * np.pi * f * n), rrc_taps(BETA, SPS, SPAN)))
            if m > best[0]:
                best = (m, f)
        coarse = best[1]
        rx = rx * np.exp(-2j * np.pi * coarse * np.arange(len(rx)))

    h = rrc_taps(BETA, SPS, SPAN)
    y = np.convolve(rx, h)                                   # matched filter (delay len(h)-1 overall)

    # ---- 2. frame + timing sync -------------------------------------------------
    n0, sync = _sync(y)
    if sync < DETECTION_THRESHOLD:
        return RxResult(False, sync_metric=sync)

    n_avail = (len(y) - n0 - 1) // SPS + 1
    sym = y[n0 + SPS * np.arange(n_avail)]
    pre = sym[:len(PREAMBLE)]

    # ---- 3. frequency offset from the preamble ------------------------------------
    dphi = _multi_lag_freq(pre * np.conj(PREAMBLE))              # rad per symbol
    sym = sym * np.exp(-1j * dphi * np.arange(n_avail))
    eq_w = None
    if equalizer == "ls" and n_avail > len(PREAMBLE) + eq_taps:
        eq_w = ls_equalizer(sym, eq_taps)
        sym = _apply_fir(sym, eq_w)
    pre = sym[:len(PREAMBLE)]

    # ---- 5. gain, phase and SNR from the preamble -----------------------------------
    g0 = np.mean(pre * np.conj(PREAMBLE))
    resid = pre - g0 * PREAMBLE
    noise_var = float(np.mean(np.abs(resid) ** 2))
    esn0 = 10 * np.log10(np.abs(g0) ** 2 / max(noise_var, 1e-15))
    res = RxResult(True, raw_symbols=sym, esn0_est_db=esn0, snr_est_db=esn0 - 10 * np.log10(SPS),
                   cfo_est=coarse + dphi / (2 * np.pi) / SPS, start=n0, sync_metric=sync)

    # ---- 4+6. header ------------------------------------------------------------------
    stream = sym[len(PREAMBLE):]
    n_hdr = HEADER_BITS                                          # BPSK: 1 bit/symbol
    if len(stream) < stream_length(n_hdr):
        return res
    hdr_syms, _ = _gain_track(stream, n_hdr, pilot_window)
    mod, n_bytes, ok = parse_header(demodulate(hdr_syms, "bpsk"))
    res.header_ok = ok
    if not ok:
        return res

    # ---- payload ----------------------------------------------------------------------
    n_pay = int(np.ceil(n_bytes * 8 / bits_per_symbol(mod)))
    n_data = n_hdr + n_pay
    if len(stream) < stream_length(n_data):
        res.header_ok = False
        return res
    resid = _pilot_slope(stream, n_data)                         # refine frequency over the frame
    stream = stream * np.exp(-1j * resid * np.arange(len(stream)))
    dphi += resid
    res.cfo_est = coarse + dphi / (2 * np.pi) / SPS
    data, gains = _gain_track(stream, n_data, pilot_window)
    bits = demodulate(data[n_hdr:], mod)[:n_bytes * 8]
    res.mod = mod
    res.payload_bits = bits
    res.payload = bits_to_bytes(bits)
    res.data_symbols = data
    res.info = {"n_bytes": n_bytes, "n_payload_symbols": n_pay, "pilot_gain_mean": complex(np.mean(gains)),
                "gain_track": gains[::4],                    # receiver's view of the channel over the frame
                "eq_taps": eq_w}
    return res
