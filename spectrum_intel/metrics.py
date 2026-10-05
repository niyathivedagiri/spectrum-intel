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

from spectrum_intel.transmitter import BETA, SPS


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


# --------------------------------------------------------------------------
# Image quality (received vs original)
# --------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (inf if identical). ~30 dB+ looks clean, <20 dB visibly damaged."""
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))


def _luma(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, float)
    return img if img.ndim == 2 else img[..., :3] @ np.array([0.299, 0.587, 0.114])


def ssim(a: np.ndarray, b: np.ndarray, peak: float = 255.0, sigma: float = 1.5) -> float:
    """Structural similarity (Wang et al. 2004) on luminance, Gaussian window. 1 = identical structure."""
    from scipy.ndimage import gaussian_filter
    x, y = _luma(a), _luma(b)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mx ** 2
    syy = gaussian_filter(y * y, sigma) - my ** 2
    sxy = gaussian_filter(x * y, sigma) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(np.mean(s))


def pixel_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixel values that differ at all."""
    return float(np.mean(np.asarray(a) != np.asarray(b)))


# --------------------------------------------------------------------------
# Link-level scorecard
# --------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE_HZ = 1e6      # 1 MS/s per channel (as in the LEO model) -> 125 kBd, 169 kHz occupied


def occupied_bandwidth_hz(sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
    """Root-raised-cosine bandwidth: symbol rate x (1 + roll-off)."""
    return sample_rate_hz / SPS * (1 + BETA)


def link_report(result, sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> dict:
    """All link metrics for one LinkResult (one message sent as a series of packets).

    Rates use airtime = total transmitted samples / sample rate, i.e. everything the
    radio spends on air (preamble, header, pilots, packet header, CRC, filter tails).
      phy_rate_bps         modulation bit rate while payload symbols are on air
      throughput_bps       user data bits sent / airtime (what the link offers)
      goodput_bps          user data bits DELIVERED CORRECTLY (CRC pass) / airtime
      spectral_eff         goodput / occupied bandwidth  (bit/s/Hz)
      latency_frame_ms     airtime of one frame (one-way, no retransmission)
      latency_message_ms   airtime until the last packet of the message has been sent
    Error accounting: raw bit errors (before any correction), symbol errors, packet failures.
    """
    p = result.packets
    if not p:
        return {}
    from spectrum_intel.modulation import bits_per_symbol
    mod = p[0].mod
    k = bits_per_symbol(mod)
    airtime_s = sum(x.n_samples for x in p) / sample_rate_hz
    demod = [x for x in p if x.bit_errors is not None]
    bits_demod = sum(x.n_bits for x in demod)
    bit_err = sum(x.bit_errors for x in demod)
    sym = [x for x in p if x.symbol_errors is not None]
    n_sym = sum(x.n_symbols for x in sym)
    data_bits_sent = 8 * sum(x.data_bytes for x in p)
    data_bits_ok = 8 * sum(x.data_bytes for x in p if x.crc_ok)
    n = len(p)
    goodput = data_bits_ok / airtime_s
    return {
        "mod": mod, "packets": n,
        "frame_detection_rate": sum(x.detected for x in p) / n,
        "header_success_rate": sum(x.header_ok for x in p) / n,
        "packet_success_rate": sum(x.crc_ok for x in p) / n,
        "packet_error_rate": 1 - sum(x.crc_ok for x in p) / n,
        "raw_bit_errors": bit_err, "bits_demodulated": bits_demod,
        "ber": bit_err / bits_demod if bits_demod else float("nan"),
        "ser": sum(x.symbol_errors for x in sym) / n_sym if n_sym else float("nan"),
        "packets_lost": sum(not x.header_ok for x in p),
        "packets_crc_failed": sum(x.header_ok and not x.crc_ok for x in p),
        "phy_rate_bps": sample_rate_hz / SPS * k,
        "throughput_bps": data_bits_sent / airtime_s,
        "goodput_bps": goodput,
        "spectral_efficiency_bps_hz": goodput / occupied_bandwidth_hz(sample_rate_hz),
        "overhead_fraction": 1 - (data_bits_sent / airtime_s) / (sample_rate_hz / SPS * k),
        "latency_frame_ms": float(1e3 * np.mean([x.n_samples for x in p]) / sample_rate_hz),
        "latency_message_ms": 1e3 * airtime_s,
        "snr_est_db_mean": float(np.nanmean([x.snr_est_db for x in p if x.detected])) if any(
            x.detected for x in p) else float("nan"),
    }


# --------------------------------------------------------------------------
# Sensing and access metrics (one place for the whole project)
# --------------------------------------------------------------------------
def detection_probability(decisions: np.ndarray, truth: np.ndarray) -> float:
    """P(say 'occupied' | occupied)."""
    d, t = np.asarray(decisions, bool), np.asarray(truth, bool)
    return float(np.mean(d[t])) if t.any() else float("nan")


def false_alarm_rate(decisions: np.ndarray, truth: np.ndarray) -> float:
    """P(say 'occupied' | empty)."""
    d, t = np.asarray(decisions, bool), np.asarray(truth, bool)
    return float(np.mean(d[~t])) if (~t).any() else float("nan")


def collision_rate(choice: np.ndarray, truth_occupied: np.ndarray, skip_first: bool = True) -> float:
    """Fraction of slots in which the chosen channel was in use by a licensed user."""
    slots = np.arange(len(choice))
    hit = np.asarray(truth_occupied)[slots, np.asarray(choice)]
    return float(np.mean(hit[1:] if skip_first else hit))
