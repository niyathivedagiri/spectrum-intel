"""Wideband spectrum: build a multi-channel band, analyse it, and split it into channels.

The monitored band is divided into N_CHANNELS equal channels. The wideband
sample rate is N_CHANNELS times the per-channel ("native") rate, so

    channel c is centred at  (c - (N-1)/2) / N   (fraction of the wideband rate)
    and is 1/N of the band wide.

Each channel can hold one of the signal classes from signals.py, generated at
the native rate (the same format the classifier is trained on), interpolated up
to the wideband rate and shifted to its channel centre. The channelizer does the
reverse, so every channel comes back in exactly the classifier's format.

SNR convention: the wideband noise has power 1, so each channel, once
channelized, carries noise power 1/N. A channel "at 10 dB" therefore has a
signal 10 dB above the noise inside that channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly, spectrogram, welch

from spectrum_intel.dsp import awgn, frequency_shift
from spectrum_intel.signals import CLASSES, clean_signal, rrc_taps

N_CHANNELS = 8
NOISE_LABEL = CLASSES.index("noise")


def channel_centres(n_channels: int = N_CHANNELS) -> np.ndarray:
    """Centre frequency of each channel as a fraction of the wideband sample rate."""
    return (np.arange(n_channels) - (n_channels - 1) / 2) / n_channels


def to_wideband(x_native: np.ndarray, channel: int, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Interpolate a native-rate signal by n_channels and move it to its channel."""
    up = resample_poly(x_native, n_channels, 1)
    return frequency_shift(up, channel_centres(n_channels)[channel], 1.0)


def channel_filter(n_channels: int = N_CHANNELS, rolloff: float = 0.5, span: int = 16) -> np.ndarray:
    """Low-pass filter used by the channelizer (unit gain at 0 Hz).

    Its squared magnitude is a raised cosine, so the parts of the spectrum that
    fold back on top of each other when we decimate add up to a FLAT level.
    Result: receiver noise stays white after channelizing, exactly like the
    noise the classifier was trained on. Pass band +/-0.25, stop band beyond
    +/-0.75 of the channel rate, so neighbouring channels are rejected.
    """
    h = rrc_taps(rolloff, n_channels, span)
    return h / h.sum()


def channelize(x_wide: np.ndarray, channel: int, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Pull one channel out of the wideband signal: shift it to 0 Hz, low-pass, decimate."""
    shifted = frequency_shift(x_wide, -channel_centres(n_channels)[channel], 1.0)
    h = channel_filter(n_channels)
    delay = (len(h) - 1) // 2
    filtered = np.convolve(shifted, h)[delay:delay + len(shifted)]
    return filtered[::n_channels]


def channelize_all(x_wide: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """All channels at once -> array of shape (n_channels, len(x_wide) // n_channels)."""
    return np.stack([channelize(x_wide, c, n_channels) for c in range(n_channels)])


# --------------------------------------------------------------------------
# Scenes: a band observed over several time slots
# --------------------------------------------------------------------------
@dataclass
class Scene:
    """A wideband recording made of consecutive time slots.

    iq:      complex wideband samples, length n_slots * slot_len
    labels:  (n_slots, n_channels) class index in each channel/slot (NOISE_LABEL = empty)
    snr_db:  (n_slots, n_channels) SNR of the signal in that channel/slot (nan if empty)
    n_native: native samples per slot per channel (= classifier input length)
    """
    iq: np.ndarray
    labels: np.ndarray
    snr_db: np.ndarray
    n_native: int
    n_channels: int = N_CHANNELS

    @property
    def n_slots(self) -> int:
        return self.labels.shape[0]

    @property
    def slot_len(self) -> int:
        return self.n_native * self.n_channels

    def slot(self, t: int) -> np.ndarray:
        return self.iq[t * self.slot_len:(t + 1) * self.slot_len]

    @property
    def occupied(self) -> np.ndarray:
        """Ground truth: True where a channel holds anything but noise."""
        return self.labels != NOISE_LABEL


def build_scene(labels: np.ndarray, snr_db: np.ndarray, n_native: int = 1024,
                rng: np.random.Generator | None = None, cfo: np.ndarray | None = None) -> Scene:
    """Synthesise a wideband scene.

    labels: (n_slots, n_channels) class indices (NOISE_LABEL for an empty channel)
    snr_db: same shape, per-channel SNR for occupied cells
    cfo:    optional same-shape frequency offsets, cycles per NATIVE sample
    """
    rng = rng or np.random.default_rng()
    labels = np.asarray(labels)
    n_slots, n_ch = labels.shape
    slots = []
    for t in range(n_slots):
        slot = awgn(n_native * n_ch, 1.0, rng)
        for c in range(n_ch):
            if labels[t, c] == NOISE_LABEL:
                continue
            x = clean_signal(CLASSES[labels[t, c]], n_native, rng)
            x = x * np.exp(1j * rng.uniform(0, 2 * np.pi))
            if cfo is not None and cfo[t, c]:
                x = frequency_shift(x, cfo[t, c], 1.0)
            amplitude = np.sqrt(10 ** (snr_db[t, c] / 10) / n_ch)   # noise in-channel = 1/n_ch
            slot += to_wideband(amplitude * x, c, n_ch)
        slots.append(slot)
    snr = np.where(labels == NOISE_LABEL, np.nan, snr_db).astype(float)
    return Scene(np.concatenate(slots), labels, snr, n_native, n_ch)


def scene_windows(scene: Scene) -> np.ndarray:
    """Channelize every slot -> complex array (n_slots, n_channels, n_native)."""
    out = np.empty((scene.n_slots, scene.n_channels, scene.n_native), dtype=complex)
    for t in range(scene.n_slots):
        out[t] = channelize_all(scene.slot(t), scene.n_channels)[:, :scene.n_native]
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def psd(x: np.ndarray, nperseg: int = 1024):
    """Welch power spectral density, centred on 0. Returns (freq, PSD) with PSD in linear units."""
    f, p = welch(x, fs=1.0, nperseg=nperseg, return_onesided=False, scaling="density")
    order = np.argsort(f)
    return f[order], p[order]


def spectrogram_db(x: np.ndarray, nperseg: int = 256, noverlap: int | None = None):
    """Spectrogram (time, freq, power in dB), frequency centred on 0."""
    f, t, S = spectrogram(x, fs=1.0, nperseg=nperseg, noverlap=noverlap,
                          return_onesided=False, scaling="density")
    order = np.argsort(f)
    return t, f[order], 10 * np.log10(S[order] + 1e-15)


def noise_floor(x: np.ndarray, nperseg: int = 1024) -> float:
    """Estimate the noise power spectral density from the MEDIAN of the PSD.

    Signals occupy only part of the band, so the median bin is noise-only.
    Total noise power in a band of width B (fraction of fs) = floor * B.
    """
    _, p = psd(x, nperseg)
    return float(np.median(p))
