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


# --------------------------------------------------------------------------
# Dynamic channel: fading, multipath, interference
# --------------------------------------------------------------------------
FADING_MODELS = ("none", "rayleigh", "rician")
MULTIPATH_MODELS = ("none", "two_ray", "exponential")
JAMMER_MODELS = ("none", "tone", "pulsed")


@dataclass
class ChannelConfig:
    """Impairments beyond AWGN. All off = plain AWGN (identical to awgn_link).

    fading       "rayleigh": no line of sight, gain is complex Gaussian (deep fades).
                 "rician": line of sight + scattered paths, k_factor_db = LOS / scattered power.
    doppler_hz   0 = gain frozen for the whole frame (block fading); > 0 = gain changes
                 during the frame (Clarke model, maximum Doppler frequency in Hz).
    multipath    "two_ray": direct path + one echo, delay_spread_us later, echo_db weaker.
                 "exponential": taps every sample, power decaying as exp(-t / delay_spread).
    jammer       "tone": a continuous carrier at jam_offset_khz.
                 "pulsed": broadband noise bursts, `duty` of the time, period jam_period_us.
                 Strength set by sir_db = average signal power / average jammer power.
    """
    fading: str = "none"
    k_factor_db: float = 6.0
    doppler_hz: float = 0.0
    multipath: str = "none"
    delay_spread_us: float = 4.0
    echo_db: float = -3.0
    jammer: str = "none"
    sir_db: float = 10.0
    jam_offset_khz: float = 20.0
    duty: float = 0.1
    jam_period_us: float = 500.0

    def __post_init__(self):
        if self.fading not in FADING_MODELS:
            raise ValueError(f"fading must be one of {FADING_MODELS}")
        if self.multipath not in MULTIPATH_MODELS:
            raise ValueError(f"multipath must be one of {MULTIPATH_MODELS}")
        if self.jammer not in JAMMER_MODELS:
            raise ValueError(f"jammer must be one of {JAMMER_MODELS}")

    @property
    def is_awgn(self) -> bool:
        return self.fading == "none" and self.multipath == "none" and self.jammer == "none"

    def describe(self) -> str:
        parts = []
        if self.fading != "none":
            parts.append(f"{self.fading}" + (f" K={self.k_factor_db:g} dB" if self.fading == "rician" else "")
                         + (f" fd={self.doppler_hz:g} Hz" if self.doppler_hz else " block"))
        if self.multipath != "none":
            parts.append(f"{self.multipath} {self.delay_spread_us:g} us")
        if self.jammer != "none":
            parts.append(f"{self.jammer} jammer SIR {self.sir_db:g} dB")
        return "AWGN" + ("".join(" + " + p for p in parts) if parts else "")


@dataclass
class DynamicChannelInfo(ChannelInfo):
    taps: list = None                 # [(delay in samples, mean power)]
    gain: np.ndarray = None           # complex gain of the first tap over time (what the receiver must track)
    jammer_power: float = 0.0


def tap_profile(cfg: ChannelConfig, fs: float) -> list[tuple[int, float]]:
    """Power-delay profile: [(delay_samples, power)], powers summing to 1."""
    d = cfg.delay_spread_us * 1e-6 * fs
    if cfg.multipath == "two_ray":
        taps = [(0, 1.0), (max(1, int(round(d))), 10 ** (cfg.echo_db / 10))]
    elif cfg.multipath == "exponential" and d > 0:
        n = np.arange(int(np.ceil(5 * d)) + 1)
        taps = list(zip(n.tolist(), np.exp(-n / d).tolist()))
    else:
        taps = [(0, 1.0)]
    total = sum(p for _, p in taps)
    return [(k, p / total) for k, p in taps]


def fading_process(n: int, doppler_hz: float, fs: float, rng: np.random.Generator,
                   n_sinusoids: int = 16) -> np.ndarray:
    """Unit-power complex Gaussian fading gain over n samples (Clarke / Jakes spectrum).

    doppler_hz = 0: one constant gain (block fading). Otherwise a sum of sinusoids with
    random arrival angles: the autocorrelation approaches J0(2*pi*fd*tau).
    """
    if doppler_hz <= 0:
        return np.full(n, (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2))
    t = np.arange(n) / fs
    alpha = rng.uniform(0, 2 * np.pi, n_sinusoids)
    phi = rng.uniform(0, 2 * np.pi, n_sinusoids)
    return np.exp(1j * (2 * np.pi * doppler_hz * np.cos(alpha)[:, None] * t[None, :] + phi[:, None])).sum(0) \
        / np.sqrt(n_sinusoids)


def dynamic_link(tx: np.ndarray, snr_db: float, rng: np.random.Generator, cfg: ChannelConfig | None = None, *,
                 cfo: float = 0.0, fs: float = 1e6, phase: float | None = None, delay: int | None = None,
                 tail: int = 256):
    """Burst over fading / multipath / interference + AWGN.

    With cfg None or all impairments off this IS awgn_link (same random draws, same output).
    SNR = average transmitted power / noise power; the channel taps have total average power 1,
    so snr_db is the AVERAGE received SNR and fades make the instantaneous SNR swing around it.
    """
    if cfg is None or cfg.is_awgn:
        return awgn_link(tx, snr_db, rng, phase=phase, delay=delay, cfo=cfo, tail=tail)
    phase = rng.uniform(0, 2 * np.pi) if phase is None else phase
    delay = int(rng.integers(32, 288)) if delay is None else int(delay)
    p_sig = signal_power(tx)
    taps = tap_profile(cfg, fs)
    max_d = max(k for k, _ in taps)
    n_out = len(tx) + max_d
    k_lin = 10 ** (cfg.k_factor_db / 10)
    y = np.zeros(n_out, complex)
    first_gain = None
    for i, (k, p) in enumerate(taps):
        if cfg.fading == "none":
            g = np.full(n_out, np.exp(1j * (0.0 if i == 0 else rng.uniform(0, 2 * np.pi))))
        else:
            g = fading_process(n_out, cfg.doppler_hz, fs, rng)
            if cfg.fading == "rician" and i == 0:          # line of sight on the direct path only
                g = np.sqrt(k_lin / (k_lin + 1)) + np.sqrt(1 / (k_lin + 1)) * g
        g = np.sqrt(p) * g
        if i == 0:
            first_gain = g[:len(tx)]
        y[k:k + len(tx)] += g[k:k + len(tx)] * tx
    x = y * np.exp(1j * phase)
    if cfo:
        x = frequency_shift(x, cfo, 1.0)
    padded = np.concatenate([np.zeros(delay, complex), x, np.zeros(tail, complex)])
    noise_power = p_sig / 10 ** (snr_db / 10)
    rx = padded + awgn(len(padded), noise_power, rng)
    jam_p = 0.0
    if cfg.jammer != "none":
        jam_p = p_sig / 10 ** (cfg.sir_db / 10)
        n = np.arange(len(rx))
        if cfg.jammer == "tone":
            jam = np.sqrt(jam_p) * np.exp(1j * (2 * np.pi * cfg.jam_offset_khz * 1e3 / fs * n + rng.uniform(0, 2 * np.pi)))
        else:
            period = max(2, int(round(cfg.jam_period_us * 1e-6 * fs)))
            on = (n + rng.integers(period)) % period < max(1, int(round(cfg.duty * period)))
            jam = on * awgn(len(rx), jam_p / max(cfg.duty, 1e-3), rng)    # average power = jam_p
        rx = rx + jam
    info = DynamicChannelInfo(snr_db, noise_power, p_sig, phase, delay, cfo,
                              taps=taps, gain=first_gain, jammer_power=jam_p)
    return rx, info
