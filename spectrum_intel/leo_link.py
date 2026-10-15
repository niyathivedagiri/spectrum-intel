"""End-to-end LEO link: the real packet link (Phases 1-9) carried through a satellite pass.

Along the pass, every packet sees the pass at that moment:
  - SNR from the link budget at the current slant range (leo.link_snr_db), so it rises and falls,
  - a carrier frequency offset = Doppler at that moment + the ground oscillator's error.

Doppler handling (the receiver alone tolerates about +-6 kHz, Doppler reaches +-46 kHz at 2 GHz):
  "none"       nothing: the receiver gets the full Doppler.
  "search"     the receiver tries frequency offsets over the whole Doppler range (blind search).
  "predicted"  the ground station predicts Doppler from the orbit and removes it before sending
               (as real ground stations do with orbital elements). What is left: the oscillator
               error (+-osc_ppm) and the Doppler change over the orbit-timing error.
  "predicted+search"  prediction plus a narrow search around zero, for larger oscillator errors.

Modulation and coding: a fixed scheme ("qpsk/1/2", "qam16/none", ...), "adaptive" (modulation
only, uncoded table, Phase 6) or "mcs" (modulation AND code, Phase 8 table). Feedback is one
packet old, as before.

Time is sampled: every dt_s seconds a burst of packets is sent; each burst stands for dt_s seconds
of continuous transmission at its measured goodput (data delivered = sum of goodput x dt_s).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spectrum_intel import leo
from spectrum_intel.adaptation import LinkAdapter, MCSAdapter
from spectrum_intel.fec import CODES
from spectrum_intel.link import transfer_packet
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import segment

DOPPLER_MODES = ("none", "search", "predicted", "predicted+search")
SAMPLE_RATE_HZ = 1e6
NARROW_SEARCH = 0.012            # cycles/sample searched in "predicted+search" (+-12 kHz at 1 MS/s)


@dataclass
class LEOLinkConfig:
    altitude_m: float = 550e3
    max_elevation_deg: float = 90.0
    min_elevation_deg: float = 10.0
    carrier_hz: float = 2.0e9
    eirp_dbm: float = 23.0
    rx_gain_dbi: float = 30.0
    noise_figure_db: float = 3.0
    doppler: str = "predicted"
    osc_ppm: float = 2.0             # ground oscillator error, uniform in +-osc_ppm
    timing_error_s: float = 0.5      # orbit prediction error, as a timing error along the track
    scheme: str = "mcs"              # "mcs", "adaptive" or a fixed "mod/code"
    dt_s: float = 4.0
    packets_per_step: int = 8
    seed: int = 0

    def validated(self) -> "LEOLinkConfig":
        if self.doppler not in DOPPLER_MODES:
            raise ValueError(f"doppler must be one of {DOPPLER_MODES}")
        if self.scheme not in ("mcs", "adaptive"):
            mod, _, code = self.scheme.partition("/")
            if mod not in LINK_MODULATIONS or code not in CODES:
                raise ValueError("scheme must be 'mcs', 'adaptive' or 'mod/code', e.g. 'qpsk/1/2'")
        return self


def timeline(cfg: LEOLinkConfig) -> dict:
    """Pass geometry and link budget every dt_s seconds."""
    p = leo.general_pass(cfg.altitude_m, cfg.max_elevation_deg, cfg.min_elevation_deg, cfg.dt_s)
    fd = leo.doppler_hz(p.range_rate_mps, cfg.carrier_hz)
    snr = leo.link_snr_db(p.range_m, cfg.carrier_hz, cfg.eirp_dbm, cfg.rx_gain_dbi, SAMPLE_RATE_HZ,
                          cfg.noise_figure_db)
    return {"t_s": p.time_s, "elev_deg": p.elevation_deg, "range_km": p.range_m / 1e3,
            "doppler_hz": fd, "doppler_rate_hz_s": np.gradient(fd, p.time_s), "snr_db": snr}


def offsets(cfg: LEOLinkConfig, tl: dict, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Frequency offset the receiver sees at each step (cycles/sample), and its search range."""
    osc = rng.uniform(-1, 1) * cfg.osc_ppm * 1e-6 * cfg.carrier_hz          # fixed for the pass
    fd = tl["doppler_hz"]
    if cfg.doppler.startswith("predicted"):
        predicted = np.interp(tl["t_s"] + cfg.timing_error_s, tl["t_s"], fd)  # orbit known with a time error
        residual = fd - predicted + osc
    else:
        residual = fd + osc
    search = {"none": 0.0, "predicted": 0.0, "predicted+search": NARROW_SEARCH,
              "search": 1.1 * float(np.max(np.abs(fd)) + cfg.osc_ppm * 1e-6 * cfg.carrier_hz) / SAMPLE_RATE_HZ}
    return residual / SAMPLE_RATE_HZ, search[cfg.doppler]


def run_pass(cfg: LEOLinkConfig) -> dict:
    """Send packets through one pass. Returns the timeline, per-step results and a summary."""
    cfg = cfg.validated()
    tl = timeline(cfg)
    rng = np.random.default_rng([cfg.seed, 1])
    cfo, search = offsets(cfg, tl, rng)
    data = np.random.default_rng([cfg.seed, 2]).integers(0, 256, 64 * 16, dtype=np.uint8).tobytes()
    pk = segment(data, 64)
    mcs = MCSAdapter() if cfg.scheme == "mcs" else None
    amc = LinkAdapter("goodput") if cfg.scheme == "adaptive" else None
    steps, k = [], 0
    for i in range(len(tl["t_s"])):
        bits, air, ok_n, det_n, used = 0, 0.0, 0, 0, []
        for j in range(cfg.packets_per_step):
            if mcs is not None:
                mod, code = mcs.choose()
            elif amc is not None:
                mod, code = amc.choose(), "none"
            else:
                mod, _, code = cfg.scheme.partition("/")
            prng = np.random.default_rng([cfg.seed, 3, i, j])               # same draws for every strategy
            r = transfer_packet(pk[k % 16], 16, mod, float(tl["snr_db"][i]), prng, cfo=float(cfo[i]),
                                cfo_search=search, code=code).record
            k += 1
            for ad in (mcs, amc):
                if ad is not None:
                    ad.update(r)
            air += r.n_samples / SAMPLE_RATE_HZ
            bits += 8 * r.data_bytes * r.crc_ok
            ok_n += r.crc_ok
            det_n += r.detected
            used.append(f"{mod}/{code}")
        steps.append({"t_s": float(tl["t_s"][i]), "elev_deg": float(tl["elev_deg"][i]), "snr_db": float(tl["snr_db"][i]),
                      "doppler_khz": float(tl["doppler_hz"][i] / 1e3), "cfo_khz": float(cfo[i] * SAMPLE_RATE_HZ / 1e3),
                      "goodput_kbps": bits / air / 1e3, "delivery": ok_n / cfg.packets_per_step,
                      "detection": det_n / cfg.packets_per_step, "scheme": max(set(used), key=used.count)})
    gp = np.array([s["goodput_kbps"] for s in steps])
    return {"config": asdict(cfg), "timeline": {k: np.asarray(v).tolist() for k, v in tl.items()}, "steps": steps,
            "summary": {"pass_s": float(tl["t_s"][-1] + cfg.dt_s), "delivered_MB": float(np.sum(gp) * 1e3 * cfg.dt_s / 8 / 1e6),
                        "link_up_fraction": float(np.mean(gp > 0)), "mean_goodput_kbps": float(gp.mean()),
                        "packet_delivery": float(np.mean([s["delivery"] for s in steps])),
                        "max_doppler_khz": float(np.max(np.abs(tl["doppler_hz"])) / 1e3),
                        "max_residual_cfo_khz": float(np.max(np.abs(cfo)) * SAMPLE_RATE_HZ / 1e3),
                        "search_range_khz": search * SAMPLE_RATE_HZ / 1e3}}
