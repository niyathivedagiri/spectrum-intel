"""Link adaptation: choose the modulation for the next packet from what the receiver measured.

Feedback model (like CQI + ACK/NACK in LTE): after each packet the receiver reports
  - its SNR estimate from that packet's preamble (None if the frame was not found)
  - whether the packet passed its CRC (ACK) or not (NACK).
The transmitter only knows these for the PREVIOUS packet, so its information is always
one packet old. A genie (perfect, current SNR) is available for experiments only.

Policies
  "threshold": pick the fastest modulation whose packet error rate (PER), read from a
               calibration table measured in AWGN, is at most `target_per` at the reported SNR.
  "olla":      the same, but first subtract an offset learned from ACK/NACK (outer-loop link
               adaptation). Each NACK raises the offset by `step_db`; each ACK lowers it by
               step_db * target / (1 - target). At equilibrium NACKs happen at the target rate,
               so the loop corrects a table that is wrong for the real channel (fading,
               multipath, a biased SNR estimate).
  "goodput":   pick the modulation with the highest EXPECTED goodput at the reported SNR:
               (data bits per second of airtime) x (1 - PER from the table). Without
               retransmissions this, not a fixed error target, is what maximises delivered data:
               a faster modulation is worth some lost packets if it delivers more overall.
If no usable feedback exists yet (start, or the last frame was not even detected), the most
robust modulation is used.

MCSAdapter (adaptive modulation AND coding): the same goodput rule over all 12 pairs of
modulation (BPSK/QPSK/16-QAM) and FEC code (none, 1/2, 2/3, 3/4), from mcs_table.json, which
holds the packet error rates measured in the FEC experiment (scripts/calibrate_mcs_table.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spectrum_intel.modulation import LINK_MODULATIONS, bits_per_symbol

TABLE_PATH = Path(__file__).resolve().parent / "link_table.json"
MCS_TABLE_PATH = Path(__file__).resolve().parent / "mcs_table.json"
POLICIES = ("threshold", "olla", "goodput")


def load_table(path: Path = TABLE_PATH) -> dict:
    """Calibration table: {"snr_db": [...], "per": {mod: [...]}, "meta": {...}}."""
    return json.loads(Path(path).read_text())


def per_at(table: dict, mod: str, snr_db: float) -> float:
    """Packet error rate of `mod` at `snr_db`, interpolated in the table (clamped at the ends)."""
    return float(np.interp(snr_db, table["snr_db"], table["per"][mod]))


def required_snr(table: dict, mod: str, target_per: float) -> float:
    """Lowest SNR at which the table's PER for `mod` is <= target (inf if never)."""
    snr, per = np.asarray(table["snr_db"], float), np.asarray(table["per"][mod], float)
    ok = np.where(per <= target_per)[0]
    if not len(ok):
        return float("inf")
    i = ok[0]
    if i == 0:
        return float(snr[0])
    # linear interpolation between the last point above target and the first at/below it
    return float(snr[i - 1] + (per[i - 1] - target_per) / (per[i - 1] - per[i]) * (snr[i] - snr[i - 1]))


@dataclass
class LinkAdapter:
    policy: str = "threshold"
    target_per: float = 0.1
    step_db: float = 1.0                       # OLLA step on a NACK
    mods: tuple = LINK_MODULATIONS
    table: dict = None
    snr_fb: float | None = None                # last reported SNR estimate (dB)
    offset_db: float = 0.0                     # OLLA correction (dB), 0 for "threshold"
    history: list = field(default_factory=list)

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if not 0 < self.target_per < 1:
            raise ValueError("target_per must be between 0 and 1")
        self.table = self.table or load_table()
        # slowest -> fastest, so "the fastest that meets the target" is a scan from the end
        self.mods = tuple(sorted(self.mods, key=bits_per_symbol))
        self.thresholds = {m: required_snr(self.table, m, self.target_per) for m in self.mods}
        self.rate_bps = {m: data_rate_bps(m, int(self.table.get("meta", {}).get("packet_bytes", 64)))
                         for m in self.mods}

    def expected_goodput(self, snr_db: float) -> dict:
        """Expected goodput (bit/s) of each modulation at snr_db, from the table."""
        return {m: self.rate_bps[m] * (1 - per_at(self.table, m, snr_db)) for m in self.mods}

    @property
    def effective_snr(self) -> float | None:
        return None if self.snr_fb is None else self.snr_fb - self.offset_db

    def choose(self, snr_override: float | None = None) -> str:
        """Modulation for the next packet. snr_override = genie SNR (experiments only)."""
        snr = snr_override if snr_override is not None else self.effective_snr
        if snr is None:
            return self.mods[0]
        if self.policy == "goodput":
            eg = self.expected_goodput(snr)
            return max(self.mods, key=lambda m: (eg[m], -bits_per_symbol(m)))    # ties -> more robust
        for m in reversed(self.mods):
            if snr >= self.thresholds[m]:
                return m
        return self.mods[0]

    def update(self, record) -> None:
        """Feedback from one received packet (a link.PacketRecord)."""
        snr = getattr(record, "snr_est_db", None)
        detected = bool(getattr(record, "detected", False))
        self.snr_fb = float(snr) if detected and snr is not None and np.isfinite(snr) else None
        if self.policy == "olla":
            if record.crc_ok:
                self.offset_db -= self.step_db * self.target_per / (1 - self.target_per)
            else:
                self.offset_db += self.step_db
            self.offset_db = float(np.clip(self.offset_db, -10.0, 20.0))
        self.history.append({"mod": record.mod, "ack": bool(record.crc_ok), "snr_fb": self.snr_fb,
                             "offset_db": self.offset_db})

    def summary(self) -> dict:
        return {"policy": self.policy, "target_per": self.target_per, "offset_db": round(self.offset_db, 3),
                "snr_feedback_db": None if self.snr_fb is None else round(self.snr_fb, 3),
                "thresholds_db": {m: (None if not np.isfinite(v) else round(v, 2)) for m, v in self.thresholds.items()},
                "next": self.choose()}


def data_rate_bps(mod: str, packet_bytes: int = 64, sample_rate_hz: float = 1e6, code: str = "none") -> float:
    """User data bits per second of airtime for one packet of `packet_bytes` (all overheads included)."""
    from spectrum_intel.packet import Packet
    from spectrum_intel.transmitter import build_frame
    frame = build_frame(Packet(0, 1, bytes(packet_bytes)).to_bytes(), mod, code)
    return 8 * packet_bytes * sample_rate_hz / len(frame.iq)


def slow_fading_snr(n: int, mean_snr_db: float, doppler_hz: float, slot_s: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Per-packet SNR (dB) under Rayleigh block fading that is CORRELATED from packet to packet.

    The complex gain is a Clarke/Jakes process sampled once per packet slot, so f_d * slot
    sets how much the channel changes between consecutive packets (small = slow fading,
    the previous packet's SNR is a good guess; ~0.5 or more = nearly independent).
    """
    from spectrum_intel.channel import fading_process
    g = fading_process(n, doppler_hz, 1.0 / slot_s, rng)
    return mean_snr_db + 20 * np.log10(np.maximum(np.abs(g), 1e-4))


# --------------------------------------------------------------------------
# Adaptive modulation AND coding
# --------------------------------------------------------------------------
def scheme_name(mod: str, code: str) -> str:
    return f"{mod}/{code}"


def load_mcs_table(path: Path = MCS_TABLE_PATH) -> dict:
    """{"snr_db": [...], "per": {"qpsk/1/2": [...], ...}, "meta": {...}}"""
    return json.loads(Path(path).read_text())


@dataclass
class MCSAdapter:
    """Chooses (modulation, code) for the next packet: highest expected goodput at the reported SNR.

    margin_db is subtracted from the reported SNR first (a fixed safety margin; 0 = trust the table).
    """
    margin_db: float = 0.0
    table: dict = None
    snr_fb: float | None = None
    history: list = field(default_factory=list)

    def __post_init__(self):
        from spectrum_intel.fec import CODES
        self.table = self.table or load_mcs_table()
        self.schemes = [(m, c) for m in LINK_MODULATIONS for c in CODES if scheme_name(m, c) in self.table["per"]]
        pb = int(self.table.get("meta", {}).get("packet_bytes", 64))
        self.rate_bps = {s: data_rate_bps(s[0], pb, code=s[1]) for s in self.schemes}
        self.robust = min(self.schemes, key=lambda s: self.rate_bps[s])

    def expected_goodput(self, snr_db: float) -> dict:
        return {s: self.rate_bps[s] * (1 - float(np.interp(snr_db, self.table["snr_db"],
                                                            self.table["per"][scheme_name(*s)])))
                for s in self.schemes}

    def choose(self, snr_override: float | None = None) -> tuple[str, str]:
        snr = snr_override if snr_override is not None else (None if self.snr_fb is None else self.snr_fb - self.margin_db)
        if snr is None:
            return self.robust
        eg = self.expected_goodput(snr)
        return max(self.schemes, key=lambda s: (eg[s], -self.rate_bps[s]))   # ties -> slower, more robust

    def update(self, record) -> None:
        snr = getattr(record, "snr_est_db", None)
        detected = bool(getattr(record, "detected", False))
        self.snr_fb = float(snr) if detected and snr is not None and np.isfinite(snr) else None
        self.history.append({"scheme": scheme_name(record.mod, getattr(record, "code", "none")),
                             "ack": bool(record.crc_ok), "snr_fb": self.snr_fb})

    def summary(self) -> dict:
        m, c = self.choose()
        return {"policy": "mcs", "margin_db": self.margin_db, "offset_db": 0.0,
                "snr_feedback_db": None if self.snr_fb is None else round(self.snr_fb, 3),
                "next": m, "next_code": c}
