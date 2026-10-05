"""Real-time link engine: sends a message one packet per step and exposes the live state.

The engine drives exactly the same per-packet chain as `link.send`
(`link.transfer_packet`), so a run that is not changed mid-way gives identical
packets, bit errors and received bytes for the same seed. What it adds:
  - stepping (one packet at a time) with pause/resume/reset,
  - settings that can change between packets (modulation, SNR, frequency offset,
    error-tolerant mode) to watch the link react,
  - display data for the last frame: equalised constellation, received spectrum,
    waveform around the detected frame start,
  - the message as it builds up, the scorecard so far and over the last packets,
  - wall-clock processing time per packet vs the packet's airtime (real-time factor).
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from spectrum_intel import metrics as m
from spectrum_intel import payload as pl
from spectrum_intel.link import PacketRecord, transfer_packet
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import reassemble, segment
from spectrum_intel.signals import constellation
from spectrum_intel.transmitter import HEADER_BITS, SPS

DEFAULT_TEXT = (
    "Hello from the cognitive radio! This message is cut into packets of 64 bytes. Each packet gets a "
    "sequence number and a CRC, is mapped onto BPSK, QPSK or 16-QAM symbols, shaped into I/Q samples "
    "and sent through a noisy channel with an unknown phase, arrival time and frequency offset. The "
    "receiver knows none of these in advance: it finds the preamble, estimates the offset, tracks the "
    "pilots, decides each symbol and checks the CRC. Green text passed the check. Orange text arrived "
    "damaged but was kept. Red dots mark packets that were lost. Turn the SNR down and watch the "
    "constellation spread out and the errors appear; switch to 16-QAM and see how much more signal it needs.")
LIVE_FIELDS = {"mod", "snr_db", "cfo", "tolerant", "rate_pps"}          # may change between packets
RESET_FIELDS = {"source", "text", "image_size", "gray", "seed", "max_payload"}  # need a restart
MAX_TEXT_CHARS = 4000
MAX_IMAGE_SIZE = 128


@dataclass
class EngineConfig:
    source: str = "text"            # "text" or "image"
    text: str = DEFAULT_TEXT
    image_size: int = 48            # image is resized to image_size x image_size
    gray: bool = False
    mod: str = "qpsk"
    snr_db: float = 4.0             # SNR per sample (dB)
    cfo: float = 0.0                # carrier frequency offset, cycles/sample
    tolerant: bool = False          # keep damaged packets (images)
    seed: int = 1
    max_payload: int = 64
    rate_pps: float = 8.0           # display pace, packets per second (server only)
    sample_rate_hz: float = m.DEFAULT_SAMPLE_RATE_HZ

    def validated(self) -> "EngineConfig":
        if self.source not in ("text", "image"):
            raise ValueError("source must be 'text' or 'image'")
        if self.mod not in LINK_MODULATIONS:
            raise ValueError(f"mod must be one of {LINK_MODULATIONS}")
        return replace(self,
                       text=str(self.text)[:MAX_TEXT_CHARS] or " ",
                       image_size=int(np.clip(self.image_size, 8, MAX_IMAGE_SIZE)),
                       snr_db=float(np.clip(self.snr_db, -20, 40)),
                       cfo=float(np.clip(self.cfo, -0.02, 0.02)),
                       seed=int(self.seed), max_payload=int(np.clip(self.max_payload, 8, 1024)),
                       rate_pps=float(np.clip(self.rate_pps, 0.2, 200)),
                       tolerant=bool(self.tolerant), gray=bool(self.gray))


@dataclass
class StepView:
    """Display data for the most recent packet."""
    seq: int
    record: PacketRecord
    constellation: np.ndarray       # equalised payload symbols (complex), empty if header lost
    psd_freq_khz: np.ndarray
    psd_db: np.ndarray
    wave_i: np.ndarray
    wave_q: np.ndarray
    wave_start: int                 # sample index of wave_i[0] in the received burst
    frame_start: int                # detected frame start (-1 if not detected)
    sync_metric: float
    snr_set_db: float
    process_ms: float
    airtime_ms: float
    outcome: str                    # "ok", "damaged" (kept), "lost"


def _psd(x: np.ndarray, nfft: int = 256) -> np.ndarray:
    """Averaged periodogram (Hann window, 50% overlap), fftshifted, linear power."""
    if len(x) < nfft:
        x = np.concatenate([x, np.zeros(nfft - len(x), complex)])
    w = np.hanning(nfft)
    hop = nfft // 2
    segs = [x[i:i + nfft] * w for i in range(0, len(x) - nfft + 1, hop)]
    p = np.mean(np.abs(np.fft.fft(segs, axis=1)) ** 2, axis=0) / np.sum(w ** 2)
    return np.fft.fftshift(p)


def _png_data_url(img: np.ndarray, scale: int = 4) -> str:
    from PIL import Image
    im = Image.fromarray(img)
    im = im.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LinkEngine:
    """Step-by-step sender/receiver. Not thread-safe on its own; the server wraps it in a lock."""

    def __init__(self, config: EngineConfig | None = None, image: np.ndarray | None = None):
        self._custom_image = image
        self.reset(config or EngineConfig())

    # ---------------------------------------------------------------- setup
    def reset(self, config: EngineConfig | None = None):
        self.config = (config or self.config).validated()
        c = self.config
        if c.source == "image":
            img = self._custom_image if self._custom_image is not None else pl.load_image(None, c.image_size, c.gray)
            if self._custom_image is not None:
                img = self._resize(img, c.image_size, c.gray)
            self.image = img
            self.data = pl.image_to_bytes(img)
            self.fill = 128                         # mid-grey for missing pixels
        else:
            self.image = None
            self.data = pl.text_to_bytes(c.text)
            self.fill = 0x00
        self.packets = segment(self.data, c.max_payload)
        self.rng = np.random.default_rng(c.seed)
        self.records: list[PacketRecord] = []
        self.outcomes: list[str] = []               # per sent packet: ok / damaged / lost
        self.snr_set: list[float] = []
        self.process_ms: list[float] = []
        self._good, self._rough = {}, {}
        self.last: StepView | None = None
        self.started = time.time()

    def set_image(self, img: np.ndarray | None):
        """Use a custom image (None = default sample) for image runs; takes effect on reset."""
        self._custom_image = None if img is None else np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _resize(img: np.ndarray, size: int, gray: bool) -> np.ndarray:
        from PIL import Image
        im = Image.fromarray(img)
        im = im.convert("L" if gray else "RGB")
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.uint8)

    def update(self, **changes) -> bool:
        """Apply settings. Live fields act from the next packet; others restart the run.

        Returns True if the run was restarted.
        """
        unknown = set(changes) - LIVE_FIELDS - RESET_FIELDS
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        new = replace(self.config, **changes).validated()
        restart = any(getattr(new, k) != getattr(self.config, k) for k in RESET_FIELDS if k in changes)
        if restart:
            self.reset(new)
        else:
            self.config = new
        return restart

    # ---------------------------------------------------------------- running
    @property
    def n_packets(self) -> int:
        return len(self.packets)

    @property
    def done(self) -> bool:
        return len(self.records) >= self.n_packets

    def step(self) -> StepView | None:
        """Send the next packet. Returns its display data, or None when the message is complete."""
        if self.done:
            return None
        c = self.config
        pkt = self.packets[len(self.records)]
        t0 = time.perf_counter()
        t = transfer_packet(pkt, self.n_packets, c.mod, c.snr_db, self.rng, max_payload=c.max_payload,
                            cfo=c.cfo, keep_corrupted=c.tolerant)
        proc_ms = 1e3 * (time.perf_counter() - t0)
        if t.good is not None:
            self._good[t.good.seq] = t.good
        if t.rough is not None:
            self._rough.setdefault(t.rough.seq, t.rough)      # first damaged copy wins, as in link.send
        outcome = "ok" if t.good is not None else ("damaged" if t.rough is not None else "lost")
        self.records.append(t.record)
        self.outcomes.append(outcome)
        self.snr_set.append(c.snr_db)
        self.process_ms.append(proc_ms)
        self.last = self._view(pkt.seq, t, outcome, proc_ms)
        return self.last

    def run_to_end(self):
        while self.step() is not None:
            pass
        return self

    def _view(self, seq, t, outcome, proc_ms) -> StepView:
        res, rx = t.res, t.rx
        const = res.data_symbols[HEADER_BITS:] if res.header_ok and res.data_symbols is not None else np.array([])
        p = _psd(rx)
        fs = self.config.sample_rate_hz
        freq = np.fft.fftshift(np.fft.fftfreq(len(p), 1 / fs)) / 1e3
        start = res.start if res.detected and res.start >= 0 else 0
        w0 = max(0, start - 16 * SPS)
        seg = rx[w0:w0 + 96 * SPS]
        return StepView(seq=seq, record=t.record, constellation=const[:800],
                        psd_freq_khz=freq, psd_db=10 * np.log10(p + 1e-12),
                        wave_i=seg.real, wave_q=seg.imag, wave_start=w0,
                        frame_start=res.start if res.detected else -1, sync_metric=float(res.sync_metric),
                        snr_set_db=self.config.snr_db, process_ms=proc_ms,
                        airtime_ms=1e3 * t.record.n_samples / fs, outcome=outcome)

    # ---------------------------------------------------------------- results so far
    def received(self) -> tuple[bytes, list[int]]:
        """Message as rebuilt so far (pending packets count as missing)."""
        pk = list(self._good.values()) + [p for s, p in self._rough.items() if s not in self._good]
        return reassemble(pk, self.n_packets, fill=self.fill, chunk=self.config.max_payload,
                          expected_len=len(self.data))

    @property
    def corrupted(self) -> list[int]:
        return sorted(s for s in self._rough if s not in self._good)

    def report(self, last: int | None = None) -> dict:
        recs = self.records[-last:] if last else self.records
        if not recs:
            return {}
        r = m.link_report(SimpleNamespace(packets=recs), self.config.sample_rate_hz)
        pm = self.process_ms[-last:] if last else self.process_ms
        air = [1e3 * x.n_samples / self.config.sample_rate_hz for x in recs]
        r["process_ms_mean"] = float(np.mean(pm))
        r["realtime_factor"] = float(np.sum(pm) / np.sum(air))   # < 1: faster than real time
        return r

    def payload_view(self) -> dict:
        data, missing = self.received()
        n_sent = len(self.records)
        status = self.outcomes + ["pending"] * (self.n_packets - n_sent)
        if self.config.source == "image":
            img, hdr_ok = pl.bytes_to_image(data, self.image.shape)
            out = {"kind": "image", "image": _png_data_url(img), "original": _png_data_url(self.image),
                   "header_ok": hdr_ok}
            if n_sent:
                out["psnr_db"] = float(min(m.psnr(self.image, img), 60.0))
                out["ssim"] = float(m.ssim(self.image, img))
            return out | {"status": status}
        # The text shows the RECEIVER's view of each slot (which may differ from what was sent:
        # in error-tolerant mode a bit error in the sequence number files data under the wrong slot).
        chunk = self.config.max_payload
        segs = []
        for i in range(self.n_packets):
            orig = self.data[i * chunk:(i + 1) * chunk]
            part = data[i * chunk:(i + 1) * chunk]
            if i in self._good:
                state = "ok" if part == orig else "damaged"      # CRC passed but wrong: undetected error
            elif i in self._rough:
                state = "damaged"
            else:
                state = "lost" if i < n_sent else "pending"
            txt = "\u00b7" * len(orig) if state in ("lost", "pending") else part.decode("utf-8", errors="replace")
            segs.append({"text": txt, "state": state})
        return {"kind": "text", "segments": segs, "status": status,
                "exact_so_far": all(s["state"] == "ok" for s in segs[:n_sent])}  # status = per sent packet

    def state(self, history: int = 300) -> dict:
        """JSON-ready snapshot for the dashboard."""
        c = self.config
        out = {"config": asdict(c), "n_packets": self.n_packets, "sent": len(self.records), "done": self.done,
               "message_bytes": len(self.data), "payload": self.payload_view(),
               "report": _clean(self.report()), "report_recent": _clean(self.report(last=20)),
               "ideal": [[float(z.real), float(z.imag)] for z in constellation(c.mod)]}
        recs = self.records[-history:]
        off = len(self.records) - len(recs)
        out["history"] = [{"i": off + j, "seq": r.seq, "mod": r.mod, "snr_set": self.snr_set[off + j],
                           "snr_est": _num(r.snr_est_db), "outcome": self.outcomes[off + j],
                           "bit_errors": r.bit_errors, "n_bits": r.n_bits,
                           "process_ms": round(self.process_ms[off + j], 2),
                           "airtime_ms": round(1e3 * r.n_samples / c.sample_rate_hz, 3)}
                          for j, r in enumerate(recs)]
        v = self.last
        if v is not None:
            dec = max(1, len(v.wave_i) // 600)
            out["last"] = {"seq": v.seq, "outcome": v.outcome, "mod": v.record.mod,
                           "snr_set_db": v.snr_set_db, "snr_est_db": _num(v.record.snr_est_db),
                           "cfo_est": _num(v.record.cfo_est), "sync_metric": round(v.sync_metric, 3),
                           "bit_errors": v.record.bit_errors, "n_bits": v.record.n_bits,
                           "process_ms": round(v.process_ms, 2), "airtime_ms": round(v.airtime_ms, 3),
                           "constellation": np.round(np.c_[v.constellation.real, v.constellation.imag], 4).tolist(),
                           "psd": {"f_khz": np.round(v.psd_freq_khz, 2).tolist(),
                                   "db": np.round(v.psd_db, 2).tolist()},
                           "wave": {"i": np.round(v.wave_i[::dec], 4).tolist(),
                                    "q": np.round(v.wave_q[::dec], 4).tolist(),
                                    "start": v.wave_start, "step": dec, "frame_start": v.frame_start}}
        return out


def _num(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def _clean(d: dict) -> dict:
    """Replace NaN/inf (not valid JSON) with None."""
    return {k: (_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
