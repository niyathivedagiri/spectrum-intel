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
  - wall-clock processing time per packet vs the packet's airtime (real-time factor),
  - optional band mode: the link lives in an 8-channel band shared with licensed users
    and a channel-selection policy (selection.py) picks the channel for every packet.
    With band mode off (the default) nothing changes.
  - optional LEO pass mode (leo_link.py): each packet moves the satellite pass on by
    leo_s_per_packet seconds (a time-lapse); SNR follows the link budget and the frequency
    offset follows the Doppler left after the chosen Doppler handling.
  - FEC "auto": adaptive modulation AND coding (adaptation.MCSAdapter) picks both per packet.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from spectrum_intel import channel as chm
from spectrum_intel import fec
from spectrum_intel import leo_link
from spectrum_intel.adaptation import POLICIES, LinkAdapter, MCSAdapter
from spectrum_intel import metrics as m
from spectrum_intel import payload as pl
from spectrum_intel import selection as sel
from spectrum_intel.link import PacketRecord, transfer_packet
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import reassemble, segment
from spectrum_intel.signals import constellation
from spectrum_intel.transmitter import HEADER_BITS, PILOT_SPACING, PREAMBLE, SPAN, SPS

DEFAULT_TEXT = (
    "Hello from the cognitive radio! This message is cut into packets of 64 bytes. Each packet gets a "
    "sequence number and a CRC, is mapped onto BPSK, QPSK or 16-QAM symbols, shaped into I/Q samples "
    "and sent through a noisy channel with an unknown phase, arrival time and frequency offset. The "
    "receiver knows none of these in advance: it finds the preamble, estimates the offset, tracks the "
    "pilots, decides each symbol and checks the CRC. Green text passed the check. Orange text arrived "
    "damaged but was kept. Red dots mark packets that were lost. Turn the SNR down and watch the "
    "constellation spread out and the errors appear; switch to 16-QAM and see how much more signal it needs.")
CHANNEL_FIELDS = ("fading", "k_factor_db", "doppler_hz", "multipath", "delay_spread_us", "echo_db",
                  "jammer", "sir_db", "jam_offset_khz", "duty")
RECEIVER_FIELDS = ("equalizer", "eq_taps", "pilot_window")
FEC_FIELDS = ("fec", "fec_decoding", "interleave")
LEO_FIELDS = ("leo", "leo_max_elev", "leo_doppler", "leo_s_per_packet")
FEC_CHOICES = fec.CODES + ("auto",)
AMC_FIELDS = ("adaptive", "amc_policy", "target_per", "snr_swing_db", "swing_period")
BAND_FIELDS = ("band", "band_policy", "harm_weight", "sensing")
LIVE_FIELDS = {"mod", "snr_db", "cfo", "tolerant", "rate_pps", *CHANNEL_FIELDS, *RECEIVER_FIELDS, *AMC_FIELDS,
               *BAND_FIELDS, *FEC_FIELDS, *LEO_FIELDS}
BAND_HISTORY = 48               # slots shown in the dashboard's occupancy grid
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
    # channel impairments beyond AWGN (see channel.ChannelConfig); defaults = plain AWGN
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
    # receiver options
    equalizer: str = "none"         # "none" or "ls" (preamble-trained, for multipath)
    eq_taps: int = 7
    pilot_window: int = 7           # pilots averaged per channel estimate
    # forward error correction (fec.py): convolutional code K=7, punctured; off by default
    fec: str = "none"               # "none", "1/2", "2/3", "3/4", or "auto" (modulation AND code adapted)
    fec_decoding: str = "soft"      # "soft" (LLRs, local noise), "soft_fixed" (preamble noise only) or "hard"
    interleave: bool = True         # spread each packet's coded bits so bursts of errors get split up
    # link adaptation (adaptation.LinkAdapter) and a slowly swinging SNR to exercise it
    adaptive: bool = False          # True: the modulation is chosen per packet from receiver feedback
    amc_policy: str = "olla"        # "threshold" (table only) or "olla" (table + learned offset)
    target_per: float = 0.1         # packet error rate the adaptation aims for
    snr_swing_db: float = 0.0       # SNR moves as snr_db + swing * sin(2 pi k / swing_period), k = packet index
    swing_period: int = 60
    # band mode (selection.py): 8 channels shared with licensed users, a policy picks the channel
    band: bool = False
    band_policy: str = "quality"    # random / hop / predict / quality / oracle
    harm_weight: float = 0.0        # quality/oracle only: penalty on the predicted chance of a collision
    sensing: str = "cnn"            # perfect / cnn / degraded / poor (detection and false-alarm rates)
    # LEO pass mode (leo_link.py): SNR and frequency offset follow a satellite pass
    leo: bool = False
    leo_max_elev: float = 90.0      # highest elevation of the pass (deg)
    leo_doppler: str = "predicted"  # none / search / predicted / predicted+search
    leo_s_per_packet: float = 2.0   # pass time between packets (time-lapse)

    def validated(self) -> "EngineConfig":
        if self.source not in ("text", "image"):
            raise ValueError("source must be 'text' or 'image'")
        if self.mod not in LINK_MODULATIONS:
            raise ValueError(f"mod must be one of {LINK_MODULATIONS}")
        if self.amc_policy not in POLICIES:
            raise ValueError(f"amc_policy must be one of {POLICIES}")
        if self.band_policy not in sel.POLICIES:
            raise ValueError(f"band_policy must be one of {sel.POLICIES}")
        if self.sensing not in sel.SENSING:
            raise ValueError(f"sensing must be one of {tuple(sel.SENSING)}")
        if self.fec not in FEC_CHOICES:
            raise ValueError(f"fec must be one of {FEC_CHOICES}")
        if self.leo_doppler not in leo_link.DOPPLER_MODES:
            raise ValueError(f"leo_doppler must be one of {leo_link.DOPPLER_MODES}")
        if self.fec_decoding not in fec.DECODINGS:
            raise ValueError(f"fec_decoding must be one of {fec.DECODINGS}")
        if self.equalizer not in ("none", "ls"):
            raise ValueError("equalizer must be 'none' or 'ls'")
        self.channel()                                   # checks fading / multipath / jammer names
        return replace(self,
                       text=str(self.text)[:MAX_TEXT_CHARS] or " ",
                       image_size=int(np.clip(self.image_size, 8, MAX_IMAGE_SIZE)),
                       snr_db=float(np.clip(self.snr_db, -20, 40)),
                       cfo=float(np.clip(self.cfo, -0.02, 0.02)),
                       seed=int(self.seed), max_payload=int(np.clip(self.max_payload, 8, 1024)),
                       rate_pps=float(np.clip(self.rate_pps, 0.2, 200)),
                       tolerant=bool(self.tolerant), gray=bool(self.gray),
                       k_factor_db=float(np.clip(self.k_factor_db, -10, 30)),
                       doppler_hz=float(np.clip(self.doppler_hz, 0, 5000)),
                       delay_spread_us=float(np.clip(self.delay_spread_us, 0, 64)),
                       echo_db=float(np.clip(self.echo_db, -30, 0)),
                       sir_db=float(np.clip(self.sir_db, -20, 60)),
                       jam_offset_khz=float(np.clip(self.jam_offset_khz, -400, 400)),
                       duty=float(np.clip(self.duty, 0.01, 1.0)),
                       eq_taps=int(np.clip(self.eq_taps, 1, 41)) | 1,          # odd
                       pilot_window=int(np.clip(self.pilot_window, 1, 15)),
                       adaptive=bool(self.adaptive), target_per=float(np.clip(self.target_per, 0.01, 0.5)),
                       snr_swing_db=float(np.clip(self.snr_swing_db, 0, 20)),
                       swing_period=int(np.clip(self.swing_period, 4, 2000)),
                       band=bool(self.band), harm_weight=float(np.clip(self.harm_weight, 0, 10)),
                       interleave=bool(self.interleave), leo=bool(self.leo),
                       leo_max_elev=float(np.clip(self.leo_max_elev, 15, 90)),
                       leo_s_per_packet=float(np.clip(self.leo_s_per_packet, 0.5, 20)))

    def channel(self) -> chm.ChannelConfig:
        """The channel impairments as a ChannelConfig (raises ValueError on an unknown model)."""
        return chm.ChannelConfig(**{k: getattr(self, k) for k in CHANNEL_FIELDS})


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
    chan_t_ms: np.ndarray = None    # time in the frame (ms) for the true channel gain
    chan_true_db: np.ndarray = None # |direct-path gain| over the frame, dB re its RMS (ground truth)
    est_t_ms: np.ndarray = None     # same time axis for the receiver's pilot-based estimate
    est_db: np.ndarray = None       # |receiver's gain estimate|, dB re its RMS (None if header lost)
    eq_taps: np.ndarray = None      # |LS equaliser taps| if the equaliser ran


def _rel_db(g: np.ndarray) -> np.ndarray:
    a = np.abs(g)
    return 20 * np.log10(np.maximum(a, 1e-6) / max(np.sqrt(np.mean(a ** 2)), 1e-12))


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
        self.adapter = LinkAdapter(c.amc_policy, c.target_per)
        self.mcs = MCSAdapter()                       # used when fec == "auto"
        self.codes: list[str] = []                    # FEC code used per packet
        self.leo_t: list[float | None] = []           # pass time of each packet (LEO mode)
        self._leo_cache = None
        self.bandlink: sel.BandLink | None = None     # created when band mode is first switched on
        self.channels: list[int | None] = []         # band channel used per packet (None outside band mode)
        self.collided: list[bool | None] = []        # whether that packet hit an active licensed user
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
            if (new.amc_policy, new.target_per) != (self.config.amc_policy, self.config.target_per):
                fb = self.adapter.snr_fb                       # keep the latest feedback, restart the loop
                self.adapter = LinkAdapter(new.amc_policy, new.target_per)
                self.adapter.snr_fb = fb
            self._band_update(self.config, new)
            self.config = new
        return restart

    # ---------------------------------------------------------------- band mode
    def _selector(self, c: EngineConfig, band: sel.Band) -> sel.Selector:
        return sel.make_selector(c.band_policy, band.n_ch, np.random.default_rng([c.seed, 9]), users=band.users,
                                 link_snr_db=c.snr_db + band.offsets, harm_weight=c.harm_weight)

    def _band_update(self, old: EngineConfig, new: EngineConfig):
        """Policy or harm-weight change: a fresh selector (it learns again); sensing: new error rates."""
        if self.bandlink is None:
            return
        if (old.band_policy, old.harm_weight) != (new.band_policy, new.harm_weight):
            self.bandlink.selector = self._selector(new, self.bandlink.band)
        self.bandlink.band.pd, self.bandlink.band.pfa = sel.SENSING[new.sensing]

    def _ensure_band(self) -> sel.BandLink:
        if self.bandlink is None:
            c = self.config
            pd, pfa = sel.SENSING[c.sensing]
            band = sel.Band(pd=pd, pfa=pfa, seed=c.seed)        # own random streams: the PHY's stay untouched
            self.bandlink = sel.BandLink(self._selector(c, band), band)
        return self.bandlink

    # ---------------------------------------------------------------- LEO pass mode
    def _leo(self) -> dict:
        c = self.config
        key = (c.leo_max_elev, c.leo_doppler, c.seed)
        if self._leo_cache is None or self._leo_cache["key"] != key:
            cfg = leo_link.LEOLinkConfig(max_elevation_deg=c.leo_max_elev, doppler=c.leo_doppler, dt_s=1.0, seed=c.seed)
            tl = leo_link.timeline(cfg)
            cfo, search = leo_link.offsets(cfg, tl, np.random.default_rng([c.seed, 1]))
            self._leo_cache = {"key": key, "tl": tl, "cfo": cfo, "search": search, "pass_s": float(tl["t_s"][-1])}
        return self._leo_cache

    def leo_at(self, k: int) -> dict:
        """Pass state when packet k is sent (the pass repeats if the message is longer than it)."""
        L = self._leo()
        t = (k * self.config.leo_s_per_packet) % L["pass_s"]
        f = lambda a: float(np.interp(t, L["tl"]["t_s"], a))
        return {"t_s": t, "elev_deg": f(L["tl"]["elev_deg"]), "snr_db": f(L["tl"]["snr_db"]),
                "doppler_khz": f(L["tl"]["doppler_hz"]) / 1e3, "cfo": f(L["cfo"]), "search": L["search"],
                "range_km": f(L["tl"]["range_km"])}

    def snr_for(self, k: int) -> float:
        """SNR used for packet k (the set SNR, plus the optional slow swing; the pass in LEO mode)."""
        c = self.config
        if c.leo:
            return self.leo_at(k)["snr_db"]
        return c.snr_db + c.snr_swing_db * float(np.sin(2 * np.pi * k / c.swing_period)) if c.snr_swing_db else c.snr_db

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
        k = len(self.records)
        pkt = self.packets[k]
        snr = self.snr_for(k)
        phy = dict(max_payload=c.max_payload, cfo=c.cfo, keep_corrupted=c.tolerant, channel=c.channel(),
                   equalizer=c.equalizer, eq_taps=c.eq_taps, pilot_window=c.pilot_window, sample_rate_hz=c.sample_rate_hz,
                   code=c.fec, decoding=c.fec_decoding, interleave=c.interleave)
        fixed_mod = None if c.adaptive else c.mod
        if c.fec == "auto":                                     # modulation AND code from the MCS table
            fixed_mod, phy["code"] = self.mcs.choose()
        if c.leo:
            L = self.leo_at(k)
            phy["cfo"] = L["cfo"] + c.cfo                        # Doppler left after the chosen handling
            phy["cfo_search"] = L["search"]
            self.leo_t.append(L["t_s"])
        else:
            self.leo_t.append(None)
        self.codes.append(phy["code"])
        t0 = time.perf_counter()
        if c.band:
            # the policy picks the channel; SNR = set SNR + that channel's offset; if its licensed
            # user is on, its signal is added as interference (replacing the jammer setting).
            # AUTO picks the modulation per channel from our last SNR report there (goodput policy).
            e = self._ensure_band().step(pkt, self.n_packets, snr, self.rng, mod=fixed_mod, **phy)
            t, snr = e["transfer"], e["snr_db"]
            self.channels.append(e["ch"])
            self.collided.append(e["collided"])
        else:
            mod = fixed_mod or self.adapter.choose()              # adaptation only knows past feedback
            t = transfer_packet(pkt, self.n_packets, mod, snr, self.rng, **phy)
            self.channels.append(None)
            self.collided.append(None)
        proc_ms = 1e3 * (time.perf_counter() - t0)
        if t.good is not None:
            self._good[t.good.seq] = t.good
        if t.rough is not None:
            self._rough.setdefault(t.rough.seq, t.rough)      # first damaged copy wins, as in link.send
        outcome = "ok" if t.good is not None else ("damaged" if t.rough is not None else "lost")
        self.adapter.update(t.record)                           # feedback for the next packet
        self.mcs.update(t.record)
        self.records.append(t.record)
        self.outcomes.append(outcome)
        self.snr_set.append(snr)
        self.process_ms.append(proc_ms)
        self.last = self._view(pkt.seq, t, outcome, proc_ms, snr)
        return self.last

    def run_to_end(self):
        while self.step() is not None:
            pass
        return self

    def _view(self, seq, t, outcome, proc_ms, snr) -> StepView:
        res, rx = t.res, t.rx
        const = res.data_symbols[HEADER_BITS:] if res.header_ok and res.data_symbols is not None else np.array([])
        p = _psd(rx)
        fs = self.config.sample_rate_hz
        freq = np.fft.fftshift(np.fft.fftfreq(len(p), 1 / fs)) / 1e3
        start = res.start if res.detected and res.start >= 0 else 0
        w0 = max(0, start - 16 * SPS)
        seg = rx[w0:w0 + 96 * SPS]
        # channel over the frame: ground truth (direct path) vs the receiver's pilot-based estimate
        g_true = getattr(t.chan, "gain", None)
        if g_true is None:
            g_true = np.ones(len(t.frame.iq), complex)          # plain AWGN: flat channel
        dec = max(1, len(g_true) // 240)
        ct = np.arange(0, len(g_true), dec)
        info = res.info or {}
        est_t = est = None
        if info.get("gain_track") is not None:
            j = np.arange(len(info["gain_track"])) * 4               # data-symbol index (gains[::4])
            sym_pos = len(PREAMBLE) + 1 + j * (PILOT_SPACING + 1) / PILOT_SPACING
            est_t = 1e3 * (sym_pos * SPS + SPAN * SPS // 2) / fs
            est = _rel_db(np.asarray(info["gain_track"]))
        eqw = info.get("eq_taps")
        return StepView(seq=seq, record=t.record, constellation=const[:800],
                        chan_t_ms=1e3 * ct / fs, chan_true_db=_rel_db(g_true)[ct], est_t_ms=est_t, est_db=est,
                        eq_taps=None if eqw is None else np.abs(eqw),
                        psd_freq_khz=freq, psd_db=10 * np.log10(p + 1e-12),
                        wave_i=seg.real, wave_q=seg.imag, wave_start=w0,
                        frame_start=res.start if res.detected else -1, sync_metric=float(res.sync_metric),
                        snr_set_db=snr, process_ms=proc_ms,
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
        out = {"config": asdict(c), "channel_desc": c.channel().describe(), "n_packets": self.n_packets, "sent": len(self.records), "done": self.done,
               "message_bytes": len(self.data), "payload": self.payload_view(),
               "report": _clean(self.report()), "report_recent": _clean(self.report(last=20)),
               "ideal": [[float(z.real), float(z.imag)] for z in constellation(self.records[-1].mod if self.records else c.mod)],
               "amc": self.adapter.summary() | {"enabled": c.adaptive},
               "snr_now_db": round(self.snr_for(len(self.records)), 3)}
        recs = self.records[-history:]
        off = len(self.records) - len(recs)
        out["history"] = [{"i": off + j, "seq": r.seq, "mod": r.mod, "snr_set": self.snr_set[off + j],
                           "snr_est": _num(r.snr_est_db), "outcome": self.outcomes[off + j],
                           "bit_errors": r.bit_errors, "n_bits": r.n_bits, "code": r.code,
                           "channel_bit_errors": r.channel_bit_errors, "n_channel_bits": r.n_channel_bits,
                           "process_ms": round(self.process_ms[off + j], 2),
                           "airtime_ms": round(1e3 * r.n_samples / c.sample_rate_hz, 3),
                           "ch": self.channels[off + j], "collided": self.collided[off + j],
                           "leo_t": self.leo_t[off + j]}
                          for j, r in enumerate(recs)]
        out["band"] = self.band_view()
        out["leo"] = self.leo_view()
        if c.fec == "auto":
            out["amc"] = self.mcs.summary() | {"enabled": True}
        if c.band and c.adaptive and self.bandlink is not None:
            # in band mode AUTO picks per channel, from our last SNR report on the channel chosen next
            bl = self.bandlink
            out["amc"] = out["amc"] | {"policy": "goodput per channel", "offset_db": 0.0,
                                       "next": bl.modulation(bl.selector.current)}
        v = self.last
        if v is not None:
            dec = max(1, len(v.wave_i) // 600)
            out["last"] = {"seq": v.seq, "outcome": v.outcome, "mod": v.record.mod,
                           "snr_set_db": v.snr_set_db, "snr_est_db": _num(v.record.snr_est_db),
                           "cfo_est": _num(v.record.cfo_est), "sync_metric": round(v.sync_metric, 3),
                           "bit_errors": v.record.bit_errors, "n_bits": v.record.n_bits, "code": v.record.code,
                           "channel_bit_errors": v.record.channel_bit_errors, "n_channel_bits": v.record.n_channel_bits,
                           "process_ms": round(v.process_ms, 2), "airtime_ms": round(v.airtime_ms, 3),
                           "constellation": np.round(np.c_[v.constellation.real, v.constellation.imag], 4).tolist(),
                           "psd": {"f_khz": np.round(v.psd_freq_khz, 2).tolist(),
                                   "db": np.round(v.psd_db, 2).tolist()},
                           "chan": {"t_ms": np.round(v.chan_t_ms, 4).tolist(),
                                    "true_db": np.round(np.maximum(v.chan_true_db, -40), 2).tolist(),
                                    "est_t_ms": None if v.est_t_ms is None else np.round(v.est_t_ms, 4).tolist(),
                                    "est_db": None if v.est_db is None else np.round(np.maximum(v.est_db, -40), 2).tolist(),
                                    "eq_taps": None if v.eq_taps is None else np.round(v.eq_taps, 4).tolist()},
                           "wave": {"i": np.round(v.wave_i[::dec], 4).tolist(),
                                    "q": np.round(v.wave_q[::dec], 4).tolist(),
                                    "start": v.wave_start, "step": dec, "frame_start": v.frame_start}}
        return out


    def leo_view(self) -> dict:
        """LEO pass mode display: the whole pass (decimated) and where the next packet is."""
        c = self.config
        if not c.leo:
            return {"enabled": False}
        L = self._leo()
        tl, step = L["tl"], max(1, len(L["tl"]["t_s"]) // 120)
        now = self.leo_at(len(self.records))
        return {"enabled": True, "mode": c.leo_doppler, "max_elev_deg": c.leo_max_elev,
                "s_per_packet": c.leo_s_per_packet, "pass_s": L["pass_s"], "search_khz": L["search"] * 1e3,
                "now": {k: round(v, 3) for k, v in now.items() if k != "search"} | {"cfo_khz": round(now["cfo"] * 1e3, 3)},
                "track": {"t_s": np.round(tl["t_s"][::step], 1).tolist(), "snr_db": np.round(tl["snr_db"][::step], 2).tolist(),
                          "doppler_khz": np.round(tl["doppler_hz"][::step] / 1e3, 2).tolist(),
                          "elev_deg": np.round(tl["elev_deg"][::step], 1).tolist(),
                          "residual_khz": np.round(L["cfo"][::step] * 1e3, 2).tolist()}}

    def band_view(self) -> dict:
        """Band mode display: who is on, what we sensed, which channel we used, over recent slots."""
        c = self.config
        if self.bandlink is None:
            return {"enabled": c.band, "started": False}
        bl, b = self.bandlink, self.bandlink.band
        log = bl.log[-BAND_HISTORY:]
        s = bl.selector
        out = {"enabled": c.band, "started": True, "policy": s.name, "n_ch": b.n_ch,
               "offsets_db": b.offsets.tolist(), "pu_snr_db": [u.snr_db for u in b.users],
               "pu_signal": [u.label for u in b.users], "pd": b.pd, "pfa": b.pfa,
               "slots": [{"slot": e["slot"], "ch": e["ch"], "collided": e["collided"], "ok": e["ok"], "mod": e["mod"],
                          "active": e["active"].astype(int).tolist(), "sensed": e["sensed"].astype(int).tolist()}
                         for e in log],
               "next_ch": int(s.current), "active_now": b.active.astype(int).tolist()}
        if bl.log:
            out["summary"] = {k: v for k, v in sel.summarise(bl.log).items() if k != "slots"} | {"slots": len(bl.log)}
            out["summary"]["goodput_kbps"] = round(out["summary"]["goodput_kbps"], 2)
        if hasattr(s, "p_free_next"):
            out["p_free_next"] = np.round(s.p_free_next(), 3).tolist()
        if hasattr(s, "snr_view"):
            out["learned_snr_db"] = [None if np.isnan(x) else round(float(x), 1) for x in s.snr]
        return out


def _num(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def _clean(d: dict) -> dict:
    """Replace NaN/inf (not valid JSON) with None."""
    return {k: (_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
