"""Online channel selection for the cognitive radio: WHICH channel to use in the next slot.

Each slot the secondary user (us):
  1. transmits one packet on the channel chosen from what it knew before the slot,
  2. senses all channels during the slot (busy / free, with sensing errors),
  3. gets feedback for its own packet (ACK/NACK and the receiver's SNR estimate).
So every decision uses only the past, as in a real radio.

Policies (all share choose / observe / feedback):
  random       no sensing, a random channel per slot                       (baseline)
  hop          the original rule: stay while free, else the free channel idle longest
               (same decisions as occupancy.policy_sense_and_hop)        (baseline)
  predict      learns each channel's on/off statistics (two-state Markov, counted from
               sensing) and picks the channel most likely to be free NEXT slot
  quality      also learns each channel's link SNR from our own packets and picks the
               highest EXPECTED GOODPUT = P(free next) x best goodput at that SNR (from the
               link-adaptation table); hops only if the gain beats a hop margin.
               harm_weight > 0 subtracts harm_weight x P(busy next) x the best possible goodput,
               trading some of our goodput for fewer collisions with licensed users
  oracle       `quality` with the TRUE statistics and SNRs (an upper bound, not realisable)
"""
from __future__ import annotations

import numpy as np

from spectrum_intel.adaptation import LinkAdapter
from spectrum_intel.occupancy import PrimaryUser

POLICIES = ("random", "hop", "predict", "quality", "oracle")


class Selector:
    name = "base"

    def __init__(self, n_ch: int):
        self.n_ch = n_ch
        self.current = 0
        self.t = 0

    def choose(self) -> int:
        return self.current

    def observe(self, sensed: np.ndarray) -> None:
        """Sensing result of the slot just finished (True = busy)."""
        self.t += 1

    def feedback(self, ch: int, record) -> None:
        """Outcome of our own packet on channel ch (a link.PacketRecord)."""


class RandomSelector(Selector):
    name = "random"

    def __init__(self, n_ch: int, rng: np.random.Generator):
        super().__init__(n_ch)
        self.rng = rng

    def choose(self) -> int:
        self.current = int(self.rng.integers(self.n_ch))
        return self.current


class HopSelector(Selector):
    """The original policy, online: identical decisions to occupancy.policy_sense_and_hop."""
    name = "hop"

    def __init__(self, n_ch: int):
        super().__init__(n_ch)
        self.idle = np.zeros(n_ch)

    def observe(self, sensed: np.ndarray) -> None:
        free = ~np.asarray(sensed, bool)
        self.idle = np.where(free, self.idle + 1, 0)
        if not free[self.current] and free.any():
            self.current = int(np.argmax(np.where(free, self.idle, -np.inf)))
        self.t += 1


class PredictSelector(Selector):
    """Learns P(free next slot | sensed state now) per channel and picks the most likely free.

    Counts transitions in the sensed sequence (with a +1 prior so unseen channels look
    50/50). Stays on the current channel unless another is more likely free by `margin`.
    """
    name = "predict"

    def __init__(self, n_ch: int, margin: float = 0.05):
        super().__init__(n_ch)
        self.counts = np.ones((n_ch, 2, 2))          # counts[c, state_now, state_next], 0 = free, 1 = busy
        self.last = None
        self.margin = margin

    def p_free_next(self) -> np.ndarray:
        if self.last is None:
            return np.full(self.n_ch, 0.5)
        row = self.counts[np.arange(self.n_ch), self.last.astype(int)]
        return row[:, 0] / row.sum(axis=1)

    def score(self) -> np.ndarray:
        return self.p_free_next()

    def observe(self, sensed: np.ndarray) -> None:
        s = np.asarray(sensed, bool)
        if self.last is not None:
            self.counts[np.arange(self.n_ch), self.last.astype(int), s.astype(int)] += 1
        self.last = s
        sc = self.score()
        best = int(np.argmax(sc))
        if sc[best] > sc[self.current] + self.margin * abs(sc[self.current]) + 1e-12:
            self.current = best
        self.t += 1


class QualitySelector(PredictSelector):
    """Expected goodput per channel: P(free next) x best expected goodput at the channel's SNR.

    SNR per channel = exponential average of the receiver's reports from our own packets
    on that channel. Channels never tried get an optimistic prior so each is explored.
    Hops only when the best score beats the current one by `margin` (a switch costs time).
    """
    name = "quality"

    def __init__(self, n_ch: int, margin: float = 0.2, prior_snr_db: float = 20.0, alpha: float = 0.3,
                 harm_weight: float = 0.0):
        super().__init__(n_ch, margin)
        self.harm_weight = harm_weight
        self.adapter = LinkAdapter("goodput")
        self.snr = np.full(n_ch, np.nan)
        self.prior = prior_snr_db
        self.alpha = alpha

    def link_goodput(self, snr_db: float) -> float:
        return max(self.adapter.expected_goodput(snr_db).values())

    def snr_view(self) -> np.ndarray:
        return np.where(np.isnan(self.snr), self.prior, self.snr)

    def score(self) -> np.ndarray:
        g = np.array([self.link_goodput(s) for s in self.snr_view()])
        pf = self.p_free_next()
        score = pf * g
        if self.harm_weight:
            g_ref = max(self.adapter.rate_bps.values())        # best goodput any channel could give
            score = score - self.harm_weight * (1 - pf) * g_ref
        return score

    def feedback(self, ch: int, record) -> None:
        snr = getattr(record, "snr_est_db", None)
        if getattr(record, "detected", False) and snr is not None and np.isfinite(snr):
            self.snr[ch] = snr if np.isnan(self.snr[ch]) else (1 - self.alpha) * self.snr[ch] + self.alpha * snr


class OracleSelector(QualitySelector):
    """Upper bound: the TRUE transition probabilities and link SNRs (perfectly known)."""
    name = "oracle"

    def __init__(self, n_ch: int, p_on: np.ndarray, p_off: np.ndarray, snr_db: np.ndarray, margin: float = 0.2):
        super().__init__(n_ch, margin)
        self.p_on, self.p_off = np.asarray(p_on, float), np.asarray(p_off, float)
        self.snr = np.asarray(snr_db, float).copy()

    def p_free_next(self) -> np.ndarray:
        if self.last is None:
            return 1 - self.p_on / (self.p_on + self.p_off)
        return np.where(self.last, self.p_off, 1 - self.p_on)

    def feedback(self, ch, record):
        pass                                            # already knows the SNRs


def make_selector(name: str, n_ch: int, rng: np.random.Generator | None = None, users=None,
                  link_snr_db=None, margin: float | None = None, harm_weight: float = 0.0) -> Selector:
    if name == "random":
        return RandomSelector(n_ch, rng or np.random.default_rng(0))
    if name == "hop":
        return HopSelector(n_ch)
    kw = {} if margin is None else {"margin": margin}
    if name == "predict":
        return PredictSelector(n_ch, **kw)
    if name == "quality":
        return QualitySelector(n_ch, harm_weight=harm_weight, **kw)
    if name == "oracle":
        sel = OracleSelector(n_ch, [u.p_on for u in users], [u.p_off for u in users], link_snr_db, **kw)
        sel.harm_weight = harm_weight
        return sel
    raise ValueError(f"policy must be one of {POLICIES}")


# --------------------------------------------------------------------------
# The band: licensed users switching on/off, our link quality per channel, sensing
# --------------------------------------------------------------------------
# Same eight licensed users as scripts/run_spectrum_demo.py (signal, their SNR, on/off rates)
DEFAULT_USERS = (
    PrimaryUser("qpsk", 8, p_on=0.15, p_off=0.10),
    PrimaryUser("ofdm", 12, p_on=0.10, p_off=0.08),
    PrimaryUser("qam16", 15, p_on=0.08, p_off=0.15),
    PrimaryUser("bpsk", 2, p_on=0.06, p_off=0.20),
    PrimaryUser("interference", 10, p_on=0.05, p_off=0.25),
    PrimaryUser("ofdm", 4, p_on=0.12, p_off=0.12),
    PrimaryUser("qpsk", -2, p_on=0.10, p_off=0.15),
    PrimaryUser("qam16", 18, p_on=0.20, p_off=0.06),
)
# Our own link quality differs per channel (distance, antenna, local noise): SNR = base + offset.
# Chosen so the channel whose owner returns least often (4) is a poor link for us, and good
# links (3, 6) are not the quietest - collision-only and goodput-aware choices then differ.
LINK_OFFSETS_DB = (0.0, 4.0, -4.0, 6.0, -8.0, 2.0, 8.0, -2.0)
SLOT_S = 0.006                 # one packet per slot; the licensed users change state once per slot
CNN_SENSING = (1.0, 0.0148)    # (detection, false alarm) measured for the CNN in results/metrics.json
SENSING = {"perfect": (1.0, 0.0), "cnn": CNN_SENSING, "degraded": (0.9, 0.10), "poor": (0.7, 0.25)}


class Band:
    """Licensed users (two-state Markov), sensing errors and per-channel interference, one slot at a time.

    All randomness comes from its own generators, so every policy sees exactly the same
    users and sensing errors (common random numbers).
    """

    def __init__(self, users=DEFAULT_USERS, offsets_db=LINK_OFFSETS_DB, pd: float = CNN_SENSING[0],
                 pfa: float = CNN_SENSING[1], seed: int = 0):
        self.users = list(users)
        self.offsets = np.asarray(offsets_db, float)
        self.n_ch = len(self.users)
        self.pd, self.pfa = pd, pfa
        self.rng_pu = np.random.default_rng([seed, 101])
        self.rng_sense = np.random.default_rng([seed, 102])
        p_on = np.array([u.p_on for u in self.users])
        p_off = np.array([u.p_off for u in self.users])
        self.p_on, self.p_off = p_on, p_off
        self.active = self.rng_pu.random(self.n_ch) < p_on / (p_on + p_off)   # start in the long-run mix
        self.t = 0

    def advance(self) -> np.ndarray:
        """Move to the next slot; returns who is active in the new slot."""
        u = self.rng_pu.random(self.n_ch)
        self.active = np.where(self.active, u >= self.p_off, u < self.p_on)
        self.t += 1
        return self.active

    def sense(self) -> np.ndarray:
        return sense(self.active, self.pd, self.pfa, self.rng_sense)

    def link_snr(self, base_snr_db: float, ch: int) -> float:
        return float(base_snr_db + self.offsets[ch])

    def collision_channel(self, ch: int, our_snr_db: float):
        """If the licensed user is on, its signal adds as continuous interference at its own SNR."""
        from spectrum_intel.channel import ChannelConfig
        if not self.active[ch]:
            return None
        return ChannelConfig(jammer="pulsed", duty=1.0, sir_db=our_snr_db - self.users[ch].snr_db)


def sense(truth_now: np.ndarray, pd: float, pfa: float, rng: np.random.Generator) -> np.ndarray:
    """Imperfect sensing: busy detected with probability pd, free misread as busy with pfa."""
    u = rng.random(len(truth_now))
    return np.where(truth_now, u < pd, u < pfa)


class BandLink:
    """One secondary-user link working inside the band, one packet per slot.

    Per slot: choose a channel (from the past only) -> pick the modulation for that channel
    from our last SNR report there (goodput policy, Phase 6; same for every selector, so a
    comparison isolates the channel choice) -> send the packet through the real PHY, with the
    licensed user's signal added as interference if it is on -> sense the band -> feedback.
    """

    def __init__(self, selector: Selector, band: Band):
        self.selector, self.band = selector, band
        self.adapter = LinkAdapter("goodput")
        self.last_snr = np.full(band.n_ch, np.nan)       # our receiver's last SNR report per channel
        self.prev_ch = None
        self.log: list[dict] = []

    def modulation(self, ch: int) -> str:
        s = self.last_snr[ch]
        return "bpsk" if np.isnan(s) else self.adapter.choose(snr_override=float(s))

    def step(self, pkt, n_packets: int, base_snr_db: float, rng: np.random.Generator, *,
             channel=None, max_payload: int = 64, mod: str | None = None, **phy) -> dict:
        from spectrum_intel.link import transfer_packet
        b = self.band
        ch = self.selector.choose()
        snr = b.link_snr(base_snr_db, ch)
        collided = bool(b.active[ch])
        cfg = b.collision_channel(ch, snr) or channel
        mod = mod or self.modulation(ch)                    # a fixed modulation can be forced (dashboard)
        t = transfer_packet(pkt, n_packets, mod, snr, rng, max_payload=max_payload, channel=cfg, **phy)
        sensed = b.sense()
        active_now = b.active.copy()
        self.selector.observe(sensed)
        self.selector.feedback(ch, t.record)
        r = t.record
        if r.detected and r.snr_est_db is not None and np.isfinite(r.snr_est_db):
            self.last_snr[ch] = r.snr_est_db
        hop = self.prev_ch is not None and ch != self.prev_ch
        self.prev_ch = ch
        entry = {"slot": b.t, "ch": ch, "mod": mod, "snr_db": snr, "collided": collided, "ok": bool(r.crc_ok),
                 "data_bytes": r.data_bytes, "airtime_s": r.n_samples / phy.get("sample_rate_hz", 1e6), "hop": hop,
                 "active": active_now, "sensed": sensed}
        self.log.append(entry)
        b.advance()                                         # the licensed users move on to the next slot
        return {"transfer": t, **entry}


def summarise(log: list[dict], hop_s: float = 0.001, skip: int = 0) -> dict:
    """Collision rate (harm to licensed users), our goodput and hops, from a BandLink log.

    goodput = user data delivered intact / (airtime of our frames + hop_s per channel change),
    the same definition as the link metrics, so faster modulations on better channels count.
    """
    L = log[skip:]
    n = len(L)
    hops = sum(e["hop"] for e in L)
    bits = 8 * sum(e["data_bytes"] for e in L if e["ok"])
    time_s = sum(e["airtime_s"] for e in L) + hops * hop_s
    return {"slots": n, "collision_rate": sum(e["collided"] for e in L) / n,
            "delivery_rate": sum(e["ok"] for e in L) / n, "goodput_kbps": bits / time_s / 1e3,
            "hops_per_100": 100 * hops / n,
            "channel_use": np.bincount([e["ch"] for e in L], minlength=len(L[0]["active"])).tolist()}
