from types import SimpleNamespace

import numpy as np
import pytest

from spectrum_intel import selection as se
from spectrum_intel.occupancy import PrimaryUser, markov_activity, policy_sense_and_hop
from spectrum_intel.packet import segment


def rec(snr, ok=True, detected=True):
    return SimpleNamespace(crc_ok=ok, snr_est_db=snr, detected=detected)


def run_online(sel, sensed):
    """Feed a sensed matrix slot by slot; return the channel chosen in each slot."""
    out = []
    for t in range(len(sensed)):
        out.append(sel.choose())
        sel.observe(sensed[t])
    return np.array(out)


def test_hop_selector_matches_original_policy():
    rng = np.random.default_rng(1)
    users = list(se.DEFAULT_USERS)
    act = markov_activity(users, 400, rng)
    sensed = se.sense(act.ravel(), 0.9, 0.05, rng).reshape(act.shape)
    assert np.array_equal(run_online(se.HopSelector(len(users)), sensed), policy_sense_and_hop(sensed))


def test_predict_learns_the_quiet_channel():
    users = [PrimaryUser("qpsk", 5, p_on=0.5, p_off=0.1), PrimaryUser("qpsk", 5, p_on=0.02, p_off=0.5),
             PrimaryUser("qpsk", 5, p_on=0.4, p_off=0.2)]
    act = markov_activity(users, 1500, np.random.default_rng(2))
    ch = run_online(se.PredictSelector(3), act)
    assert np.mean(ch[500:] == 1) > 0.9                     # settles on the channel that is rarely used
    assert np.mean(act[np.arange(500, 1500), ch[500:]]) < 0.1


def test_quality_prefers_the_better_link_when_both_are_free():
    sel = se.QualitySelector(2, prior_snr_db=20.0)
    for _ in range(20):
        sel.feedback(0, rec(0.0))
        sel.feedback(1, rec(15.0))
        sel.observe(np.array([False, False]))
    assert sel.choose() == 1
    assert sel.snr[1] == pytest.approx(15.0)
    sel.feedback(1, rec(float("nan"), ok=False, detected=False))   # a lost frame tells nothing about SNR
    assert sel.snr[1] == pytest.approx(15.0)


def test_quality_hop_margin_stops_small_moves():
    sel = se.QualitySelector(2, margin=0.5)
    sel.snr[:] = [10.0, 11.0]                                 # channel 1 is only slightly better
    sel.observe(np.array([False, False]))
    assert sel.choose() == 0
    sel0 = se.QualitySelector(2, margin=0.0)
    sel0.snr[:] = [10.0, 11.0]
    sel0.observe(np.array([False, False]))
    assert sel0.choose() == 1


def test_harm_weight_avoids_a_good_but_busy_channel():
    # channel 0: great link but usually busy; channel 1: poorer link, usually free
    def make(h):
        s = se.QualitySelector(2, margin=0.0, harm_weight=h)
        s.snr[:] = [20.0, 8.0]
        s.counts[0] = [[3, 2], [3, 2]]                          # P(free next) = 0.6
        s.counts[1] = [[9, 1], [9, 1]]                          # P(free next) = 0.9
        s.observe(np.array([False, False]))
        return s
    assert make(0.0).choose() == 0
    assert make(1.0).choose() == 1


def test_band_duty_cycle_and_sensing_rates():
    band = se.Band(pd=0.8, pfa=0.1, seed=3)
    act, sen = [], []
    for _ in range(20000):
        act.append(band.active.copy())
        sen.append(band.sense())
        band.advance()
    act, sen = np.array(act), np.array(sen)
    expect = band.p_on / (band.p_on + band.p_off)
    assert np.allclose(act.mean(0), expect, atol=0.04)
    assert sen[act].mean() == pytest.approx(0.8, abs=0.01)
    assert sen[~act].mean() == pytest.approx(0.1, abs=0.01)


def test_band_is_reproducible_per_seed():
    a, b = se.Band(seed=4), se.Band(seed=4)
    for _ in range(50):
        assert np.array_equal(a.advance(), b.advance())
        assert np.array_equal(a.sense(), b.sense())


def test_collision_channel_uses_the_licensed_users_power():
    band = se.Band(seed=0)
    band.active[:] = False
    assert band.collision_channel(2, 6.0) is None
    band.active[2] = True
    cfg = band.collision_channel(2, 6.0)
    assert cfg.jammer == "pulsed" and cfg.duty == 1.0
    assert cfg.sir_db == pytest.approx(6.0 - band.users[2].snr_db)
    assert band.link_snr(6.0, 3) == pytest.approx(6.0 + se.LINK_OFFSETS_DB[3])


def test_summarise_maths():
    act = np.zeros(2, bool)
    log = [{"ch": 0, "collided": True, "ok": False, "data_bytes": 64, "airtime_s": 0.002, "hop": False, "active": act},
           {"ch": 1, "collided": False, "ok": True, "data_bytes": 64, "airtime_s": 0.002, "hop": True, "active": act},
           {"ch": 1, "collided": False, "ok": True, "data_bytes": 32, "airtime_s": 0.001, "hop": False, "active": act}]
    s = se.summarise(log, hop_s=0.001)
    assert s["collision_rate"] == pytest.approx(1 / 3)
    assert s["delivery_rate"] == pytest.approx(2 / 3)
    assert s["goodput_kbps"] == pytest.approx(8 * 96 / 0.006 / 1e3)
    assert s["hops_per_100"] == pytest.approx(100 / 3)
    assert s["channel_use"] == [1, 2]
    assert se.summarise(log, skip=1)["collision_rate"] == 0.0


def test_bandlink_runs_through_the_real_phy():
    band = se.Band(seed=5)
    sel = se.make_selector("quality", band.n_ch)
    link = se.BandLink(sel, band)
    pk = segment(bytes(range(64)) * 4, 64)
    for k in range(25):
        e = link.step(pk[k % 4], 4, 12.0, np.random.default_rng([5, k]))
        assert e["collided"] == bool(e["active"][e["ch"]])
    s = se.summarise(link.log)
    assert 0 <= s["collision_rate"] <= 1 and s["goodput_kbps"] > 0
    assert np.isfinite(sel.snr).sum() >= 1                    # learned the SNR of the channels it used


def test_make_selector_and_oracle():
    users = se.DEFAULT_USERS
    for p in se.POLICIES:
        s = se.make_selector(p, 8, np.random.default_rng(0), users=users, link_snr_db=np.zeros(8))
        assert 0 <= s.choose() < 8
    o = se.make_selector("oracle", 8, users=users, link_snr_db=np.zeros(8))
    o.observe(np.array([True] + [False] * 7))
    pf = o.p_free_next()
    assert pf[0] == pytest.approx(users[0].p_off) and pf[1] == pytest.approx(1 - users[1].p_on)
    with pytest.raises(ValueError):
        se.make_selector("magic", 8)


def test_engine_band_mode_reports_and_switches_policy():
    import json
    from spectrum_intel.engine import EngineConfig, LinkEngine
    e = LinkEngine(EngineConfig(band=True, adaptive=True, snr_db=8, text="y" * 64 * 30))
    assert e.state()["band"] == {"enabled": True, "started": False}
    e.run_to_end()
    s = e.state()
    json.dumps(s, allow_nan=False)
    b = s["band"]
    assert b["started"] and b["policy"] == "quality" and b["summary"]["slots"] == 30
    assert all(r["ch"] is not None and isinstance(r["collided"], bool) for r in s["history"])
    assert len(b["slots"]) == 30 and len(b["slots"][0]["active"]) == 8
    for r, slot in zip(s["history"], b["slots"]):                 # set SNR = base + the channel's offset
        assert r["snr_set"] == pytest.approx(8 + se.LINK_OFFSETS_DB[slot["ch"]])
    e.update(band_policy="hop", sensing="poor")
    assert e.bandlink.selector.name == "hop" and e.bandlink.band.pfa == 0.25
    with pytest.raises(ValueError):
        e.update(band_policy="magic")
    with pytest.raises(ValueError):
        e.update(sensing="magic")


def test_engine_band_off_is_unchanged_and_fixed_mod_is_respected():
    from spectrum_intel.engine import EngineConfig, LinkEngine
    a = LinkEngine(EngineConfig(snr_db=6)).run_to_end()
    b = LinkEngine(EngineConfig(snr_db=6, band_policy="hop", sensing="poor", harm_weight=2)).run_to_end()
    assert [r.bit_errors for r in a.records] == [r.bit_errors for r in b.records]   # band settings inert when off
    assert a.state()["band"] == {"enabled": False, "started": False}
    c = LinkEngine(EngineConfig(snr_db=6, band=True, mod="qam16")).run_to_end()
    assert {r.mod for r in c.records} == {"qam16"}
