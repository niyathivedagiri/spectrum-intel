import json

import numpy as np
import pytest

from spectrum_intel import leo, leo_link
from spectrum_intel.adaptation import MCSAdapter, data_rate_bps, load_mcs_table


def test_general_pass_matches_the_overhead_pass():
    a, b = leo.overhead_pass(dt_s=1.0), leo.general_pass(max_elevation_deg=90, dt_s=1.0)
    assert len(a.time_s) == len(b.time_s)
    assert np.max(np.abs(a.range_rate_mps)) == pytest.approx(np.max(np.abs(b.range_rate_mps)), rel=1e-3)


def test_lower_passes_are_shorter_farther_and_slower_in_doppler():
    p = {e: leo.general_pass(max_elevation_deg=e) for e in (90, 30, 15)}
    assert p[90].time_s[-1] > p[30].time_s[-1] > p[15].time_s[-1]
    assert p[15].elevation_deg.max() == pytest.approx(15, abs=0.2)
    assert p[90].range_m.min() < p[30].range_m.min() < p[15].range_m.min()
    with pytest.raises(ValueError):
        leo.general_pass(max_elevation_deg=5)


def test_timeline_doppler_and_snr_shape():
    tl = leo_link.timeline(leo_link.LEOLinkConfig(dt_s=2.0))
    fd = tl["doppler_hz"]
    assert fd[0] > 40e3 and fd[-1] < -40e3                     # approaching -> positive, receding -> negative
    i = int(np.argmax(tl["snr_db"]))
    assert abs(fd[i]) < 2e3                                    # strongest and zero Doppler at the top
    assert np.max(np.abs(tl["doppler_rate_hz_s"])) * 0.006 < 10  # Doppler hardly changes within one packet


def test_offsets_per_doppler_mode():
    cfg = leo_link.LEOLinkConfig(dt_s=2.0, osc_ppm=2.0, timing_error_s=0.5)
    tl = leo_link.timeline(cfg)
    rng = lambda: np.random.default_rng(0)
    none, s0 = leo_link.offsets(cfg, tl, rng())
    pred, s1 = leo_link.offsets(leo_link.LEOLinkConfig(**{**vars(cfg), "doppler": "predicted"}), tl, rng())
    srch, s2 = leo_link.offsets(leo_link.LEOLinkConfig(**{**vars(cfg), "doppler": "search"}), tl, rng())
    assert np.max(np.abs(pred)) * 1e6 < 2e-6 * 2e9 + 0.5 * 700  # oscillator + timing error x Doppler rate
    assert np.max(np.abs(srch)) > 0.04 and s2 >= np.max(np.abs(srch))
    assert s1 == 0.0
    with pytest.raises(ValueError):
        leo_link.LEOLinkConfig(doppler="magic").validated()
    with pytest.raises(ValueError):
        leo_link.LEOLinkConfig(scheme="qpsk/5/6").validated()


def test_doppler_handling_decides_whether_the_link_works():
    base = dict(dt_s=60.0, packets_per_step=2, seed=1)
    none = leo_link.run_pass(leo_link.LEOLinkConfig(doppler="none", **base))["summary"]
    pred = leo_link.run_pass(leo_link.LEOLinkConfig(doppler="predicted", **base))["summary"]
    assert pred["link_up_fraction"] == 1.0 and pred["delivered_MB"] > 5 * max(none["delivered_MB"], 0.01)
    assert none["max_residual_cfo_khz"] > 40 and pred["max_residual_cfo_khz"] < 5


def test_mcs_table_and_adapter():
    t = load_mcs_table()
    assert len(t["per"]) == 12 and all(np.all(np.diff(v) <= 0) for v in t["per"].values())
    a = MCSAdapter()
    assert a.choose() == ("bpsk", "1/2")                      # nothing known yet: most robust
    assert a.choose(-8) == ("bpsk", "1/2") and a.choose(15) == ("qam16", "none")
    picks = [a.choose(s) for s in range(-8, 16)]
    assert [a.rate_bps[p] for p in picks] == sorted(a.rate_bps[p] for p in picks)   # faster as SNR rises
    assert data_rate_bps("qpsk", code="1/2") < data_rate_bps("qpsk")


def test_engine_leo_mode_and_auto_mcs():
    from spectrum_intel.engine import EngineConfig, LinkEngine
    e = LinkEngine(EngineConfig(leo=True, fec="auto", leo_s_per_packet=8, text="v" * 64 * 30)).run_to_end()
    s = e.state()
    json.dumps(s, allow_nan=False)
    assert s["leo"]["enabled"] and s["amc"]["policy"] == "mcs"
    assert len({r.code for r in e.records}) >= 2                # the code changes during the pass
    assert sum(o == "ok" for o in e.outcomes) >= 25
    snrs = [h["snr_set"] for h in s["history"]]
    assert max(snrs) - min(snrs) > 5                           # SNR follows the pass
    off = LinkEngine(EngineConfig(snr_db=6)).run_to_end()
    assert off.state()["leo"] == {"enabled": False}
    with pytest.raises(ValueError):
        e.update(leo_doppler="magic")
