import json
from types import SimpleNamespace

import numpy as np
import pytest

from spectrum_intel import adaptation as ad
from spectrum_intel.engine import EngineConfig, LinkEngine


@pytest.fixture(scope="module")
def table():
    return ad.load_table()


def rec(ok=True, snr=10.0, detected=True, mod="qpsk"):
    return SimpleNamespace(crc_ok=ok, snr_est_db=snr, detected=detected, mod=mod)


def test_table_is_monotone_and_ordered(table):
    for mod, per in table["per"].items():
        assert all(a >= b for a, b in zip(per, per[1:])), mod          # PER never rises with SNR
        assert per[0] == 1.0 and per[-1] == 0.0
    t = {m: ad.required_snr(table, m, 0.1) for m in table["per"]}
    assert t["bpsk"] < t["qpsk"] < t["qam16"]
    assert 6 < t["qam16"] - t["qpsk"] < 9                               # 16-QAM needs ~7 dB more than QPSK


def test_required_snr_interpolates():
    t = {"snr_db": [0, 1, 2], "per": {"m": [1.0, 0.5, 0.0]}}
    assert ad.required_snr(t, "m", 0.25) == pytest.approx(1.5)
    assert ad.required_snr(t, "m", 0.0) == pytest.approx(2.0)


def test_rates_match_measured_peak_goodput():
    r = {m: ad.data_rate_bps(m) / 1e3 for m in ("bpsk", "qpsk", "qam16")}
    assert r == pytest.approx({"bpsk": 89.1, "qpsk": 155.3, "qam16": 247.1}, abs=0.1)


def test_threshold_choices_and_fallback():
    a = ad.LinkAdapter("threshold")
    assert a.choose() == "bpsk"                                         # no feedback yet: most robust
    a.update(rec(snr=20))
    assert a.choose() == "qam16"
    a.update(rec(snr=5))
    assert a.choose() == "qpsk"
    a.update(rec(ok=False, detected=False, snr=float("nan")))
    assert a.snr_fb is None and a.choose() == "bpsk"                     # lost frame: back to robust
    assert a.choose(snr_override=20) == "qam16"                          # genie uses the given SNR


def test_goodput_policy_takes_a_faster_mode_when_it_pays():
    thr, gp = ad.LinkAdapter("threshold"), ad.LinkAdapter("goodput")
    # at 2 dB QPSK misses the 10% error target but still delivers more data than BPSK
    assert thr.choose(snr_override=2.0) == "bpsk"
    assert gp.choose(snr_override=2.0) == "qpsk"
    eg = gp.expected_goodput(2.0)
    assert max(eg, key=eg.get) == "qpsk"


def test_olla_step_sizes():
    a = ad.LinkAdapter("olla", target_per=0.1, step_db=1.0)
    a.update(rec(ok=False))
    assert a.offset_db == pytest.approx(1.0)
    a.update(rec(ok=True))
    assert a.offset_db == pytest.approx(1.0 - 1.0 / 9)
    assert ad.LinkAdapter("threshold").offset_db == 0.0


def test_olla_learns_a_channel_worse_than_the_table(table):
    # the real channel needs 3 dB more than the AWGN table says; the outer loop must learn ~+3 dB
    rng = np.random.default_rng(0)
    a = ad.LinkAdapter("olla", target_per=0.1, step_db=0.5)
    true_snr, nacks, offsets = 12.0, [], []
    for _ in range(4000):
        a.snr_fb = true_snr
        mod = a.choose()
        ok = rng.random() >= ad.per_at(table, mod, true_snr - 3.0)
        a.update(rec(ok=ok, snr=true_snr, mod=mod))
        nacks.append(not ok)
        offsets.append(a.offset_db)
    assert np.mean(nacks[1000:]) == pytest.approx(0.1, abs=0.04)          # error rate pulled to the target
    assert 0.5 < np.mean(offsets[1000:]) < 5                             # learned a positive correction


def test_validation():
    with pytest.raises(ValueError):
        ad.LinkAdapter("magic")
    with pytest.raises(ValueError):
        ad.LinkAdapter("olla", target_per=1.5)


def test_slow_fading_trace_statistics():
    rng = np.random.default_rng(3)
    slow = ad.slow_fading_snr(4000, 10.0, 2.0, 0.005, rng)
    fast = ad.slow_fading_snr(4000, 10.0, 400.0, 0.005, rng)
    for x in (slow, fast):
        assert 10 * np.log10(np.mean(10 ** (x / 10))) == pytest.approx(10.0, abs=1.5)   # mean power kept
    corr = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
    assert corr(slow) > 0.95 and corr(fast) < 0.5                        # slow fading = predictable


def test_engine_auto_mode_switches_and_reports():
    cfg = EngineConfig(adaptive=True, amc_policy="goodput", snr_db=7, snr_swing_db=9, swing_period=24,
                       text="x" * 64 * 48)
    e = LinkEngine(cfg).run_to_end()
    assert {r.mod for r in e.records} == {"bpsk", "qpsk", "qam16"}
    assert e.records[0].mod == "bpsk"                                    # nothing known before packet 0
    s = e.state()
    json.dumps(s, allow_nan=False)
    assert s["amc"]["enabled"] and s["amc"]["policy"] == "goodput"
    assert e.report()["mod"] == "mixed"
    assert sum(o == "ok" for o in e.outcomes) / len(e.outcomes) > 0.85


def test_engine_policy_change_keeps_feedback_and_off_means_fixed():
    e = LinkEngine(EngineConfig(adaptive=True, snr_db=15))
    e.step()
    fb = e.adapter.snr_fb
    assert fb is not None
    e.update(amc_policy="threshold")
    assert e.adapter.policy == "threshold" and e.adapter.snr_fb == fb
    e.update(adaptive=False, mod="qam16")
    e.step()
    assert e.records[-1].mod == "qam16"
    with pytest.raises(ValueError):
        e.update(amc_policy="magic")
