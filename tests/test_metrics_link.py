import numpy as np
import pytest

from spectrum_intel import link
from spectrum_intel import metrics as m
from spectrum_intel import occupancy as oc
from spectrum_intel.modulation import bits_per_symbol


@pytest.fixture(scope="module")
def clean_qpsk():
    return link.send(bytes(range(200)), "qpsk", 15.0, np.random.default_rng(0))


def test_report_rates_are_consistent(clean_qpsk):
    r = m.link_report(clean_qpsk)
    assert r["packets"] == 4 and r["packet_success_rate"] == 1.0 and r["ber"] == 0.0 and r["ser"] == 0.0
    assert r["goodput_bps"] == pytest.approx(r["throughput_bps"])           # nothing lost
    assert r["throughput_bps"] < r["phy_rate_bps"] == 250_000               # 125 kBd x 2 bits
    assert r["overhead_fraction"] == pytest.approx(1 - r["throughput_bps"] / r["phy_rate_bps"])
    assert r["spectral_efficiency_bps_hz"] == pytest.approx(r["goodput_bps"] / m.occupied_bandwidth_hz())


def test_latency_is_airtime(clean_qpsk):
    r = m.link_report(clean_qpsk, sample_rate_hz=2e6)
    samples = [p.n_samples for p in clean_qpsk.packets]
    assert r["latency_message_ms"] == pytest.approx(1e3 * sum(samples) / 2e6)
    assert r["latency_frame_ms"] == pytest.approx(1e3 * np.mean(samples) / 2e6)


def test_occupied_bandwidth():
    assert m.occupied_bandwidth_hz(1e6) == pytest.approx(1e6 / 8 * 1.35)


@pytest.mark.parametrize("mod", ["qpsk", "qam16"])
def test_ser_bounds_ber(mod):
    r = m.link_report(link.send(bytes(range(256)) * 2, mod, {"qpsk": 0.0, "qam16": 6.0}[mod],
                                np.random.default_rng(1)))
    k = bits_per_symbol(mod)
    assert r["ber"] > 0
    assert r["ber"] <= r["ser"] <= k * r["ber"] + 1e-12     # each symbol error costs 1..k bit errors


def test_goodput_zero_when_all_packets_fail():
    r = m.link_report(link.send(bytes(300), "qam16", -4.0, np.random.default_rng(2)))
    assert r["goodput_bps"] == 0.0 and r["packet_error_rate"] == 1.0 and r["throughput_bps"] > 0


def test_error_accounting_adds_up():
    res = link.send(bytes(range(256)) * 3, "qpsk", 0.5, np.random.default_rng(3))
    r = m.link_report(res)
    assert r["packets_lost"] + r["packets_crc_failed"] + round(r["packet_success_rate"] * r["packets"]) == r["packets"]


def test_sensing_metrics_match_existing_implementation():
    rng = np.random.default_rng(4)
    truth = rng.random((50, 8)) < 0.4
    sensed = truth ^ (rng.random((50, 8)) < 0.1)
    ref = oc.occupancy_accuracy(sensed, truth)
    assert m.detection_probability(sensed, truth) == pytest.approx(ref["detection_rate"])
    assert m.false_alarm_rate(sensed, truth) == pytest.approx(ref["false_alarm_rate"])
    choice = rng.integers(0, 8, 50)
    assert m.collision_rate(choice, truth) == pytest.approx(oc.evaluate_policy(choice, truth)["collision_rate"])
