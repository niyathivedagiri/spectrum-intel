import numpy as np
import pytest

from spectrum_intel import leo
from spectrum_intel import occupancy as oc
from spectrum_intel.spectrum import NOISE_LABEL


# ---- occupancy & channel selection -------------------------------------------
def test_markov_activity_long_run_share():
    users = [oc.PrimaryUser("qpsk", 10, p_on=0.1, p_off=0.3)]
    active = oc.markov_activity(users, 20_000, np.random.default_rng(0))
    assert active.mean() == pytest.approx(0.1 / 0.4, abs=0.03)


def test_activity_to_scene_inputs():
    users = [oc.PrimaryUser("ofdm", 5, 0.5, 0.5), oc.PrimaryUser("bpsk", 7, 0.5, 0.5)]
    active = np.array([[True, False], [False, True]])
    labels, snr = oc.activity_to_scene_inputs(users, active)
    assert labels[1, 0] == NOISE_LABEL and labels[0, 0] != NOISE_LABEL
    assert snr[0, 1] == 7


def test_sense_and_hop_avoids_known_busy_channels():
    truth = np.zeros((10, 3), dtype=bool)
    truth[:, 0] = True                     # channel 0 always busy
    choice = oc.policy_sense_and_hop(truth)
    r = oc.evaluate_policy(choice, truth)
    assert r["collision_rate"] == 0.0 and choice[-1] != 0


def test_fixed_policy_collides_on_busy_channel():
    truth = np.ones((5, 2), dtype=bool)
    assert oc.evaluate_policy(oc.policy_fixed(truth, 0), truth)["collision_rate"] == 1.0


def test_occupancy_accuracy():
    truth = np.array([[True, False], [False, False]])
    sensed = np.array([[True, True], [False, False]])
    r = oc.occupancy_accuracy(sensed, truth)
    assert r["detection_rate"] == 1.0 and r["false_alarm_rate"] == pytest.approx(1 / 3)


# ---- LEO -----------------------------------------------------------------------
def test_orbital_speed_550km():
    assert leo.orbital_speed(550e3) == pytest.approx(7590, rel=0.01)


def test_pass_geometry():
    p = leo.overhead_pass(550e3, min_elevation_deg=10)
    assert p.range_m.min() == pytest.approx(550e3, rel=0.01)       # overhead
    assert p.elevation_deg.max() > 89 and p.elevation_deg.min() >= 9.9
    assert 5 < p.time_s[-1] / 60 < 12                               # a few minutes long
    fd = leo.doppler_hz(p.range_rate_mps, 2e9)
    assert fd[0] > 0 > fd[-1]                                        # approaching, then leaving
    assert 40e3 < np.max(np.abs(fd)) < 50.7e3                        # below v/c * fc


def test_fspl_and_link_budget():
    assert leo.fspl_db(1000, 1e9) == pytest.approx(92.4, abs=0.1)    # textbook value
    assert leo.link_snr_db(550e3) > leo.link_snr_db(2000e3)
