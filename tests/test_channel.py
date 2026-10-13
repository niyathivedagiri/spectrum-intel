import json

import numpy as np
import pytest

from spectrum_intel import channel as ch
from spectrum_intel import link
from spectrum_intel.engine import EngineConfig, LinkEngine
from spectrum_intel.packet import segment
from spectrum_intel.receiver import receive
from spectrum_intel.signals import rrc_taps
from spectrum_intel.transmitter import SPAN, SPS, build_frame

C = ch.ChannelConfig


@pytest.fixture(scope="module")
def frame():
    return build_frame(segment(bytes(range(64)), 64)[0].to_bytes(), "qpsk")


def test_all_impairments_off_is_exactly_awgn(frame):
    a, _ = ch.awgn_link(frame.iq, 5.0, np.random.default_rng(3), cfo=0.001)
    b, _ = ch.dynamic_link(frame.iq, 5.0, np.random.default_rng(3), C(), cfo=0.001)
    c, _ = ch.dynamic_link(frame.iq, 5.0, np.random.default_rng(3), None, cfo=0.001)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_config_validation_and_description():
    with pytest.raises(ValueError):
        C(fading="nakagami")
    with pytest.raises(ValueError):
        C(jammer="laser")
    assert C().describe() == "AWGN"
    assert "rayleigh" in C(fading="rayleigh").describe() and "two_ray" in C(multipath="two_ray").describe()


@pytest.mark.parametrize("cfg", [C(multipath="two_ray", delay_spread_us=8, echo_db=-3),
                                 C(multipath="exponential", delay_spread_us=3)])
def test_tap_profiles_have_unit_power(cfg):
    taps = ch.tap_profile(cfg, 1e6)
    assert sum(p for _, p in taps) == pytest.approx(1.0)
    assert taps[0][0] == 0
    if cfg.multipath == "two_ray":
        assert taps[1][0] == 8 and taps[1][1] / taps[0][1] == pytest.approx(10 ** -0.3)
    else:
        assert all(p1 > p2 for (_, p1), (_, p2) in zip(taps, taps[1:]))     # decays with delay


def test_rayleigh_gain_has_unit_average_power_and_deep_fades():
    rng = np.random.default_rng(0)
    g = np.array([ch.fading_process(1, 0.0, 1e6, rng)[0] for _ in range(4000)])
    assert np.mean(np.abs(g) ** 2) == pytest.approx(1.0, rel=0.08)
    assert np.mean(np.abs(g) ** 2 < 0.1) == pytest.approx(1 - np.exp(-0.1), abs=0.03)   # P(10 dB fade) theory


def test_doppler_makes_the_gain_change_within_a_frame():
    rng = np.random.default_rng(1)
    still = ch.fading_process(4000, 0.0, 1e6, rng)
    moving = ch.fading_process(4000, 500.0, 1e6, rng)
    assert np.ptp(np.abs(still)) == 0 and np.ptp(np.abs(moving)) > 0.05


def test_strong_line_of_sight_is_nearly_flat(frame):
    rng = np.random.default_rng(2)
    gains = [ch.dynamic_link(frame.iq, 30, rng, C(fading="rician", k_factor_db=30))[1].gain[0] for _ in range(200)]
    assert np.std(np.abs(gains)) < 0.05


@pytest.mark.parametrize("kind", ["tone", "pulsed"])
def test_jammer_has_the_requested_average_power(frame, kind):
    cfg = C(jammer=kind, sir_db=0.0)
    rx_j, info = ch.dynamic_link(frame.iq, 80.0, np.random.default_rng(4), cfg, phase=0.0, delay=100)
    rx_0, _ = ch.dynamic_link(frame.iq, 80.0, np.random.default_rng(4), C(), phase=0.0, delay=100)  # same draws, no jammer
    jam = rx_j - rx_0
    assert np.mean(np.abs(jam) ** 2) == pytest.approx(info.jammer_power, rel=0.25)
    if kind == "pulsed":
        assert np.mean(np.abs(jam) > 1e-6) == pytest.approx(cfg.duty, abs=0.03)


def _success(cfg, n=30, eq="none", taps=7, snr=20.0, seed=5):
    rng = np.random.default_rng(seed)
    pk = segment(bytes(range(256)) * 2, 64)
    return np.mean([link.transfer_packet(pk[i % 8], 8, "qpsk", snr, rng, channel=cfg, equalizer=eq,
                                         eq_taps=taps).record.crc_ok for i in range(n)])


def test_equaliser_removes_a_one_symbol_echo():
    cfg = C(multipath="two_ray", delay_spread_us=8, echo_db=-3)
    assert _success(cfg) < 0.2
    assert _success(cfg, eq="ls") >= 0.95


def test_rayleigh_follows_the_outage_law():
    # AWGN 50% threshold for 64-byte QPSK is ~1.1 dB; 10 dB above it the outage law gives exp(-0.1) = 0.90
    s = _success(C(fading="rayleigh"), n=120, snr=11.1, seed=9)
    assert s == pytest.approx(np.exp(-0.1), abs=0.08)


def test_sync_finds_a_preamble_that_arrives_in_a_fade(frame):
    # gain rises from 0.15 to 1 across the frame: the preamble is weak, the data 16 dB stronger.
    # The strongest RAW correlation then lies in the data; the receiver must still find the preamble.
    ramp = np.linspace(0.15, 1.0, len(frame.iq))
    rx = np.concatenate([np.zeros(200, complex), ramp * frame.iq, np.zeros(256, complex)])
    rx = rx + 1e-4 * (np.random.default_rng(6).standard_normal(len(rx)) + 1j * np.random.default_rng(7).standard_normal(len(rx)))
    res = receive(rx)
    assert res.detected and res.header_ok
    assert res.start == 200 + SPAN * SPS                    # TX + RX filter delays, 32 samples each


def test_engine_channel_settings_are_live_and_match_link_send():
    cfg = EngineConfig(snr_db=15.0, fading="rayleigh", doppler_hz=200, multipath="two_ray", delay_spread_us=4,
                       equalizer="ls", eq_taps=7, pilot_window=3)
    e = LinkEngine(cfg)
    e.step()
    assert e.update(jammer="tone", sir_db=12.0) is False             # live: no restart
    e.run_to_end()
    s = e.state()
    json.dumps(s, allow_nan=False)
    assert "tone jammer" in s["channel_desc"] and s["last"]["chan"]["eq_taps"] is not None
    with pytest.raises(ValueError):
        e.update(fading="nakagami")
    # with the settings fixed for the whole run, the engine reproduces link.send exactly
    e2 = LinkEngine(cfg).run_to_end()
    ref = link.send(e2.data, cfg.mod, cfg.snr_db, np.random.default_rng(cfg.seed), channel=cfg.channel(),
                    equalizer="ls", eq_taps=7, pilot_window=3)
    assert e2.received()[0] == ref.received and [vars(a) for a in e2.records] == [vars(b) for b in ref.packets]


def test_rrc_delay_assumption():
    assert (len(rrc_taps(0.35, SPS, SPAN)) - 1) // 2 == SPAN * SPS // 2
