import json

import numpy as np
import pytest

from spectrum_intel import channel as ch
from spectrum_intel import experiments as ex
from spectrum_intel import link, metrics, payload
from spectrum_intel.coding import crc16, crc_append, crc_check
from spectrum_intel.modulation import LINK_MODULATIONS, demodulate, modulate, nearest_index, symbol_indices
from spectrum_intel.packet import Packet, reassemble, segment
from spectrum_intel.receiver import receive
from spectrum_intel.transmitter import PREAMBLE, build_frame, header_bits, parse_header


# ---- payload: text / bytes / bits ------------------------------------------------
def test_bits_msb_first():
    assert "".join(map(str, payload.text_to_bits("H"))) == "01001000"


@pytest.mark.parametrize("text", ["HELLO", "Hello satellite", "naïve ✓ 🛰️", ""])
def test_text_round_trip(text):
    assert payload.bits_to_text(payload.text_to_bits(text)) == text


def test_damaged_utf8_is_visible_not_crash():
    bad = payload.bytes_to_text(b"\xff\xfeHi")
    assert "�" in bad and bad.endswith("Hi")


def test_bytes_bits_round_trip_random():
    data = bytes(np.random.default_rng(0).integers(0, 256, 300, dtype=np.uint8))
    assert payload.bits_to_bytes(payload.bytes_to_bits(data)) == data


# ---- CRC -------------------------------------------------------------------------
def test_crc16_standard_check_value():
    assert crc16(b"123456789") == 0x29B1


def test_crc_detects_every_single_bit_flip():
    block = bytearray(crc_append(b"spectrum intelligence"))
    for byte in range(len(block)):
        for bit in range(8):
            block[byte] ^= 1 << bit
            assert crc_check(bytes(block))[1] is False
            block[byte] ^= 1 << bit
    assert crc_check(bytes(block)) == (b"spectrum intelligence", True)


# ---- packets ----------------------------------------------------------------------
def test_packet_round_trip_and_corruption():
    p = Packet(3, 7, b"abc")
    q, ok = Packet.from_bytes(p.to_bytes())
    assert ok and (q.seq, q.total, q.data) == (3, 7, b"abc")
    damaged = bytearray(p.to_bytes())
    damaged[1] ^= 0x01
    assert Packet.from_bytes(bytes(damaged)) == (None, False)


def test_segment_and_reassemble_reports_missing():
    data = bytes(range(200))
    pkts = segment(data, 64)
    assert [len(p.data) for p in pkts] == [64, 64, 64, 8]
    out, missing = reassemble([pkts[0], pkts[2], pkts[3]], 4, chunk=64, expected_len=200)
    assert missing == [1] and len(out) == 200
    assert out[:64] == data[:64] and out[64:128] == bytes(64) and out[128:] == data[128:]


# ---- modulation -------------------------------------------------------------------
@pytest.mark.parametrize("mod", LINK_MODULATIONS)
def test_modulation_round_trip(mod):
    bits = np.random.default_rng(1).integers(0, 2, 400)
    assert np.array_equal(demodulate(modulate(bits, mod), mod)[:400], bits)


@pytest.mark.parametrize("mod", LINK_MODULATIONS)
def test_symbol_indices_match_nearest(mod):
    bits = np.random.default_rng(2).integers(0, 2, 160)
    assert np.array_equal(nearest_index(modulate(bits, mod), mod), symbol_indices(bits, mod))


# ---- transmitter --------------------------------------------------------------------
def test_preamble_is_m_sequence():
    ac = np.correlate(PREAMBLE, PREAMBLE, mode="full").real
    assert ac.max() == 63 and len(PREAMBLE) == 63


@pytest.mark.parametrize("mod", LINK_MODULATIONS)
def test_header_round_trip(mod):
    assert parse_header(header_bits(mod, 1234)) == (mod, 1234, True)


def test_header_detects_corruption():
    h = header_bits("qpsk", 10)
    h[5] ^= 1
    assert parse_header(h)[2] is False


# ---- channel ------------------------------------------------------------------------
def test_awgn_link_sets_noise_from_snr():
    rng = np.random.default_rng(3)
    f = build_frame(bytes(500), "qpsk")
    rx, info = ch.awgn_link(f.iq, 7.0, rng, delay=100)
    noise = rx[:100]                                   # before the burst arrives
    assert info.delay == 100
    assert 10 * np.log10(info.signal_power / np.mean(np.abs(noise) ** 2)) == pytest.approx(7.0, abs=1.5)


# ---- receiver -------------------------------------------------------------------------
@pytest.mark.parametrize("mod", LINK_MODULATIONS)
def test_receiver_recovers_bytes_at_high_snr(mod):
    rng = np.random.default_rng(4)
    data = b"Hello satellite, this is a test packet."
    rx, _ = ch.awgn_link(build_frame(data, mod).iq, 25.0, rng)
    res = receive(rx)
    assert res.header_ok and res.mod == mod and res.payload == data
    assert res.snr_est_db == pytest.approx(25.0, abs=2.0)


@pytest.mark.parametrize("cfo", [0.002, -0.004])
def test_receiver_corrects_small_frequency_offset(cfo):
    rng = np.random.default_rng(5)
    rx, _ = ch.awgn_link(build_frame(b"frequency offset" * 8, "qam16").iq, 22.0, rng, cfo=cfo)
    res = receive(rx)
    assert res.payload == b"frequency offset" * 8
    assert res.cfo_est == pytest.approx(cfo, abs=2e-4)


def test_receiver_coarse_search_handles_leo_doppler():
    rng = np.random.default_rng(6)
    rx, _ = ch.awgn_link(build_frame(b"LEO", "qpsk").iq, 12.0, rng, cfo=0.045)
    res = receive(rx, cfo_search=0.05)
    assert res.payload == b"LEO" and res.cfo_est == pytest.approx(0.045, abs=5e-4)


def test_no_frame_in_pure_noise():
    rng = np.random.default_rng(7)
    noise = np.sqrt(0.5) * (rng.standard_normal(5000) + 1j * rng.standard_normal(5000))
    assert receive(noise).detected is False


# ---- end-to-end link --------------------------------------------------------------------
def test_link_exact_text_at_good_snr():
    r = link.send(payload.text_to_bytes("HELLO " * 30), "qpsk", 10.0, np.random.default_rng(8))
    assert r.exact and r.missing == [] and r.packet_success_rate == 1.0 and r.raw_ber() == 0.0


def test_link_reports_losses_at_terrible_snr():
    r = link.send(bytes(range(256)), "qam16", -6.0, np.random.default_rng(9))
    assert not r.exact and len(r.missing) > 0 and r.packet_success_rate < 1.0


def test_link_is_deterministic_with_seed():
    a = link.send(b"same seed" * 20, "qam16", 4.0, np.random.default_rng(10))
    b = link.send(b"same seed" * 20, "qam16", 4.0, np.random.default_rng(10))
    assert a.received == b.received and [p.bit_errors for p in a.packets] == [p.bit_errors for p in b.packets]


# ---- metrics ----------------------------------------------------------------------------
def test_theory_ber_reference_point():
    assert metrics.theory_ber("bpsk", 9.6) == pytest.approx(1e-5, rel=0.1)   # textbook: 9.6 dB -> 1e-5


def test_snr_conversions():
    assert float(metrics.esn0_from_snr(0.0)) == pytest.approx(9.03, abs=0.01)
    assert float(metrics.ebn0_from_snr(0.0, 4)) == pytest.approx(3.01, abs=0.01)


def test_ber_and_wilson_interval():
    assert metrics.ber(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0])) == 0.25
    lo, hi = metrics.wilson_interval(0, 1000)
    assert lo == 0.0 and 0 < hi < 0.005


# ---- experiment framework ------------------------------------------------------------------
def test_runs_never_overwrite(tmp_path):
    a = ex.new_run("demo", {"x": 1}, 5, base=tmp_path)
    b = ex.new_run("demo", {"x": 2}, 5, base=tmp_path)
    assert a != b and a.exists() and b.exists()
    assert json.loads((a / "config.json").read_text())["config"] == {"x": 1}
    ex.save_csv(b, "t.csv", [{"snr": 1.0, "ber": 0.1}])
    assert (b / "t.csv").read_text().splitlines()[0] == "snr,ber"
    assert ex.latest_run("demo", base=tmp_path) == b
