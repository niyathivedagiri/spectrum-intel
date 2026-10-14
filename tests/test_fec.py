import json

import numpy as np
import pytest

from spectrum_intel import fec
from spectrum_intel.link import send, transfer_packet
from spectrum_intel.metrics import link_report
from spectrum_intel.modulation import LINK_MODULATIONS, demodulate, modulate
from spectrum_intel.packet import segment
from spectrum_intel.payload import bits_to_bytes, bytes_to_bits
from spectrum_intel.transmitter import build_frame, header_bits, parse_header


def bits(n, seed=0):
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def test_code_is_the_standard_k7_code():
    # impulse response: a single 1 produces the two generator polynomials (133, 171 octal), weight 10
    c = fec.conv_encode(np.array([1])).reshape(-1, 2)
    a = int("".join(map(str, c[:, 0])), 2)
    b = int("".join(map(str, c[:, 1])), 2)
    assert (a, b) == (0o133, 0o171)
    assert c.sum() == 10                                         # free distance of this code


def test_coded_lengths():
    assert [fec.coded_length(576, c) for c in fec.CODES] == [576, 1164, 873, 776]
    for c in fec.CODES[1:]:
        assert len(fec.encode(bits(576), c)) == fec.coded_length(576, c)
        assert fec.coded_length(576, c) * fec.RATE[c] == pytest.approx(576 + fec.TAIL, rel=0.01)


@pytest.mark.parametrize("code", fec.CODES)
@pytest.mark.parametrize("inter", [True, False])
def test_round_trip_without_errors(code, inter):
    b = bits(600, 1)
    assert np.array_equal(fec.decode(fec.hard_llr(fec.encode(b, code, inter)), 600, code, inter), b)


def test_corrects_scattered_errors():
    b = bits(576, 2)
    c = fec.encode(b, "1/2")
    rng = np.random.default_rng(3)
    for _ in range(20):
        e = c.copy()
        e[rng.choice(len(c), 23, replace=False)] ^= 1           # 2% of the coded bits wrong
        assert np.array_equal(fec.decode(fec.hard_llr(e), 576, "1/2"), b)


def test_interleaver_splits_a_burst():
    b = bits(576, 4)
    for inter, expect_ok in [(True, True), (False, False)]:
        c = fec.encode(b, "1/2", inter)
        e = c.copy()
        e[500:530] ^= 1                                          # 30 consecutive bits on air destroyed
        ok = np.array_equal(fec.decode(fec.hard_llr(e), 576, "1/2", inter), b)
        assert ok == expect_ok
    x = np.arange(1000)
    assert np.array_equal(fec.deinterleave(fec.interleave(x)), x)
    assert not np.array_equal(fec.interleave(x), x)


@pytest.mark.parametrize("mod", LINK_MODULATIONS)
def test_soft_demap_agrees_with_hard_and_scales(mod):
    b = bits(480, 5)
    rng = np.random.default_rng(6)
    y = modulate(b, mod) + 0.2 * (rng.normal(size=480 // {"bpsk": 1, "qpsk": 2, "qam16": 4}[mod]) +
                                  1j * rng.normal(size=480 // {"bpsk": 1, "qpsk": 2, "qam16": 4}[mod]))
    llr = fec.soft_demap(y, mod, 0.08)
    assert np.array_equal((llr < 0).astype(np.uint8), demodulate(y, mod))
    assert np.allclose(fec.soft_demap(y, mod, 0.16), llr / 2)       # noisier symbols -> less sure


def test_soft_decoding_beats_hard():
    rng = np.random.default_rng(7)
    err = {"soft": 0, "hard": 0}
    for k in range(40):
        b = bits(576, 100 + k)
        y = modulate(fec.encode(b, "1/2"), "bpsk") + 0.75 * rng.normal(size=1164)
        llr = fec.soft_demap(y, "bpsk", 0.75 ** 2)
        err["soft"] += int(np.sum(fec.decode(llr, 576, "1/2") != b))
        err["hard"] += int(np.sum(fec.decode(fec.hard_llr(llr < 0), 576, "1/2") != b))
    assert err["soft"] < err["hard"] / 3


def test_header_keeps_uncoded_ids_and_carries_the_code():
    for i, m in enumerate(LINK_MODULATIONS):
        expect = bits_to_bytes(header_bits(m, 72))[:2]
        assert int.from_bytes(expect, "big") == (i << 12) | 72      # same header as before FEC existed
        for c in fec.CODES:
            assert parse_header(header_bits(m, 72, c), with_code=True) == (m, c, 72, True)
    assert parse_header(header_bits("qpsk", 72, "1/2")) == (None, 0, False)   # old reader rejects coded frames
    assert parse_header(header_bits("qpsk", 72)) == ("qpsk", 72, True)


def test_coded_frame_is_longer_by_the_rate():
    pay = bytes(range(72))
    n0 = build_frame(pay, "qpsk").n_payload_symbols
    for c in fec.CODES[1:]:
        f = build_frame(pay, "qpsk", c)
        assert f.code == c and len(f.coded_bits) == fec.coded_length(576, c)
        assert f.n_payload_symbols == pytest.approx(n0 / fec.RATE[c], rel=0.02)
    assert np.array_equal(build_frame(pay, "qpsk").coded_bits, bytes_to_bits(pay))


def test_link_with_fec_survives_where_uncoded_fails():
    data = bytes(np.random.default_rng(8).integers(0, 256, 64 * 12, dtype=np.uint8))
    plain = send(data, "qpsk", -1.0, np.random.default_rng(9))
    coded = send(data, "qpsk", -1.0, np.random.default_rng(9), code="1/2")
    assert plain.packet_success_rate < 0.3 and coded.exact
    r = link_report(coded)
    assert r["ber"] > 1e-3 and r["ber_decoded"] == 0 and r["codes"] == ["1/2"]
    assert r["goodput_bps"] > link_report(plain)["goodput_bps"]
    rec = coded.packets[0]
    assert rec.code == "1/2" and rec.n_channel_bits == fec.coded_length(rec.n_bits, "1/2")


def test_uncoded_records_keep_their_meaning():
    pk = segment(bytes(range(64)), 64)
    rec = transfer_packet(pk[0], 1, "qam16", 6.0, np.random.default_rng(1)).record
    assert rec.code == "none" and rec.channel_bit_errors == rec.bit_errors and rec.n_channel_bits == rec.n_bits
    with pytest.raises(ValueError):
        transfer_packet(pk[0], 1, "qpsk", 6.0, np.random.default_rng(1), code="5/6")


def test_engine_fec_settings():
    from spectrum_intel.engine import EngineConfig, LinkEngine
    e = LinkEngine(EngineConfig(snr_db=0, mod="qpsk", fec="1/2", text="z" * 64 * 6)).run_to_end()
    s = e.state()
    json.dumps(s, allow_nan=False)
    assert all(h["code"] == "1/2" for h in s["history"])
    assert s["report"]["ber_decoded"] <= s["report"]["ber"]
    assert sum(o == "ok" for o in e.outcomes) >= 5
    e.update(fec="3/4", fec_decoding="hard", interleave=False)
    e.reset()
    e.step()
    assert e.records[-1].code == "3/4"
    with pytest.raises(ValueError):
        e.update(fec="5/6")
    with pytest.raises(ValueError):
        e.update(fec_decoding="fuzzy")


def test_local_noise_flags_a_burst():
    rng = np.random.default_rng(10)
    y = modulate(bits(400, 11), "qpsk") + 0.05 * (rng.normal(size=200) + 1j * rng.normal(size=200))
    y[100:120] += 2.0 * (rng.normal(size=20) + 1j * rng.normal(size=20))        # an interference burst
    nl = fec.local_noise(y, "qpsk")
    assert nl[105:115].min() > 20 * np.median(nl[:80])
    assert np.median(nl[:80]) < 0.02


def test_soft_with_local_noise_survives_a_jammer_burst():
    from spectrum_intel.channel import ChannelConfig
    pk = segment(bytes(np.random.default_rng(12).integers(0, 256, 64 * 8, dtype=np.uint8)), 64)
    cfg = ChannelConfig(jammer="pulsed", duty=0.1, sir_db=-5, jam_period_us=5000)
    fails = {}
    for dec in fec.DECODINGS:
        rng = np.random.default_rng(13)
        recs = [transfer_packet(pk[i % 8], 8, "qpsk", 12.0, rng, channel=cfg, code="1/2", decoding=dec).record
                for i in range(40)]
        fails[dec] = sum(r.header_ok and not r.crc_ok for r in recs)
    assert fails["soft"] <= 3 and fails["hard"] <= 3          # both cope; the experiment compares them in detail
    assert fails["soft_fixed"] >= 10                         # one noise level for the whole frame is fooled
