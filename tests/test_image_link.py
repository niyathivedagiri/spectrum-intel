import numpy as np
import pytest

from spectrum_intel import link
from spectrum_intel import metrics as m
from spectrum_intel import payload as pl
from spectrum_intel.packet import Packet, parse_unchecked


@pytest.fixture
def img():
    return pl.load_image(size=16)


def test_default_image_loads(img):
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8


@pytest.mark.parametrize("shape", [(8, 8), (8, 12, 3)])
def test_image_serialisation_round_trip(shape):
    a = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    b, ok = pl.bytes_to_image(pl.image_to_bytes(a))
    assert ok and np.array_equal(a, b)


def test_damaged_image_header_uses_fallback_shape():
    a = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    data = bytearray(pl.image_to_bytes(a))
    data[0] ^= 0xFF
    b, ok = pl.bytes_to_image(bytes(data), shape=(4, 4, 3))
    assert not ok and np.array_equal(a, b)


def test_png_round_trip_and_damage(img):
    png = pl.png_bytes(img)
    assert np.array_equal(pl.decode_png(png), img)
    broken = png[:len(png) // 2] + bytes(len(png) - len(png) // 2)
    out = pl.decode_png(broken)
    assert out is None or not np.array_equal(out, img)


def test_image_metrics():
    a = np.full((16, 16), 100, np.uint8)
    assert m.psnr(a, a) == float("inf") and m.ssim(a, a) == pytest.approx(1.0)
    b = a.copy()
    b[0, 0] = 110
    assert m.psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2 / (100 / 256)))
    assert m.pixel_error_rate(a, b) == pytest.approx(1 / 256)
    noisy = np.clip(a + np.random.default_rng(1).normal(0, 30, a.shape), 0, 255).astype(np.uint8)
    assert m.ssim(a, noisy) < 0.9


def test_parse_unchecked_rejects_implausible_headers():
    good = Packet(2, 5, b"x" * 10).to_bytes()
    assert parse_unchecked(good, 5, 64).seq == 2
    assert parse_unchecked(good, 6, 64) is None            # wrong total
    bad = bytearray(good)
    bad[0:2] = (9).to_bytes(2, "big")                       # seq beyond total
    assert parse_unchecked(bytes(bad), 5, 64) is None


def test_image_exact_at_good_snr(img):
    r = link.send(pl.image_to_bytes(img), "qpsk", 8.0, np.random.default_rng(2), fill=128)
    out, ok = pl.bytes_to_image(r.received, img.shape)
    assert ok and np.array_equal(out, img) and r.missing == []


def test_tolerant_mode_beats_strict_on_same_channel(img):
    data = pl.image_to_bytes(img)
    s = link.send(data, "qpsk", 1.0, np.random.default_rng(3), fill=128)
    t = link.send(data, "qpsk", 1.0, np.random.default_rng(3), keep_corrupted=True, fill=128)
    assert [p.bit_errors for p in s.packets] == [p.bit_errors for p in t.packets]   # identical channel
    a, _ = pl.bytes_to_image(s.received, img.shape)
    b, _ = pl.bytes_to_image(t.received, img.shape)
    assert len(t.missing) <= len(s.missing) and m.psnr(img, b) >= m.psnr(img, a)
    assert set(t.corrupted).isdisjoint({p.seq for p in s.packets if p.crc_ok})       # good packets never replaced
