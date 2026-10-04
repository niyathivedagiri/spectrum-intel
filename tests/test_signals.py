import numpy as np
import pytest

from spectrum_intel import dsp, signals
from spectrum_intel.signals import CLASSES, MODULATIONS

N = 4096


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---- constellations --------------------------------------------------------
@pytest.mark.parametrize("mod,points", [("bpsk", 2), ("qpsk", 4), ("qam16", 16)])
def test_constellation_size_and_unit_power(mod, points):
    c = signals.constellation(mod)
    assert len(np.unique(np.round(c, 6))) == points
    assert dsp.signal_power(c) == pytest.approx(1.0)


def test_bpsk_is_real():
    assert np.allclose(signals.constellation("bpsk").imag, 0)


def test_qpsk_points_at_45_degree_diagonals():
    angles = np.sort(np.degrees(np.angle(signals.constellation("qpsk"))) % 360)
    assert np.allclose(angles, [45, 135, 225, 315])


def test_qam16_gray_coding_neighbours_differ_by_one_bit():
    levels = {v: k for k, v in signals._QAM16_LEVELS.items()}
    order = [-3, -1, 1, 3]
    for a, b in zip(order, order[1:]):
        assert sum(x != y for x, y in zip(levels[a], levels[b])) == 1


def test_bits_to_symbols_rejects_wrong_length():
    with pytest.raises(ValueError):
        signals.bits_to_symbols([0, 1, 1], "qpsk")


# ---- pulse shaping ---------------------------------------------------------
def test_rrc_taps_symmetric_unit_energy():
    h = signals.rrc_taps(0.35, 8, 8)
    assert np.allclose(h, h[::-1])
    assert np.sum(h**2) == pytest.approx(1.0)


def test_rrc_pair_gives_zero_intersymbol_interference():
    sps = 8
    h = signals.rrc_taps(0.35, sps, 16)
    rc = np.convolve(h, h)                      # transmit RRC + receive RRC = raised cosine
    centre = len(rc) // 2
    others = rc[centre % sps::sps]
    others = np.delete(others, centre // sps)
    assert np.max(np.abs(others)) < 0.02 * rc[centre]


@pytest.mark.parametrize("mod", MODULATIONS)
def test_single_carrier_length_power_and_bandwidth(mod, rng):
    sps, beta = 8, 0.35
    x = signals.single_carrier(mod, N, rng, sps=sps, beta=beta)
    assert len(x) == N
    assert dsp.signal_power(x) == pytest.approx(1.0)
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(N))
    inside = np.abs(f) <= (1 + beta) / (2 * sps) * 1.1
    assert X[inside].sum() / X.sum() > 0.99


# ---- OFDM ------------------------------------------------------------------
def test_ofdm_cyclic_prefix_copies_end_of_symbol(rng):
    sym = signals.ofdm_symbol(signals.random_symbols("qpsk", 20, rng), n_fft=128, cp_len=32)
    assert len(sym) == 160
    assert np.allclose(sym[:32], sym[-32:])


def test_ofdm_subcarriers_recovered_by_fft(rng):
    data = signals.random_symbols("qpsk", 20, rng)
    sym = signals.ofdm_symbol(data, n_fft=128, cp_len=32)
    X = np.fft.fft(sym[32:])                    # drop CP, FFT back
    assert np.allclose(X[signals.active_subcarriers(128, 20)], data)


def test_ofdm_power_and_bandwidth(rng):
    x = signals.ofdm(N, rng)
    assert dsp.signal_power(x) == pytest.approx(1.0)
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(N))
    assert X[np.abs(f) <= 12 / 128].sum() / X.sum() > 0.95


# ---- interference ----------------------------------------------------------
def test_tone_is_one_spectral_line(rng):
    x = signals.interference(N, rng, kind="tone")
    X = np.abs(np.fft.fft(x)) ** 2
    assert np.sort(X)[-3:].sum() / X.sum() > 0.9


def test_chirp_spreads_across_band(rng):
    x = signals.interference(N, rng, kind="chirp")
    X = np.abs(np.fft.fft(x)) ** 2
    assert np.sort(X)[-3:].sum() / X.sum() < 0.2


# ---- generate() ------------------------------------------------------------
@pytest.mark.parametrize("label", CLASSES)
def test_generate_every_class(label, rng):
    x = signals.generate(label, 1024, snr_db=10, rng=rng)
    assert x.shape == (1024,) and np.iscomplexobj(x)
    assert np.all(np.isfinite(x))


def test_generate_is_reproducible_with_seed():
    a = signals.generate("qam16", 512, 5, np.random.default_rng(7))
    b = signals.generate("qam16", 512, 5, np.random.default_rng(7))
    assert np.allclose(a, b)


def test_generate_unknown_class_raises(rng):
    with pytest.raises(ValueError):
        signals.generate("fm", 128, 10, rng)


def test_normalize_power():
    x = 3 * np.exp(1j * np.linspace(0, 10, 500))
    assert dsp.signal_power(dsp.normalize_power(x)) == pytest.approx(1.0)
