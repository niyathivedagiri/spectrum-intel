import numpy as np
import pytest

from spectrum_intel import dsp

FS = 8000
N = 8000


def test_db_round_trip():
    assert dsp.db(100) == pytest.approx(20)
    assert dsp.from_db(3) == pytest.approx(2, rel=0.01)


def test_tone_power_is_amplitude_squared():
    assert dsp.signal_power(dsp.tone(1000, FS, N, amplitude=2)) == pytest.approx(4)


@pytest.mark.parametrize("freq", [1000, -1500, 3000])
def test_fft_peak_at_tone_frequency(freq):
    f, mag = dsp.fft_spectrum(dsp.tone(freq, FS, N), FS)
    assert f[np.argmax(mag)] == pytest.approx(freq)


def test_frequency_shift_moves_peak():
    shifted = dsp.frequency_shift(dsp.tone(500, FS, N), 1200, FS)
    f, mag = dsp.fft_spectrum(shifted, FS)
    assert f[np.argmax(mag)] == pytest.approx(1700)


@pytest.mark.parametrize("snr_db", [-10, 0, 20])
def test_add_awgn_hits_target_snr(snr_db):
    rng = np.random.default_rng(0)
    clean = dsp.tone(1000, FS, 100_000)
    noisy = dsp.add_awgn(clean, snr_db, rng)
    measured = dsp.db(dsp.signal_power(clean) / dsp.signal_power(noisy - clean))
    assert measured == pytest.approx(snr_db, abs=0.2)


def test_tone_visible_below_zero_snr():
    rng = np.random.default_rng(1)
    noisy = dsp.add_awgn(dsp.tone(1000, FS, N), -10, rng)
    f, mag = dsp.fft_spectrum(noisy, FS)
    assert f[np.argmax(mag)] == pytest.approx(1000)
