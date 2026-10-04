import numpy as np
import pytest

from spectrum_intel.detector import EnergyDetector, energy, theoretical_pd, threshold
from spectrum_intel.dsp import awgn


def test_energy_of_unit_noise_is_about_one():
    x = awgn(100_000, 1.0, np.random.default_rng(0))
    assert energy(x) == pytest.approx(1.0, rel=0.02)


def test_threshold_above_noise_power():
    assert threshold(1.0, 1024, 0.01) > 1.0
    assert threshold(1.0, 1024, 0.01) > threshold(1.0, 1024, 0.1)


@pytest.mark.parametrize("pfa", [0.01, 0.1])
def test_false_alarm_rate_matches_design(pfa):
    rng = np.random.default_rng(1)
    noise = np.sqrt(0.5) * (rng.standard_normal((20_000, 256)) + 1j * rng.standard_normal((20_000, 256)))
    rate = EnergyDetector(1.0, 256, pfa).detect(noise).mean()
    assert rate == pytest.approx(pfa, abs=0.3 * pfa + 0.002)


def test_strong_signal_always_detected():
    rng = np.random.default_rng(2)
    x = awgn(1024, 1.0, rng) + 2 * np.exp(1j * 0.3 * np.arange(1024))
    assert EnergyDetector(1.0, 1024).detect(x)


def test_theoretical_pd_rises_with_snr():
    pd = theoretical_pd([-20, -10, 0], 1024, 0.01)
    assert pd[0] < pd[1] < pd[2] and pd[2] > 0.99
