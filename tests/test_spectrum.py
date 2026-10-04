import numpy as np
import pytest

from spectrum_intel import spectrum as sp
from spectrum_intel.signals import CLASSES

QPSK, OFDM = CLASSES.index("qpsk"), CLASSES.index("ofdm")


def test_channel_centres_cover_band():
    c = sp.channel_centres(8)
    assert len(c) == 8
    assert c[0] == pytest.approx(-0.4375) and c[-1] == pytest.approx(0.4375)
    assert np.allclose(np.diff(c), 1 / 8)


def test_wideband_round_trip_recovers_signal():
    rng = np.random.default_rng(0)
    x = np.exp(1j * 2 * np.pi * 0.05 * np.arange(2048))
    back = sp.channelize(sp.to_wideband(x, 5), 5)[:2048]
    mid = slice(100, -100)                          # ignore filter edges
    assert np.allclose(back[mid], x[mid], atol=0.02)


def test_scene_shape_and_truth():
    labels = np.full((3, 8), sp.NOISE_LABEL)
    labels[:, 2] = QPSK
    scene = sp.build_scene(labels, np.full((3, 8), 10.0), 512, np.random.default_rng(1))
    assert scene.iq.shape == (3 * 512 * 8,)
    assert scene.occupied.sum() == 3
    assert np.isnan(scene.snr_db[0, 0]) and scene.snr_db[0, 2] == 10


@pytest.mark.parametrize("snr_db", [0.0, 10.0])
def test_channelized_snr_matches_target(snr_db):
    labels = np.full((6, 8), sp.NOISE_LABEL)
    labels[:, 3] = OFDM
    scene = sp.build_scene(labels, np.full((6, 8), snr_db), 1024, np.random.default_rng(2))
    p = np.mean(np.abs(sp.scene_windows(scene)) ** 2, axis=-1)
    noise = p[:, [0, 1, 5, 6, 7]].mean()            # channels far from the signal
    measured = 10 * np.log10(p[:, 3].mean() / noise - 1)
    assert measured == pytest.approx(snr_db, abs=0.6)


def test_empty_channels_stay_empty_next_to_strong_signal():
    labels = np.full((4, 8), sp.NOISE_LABEL)
    labels[:, 4] = QPSK
    scene = sp.build_scene(labels, np.full((4, 8), 25.0), 1024, np.random.default_rng(3))
    p = np.mean(np.abs(sp.scene_windows(scene)) ** 2, axis=-1).mean(0)
    assert p[3] < 1.1 * p[0] and p[5] < 1.1 * p[0]  # neighbours: no leakage above noise


def test_noise_floor_estimate():
    rng = np.random.default_rng(4)
    labels = np.full((4, 8), sp.NOISE_LABEL)
    labels[:, [1, 6]] = QPSK
    scene = sp.build_scene(labels, np.full((4, 8), 15.0), 1024, rng)
    assert sp.noise_floor(scene.iq) == pytest.approx(1.0, rel=0.1)   # wideband noise PSD = 1


def test_channelized_noise_is_white():
    """Empty channels must look like the classifier's training noise: flat spectrum."""
    rng = np.random.default_rng(5)
    labels = np.full((8, 8), sp.NOISE_LABEL)
    scene = sp.build_scene(labels, np.zeros((8, 8)), 1024, rng)
    w = sp.scene_windows(scene)[:, 2].ravel()
    f, p = sp.psd(w, nperseg=128)
    edge, centre = p[np.abs(f) > 0.4].mean(), p[np.abs(f) < 0.1].mean()
    assert 10 * np.log10(edge / centre) == pytest.approx(0, abs=0.5)
