import numpy as np
import pytest

from spectrum_intel import classifier as cl
from spectrum_intel import openset as osr
from spectrum_intel.dataset import make_dataset
from spectrum_intel.signals import CLASSES

MODEL = "models/cnn_baseline.pt"


@pytest.fixture(scope="module")
def fitted():
    model = cl.load_model(MODEL)
    Xf, yf, _ = make_dataset(60, seed=0)
    Xv, _, _ = make_dataset(60, seed=12)
    det = osr.OpenSetDetector.fit(model, Xf, yf)
    det.calibrate(model, Xv)
    return model, det, Xv


@pytest.mark.parametrize("kind", osr.UNKNOWN_KINDS)
def test_unknown_signals_are_unit_power_and_new(kind):
    x = osr.unknown_clean(kind, 1024, np.random.default_rng(0))
    assert len(x) == 1024 and np.mean(np.abs(x) ** 2) == pytest.approx(1.0)
    assert kind not in CLASSES


def test_unknown_signal_properties():
    rng = np.random.default_rng(1)
    for kind in ("gfsk", "fm"):                                    # constant envelope
        assert np.std(np.abs(osr.unknown_clean(kind, 1024, rng))) < 1e-9
    am = osr.unknown_clean("am", 1024, rng)
    assert np.allclose(am.imag, 0) and am.real.min() > 0           # AM with carrier never crosses zero
    with pytest.raises(ValueError):
        osr.unknown_clean("ufo", 1024, rng)
    X, k, s = osr.make_unknown_set(3, [10], seed=2)
    assert X.shape == (18, 2, 1024) and np.bincount(k).tolist() == [3] * 6 and set(s) == {10}


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 30).astype(float)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert osr.auroc(a, b) == pytest.approx(brute)
    assert osr.auroc([2, 3], [0, 1]) == 1.0 and osr.auroc([0, 1], [2, 3]) == 0.0


def test_thresholds_keep_95_percent_of_known(fitted):
    model, det, Xv = fitted
    s = det.scores(model, Xv)
    for name in osr.SCORES:
        assert np.mean(s[name] >= det.thresholds[name]) == pytest.approx(osr.TARGET_TPR, abs=0.01)


def test_mahalanobis_rejects_far_unknowns_softmax_does_not(fitted):
    model, det, _ = fitted
    Xu, _, _ = osr.make_unknown_set(40, [10, 15], seed=4, kinds=("am", "fm"))
    s = det.scores(model, Xu)
    assert np.mean(s["mahalanobis"] >= det.thresholds["mahalanobis"]) < 0.1
    assert np.mean(s["msp"] >= det.thresholds["msp"]) > 0.8        # the CNN is confident about what it never saw
    Xk, yk, _ = make_dataset(20, snrs=(10, 15), seed=5)
    pred = det.predict(model, Xk)
    assert np.mean(pred != osr.UNKNOWN) > 0.85
    assert np.mean(pred[pred != osr.UNKNOWN] == yk[pred != osr.UNKNOWN]) > 0.95


def test_save_load_round_trip(fitted, tmp_path):
    model, det, Xv = fitted
    det2 = osr.OpenSetDetector.load(det.save(tmp_path / "d.npz"))
    a, b = det.scores(model, Xv[:50]), det2.scores(model, Xv[:50])
    for name in osr.SCORES:
        assert np.allclose(a[name], b[name], atol=1e-4)
    assert det2.thresholds == pytest.approx(det.thresholds)
