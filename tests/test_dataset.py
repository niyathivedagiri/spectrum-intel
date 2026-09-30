import numpy as np

from spectrum_intel import dataset
from spectrum_intel.signals import CLASSES


def test_shapes_dtypes_and_balance():
    X, y, snr = dataset.make_dataset(per_class=10, n_samples=256, seed=1)
    assert X.shape == (60, 2, 256) and X.dtype == np.float32
    assert np.bincount(y, minlength=len(CLASSES)).tolist() == [10] * len(CLASSES)
    assert snr.min() >= -10 and snr.max() <= 20


def test_examples_are_unit_power():
    X, _, _ = dataset.make_dataset(per_class=5, n_samples=256, seed=2)
    power = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1)
    assert np.allclose(power, 1.0, atol=1e-4)


def test_reproducible_with_seed():
    a = dataset.make_dataset(per_class=3, n_samples=128, seed=5)
    b = dataset.make_dataset(per_class=3, n_samples=128, seed=5)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_save_and_load_round_trip(tmp_path):
    X, y, snr = dataset.make_dataset(per_class=2, n_samples=64, seed=3)
    path = dataset.save_dataset(tmp_path / "d.npz", X, y, snr)
    X2, y2, snr2, classes = dataset.load_dataset(path)
    assert np.array_equal(X, X2) and np.array_equal(y, y2) and np.array_equal(snr, snr2)
    assert classes == CLASSES
