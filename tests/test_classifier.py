import numpy as np
import pytest
import torch

from spectrum_intel import classifier as cl
from spectrum_intel.dataset import make_dataset
from spectrum_intel.signals import CLASSES


def test_model_output_shape():
    model = cl.SpectrumCNN()
    out = model(torch.zeros(4, 2, 1024))
    assert out.shape == (4, len(CLASSES))


def test_complex_to_input_shape_and_power():
    x = 5 * np.exp(1j * np.linspace(0, 20, 3 * 128)).reshape(3, 128)
    inp = cl.complex_to_input(x)
    assert inp.shape == (3, 2, 128) and inp.dtype == np.float32
    assert np.allclose((inp ** 2).sum(axis=1).mean(axis=-1), 1.0, atol=1e-5)


def test_training_learns_easy_task_and_saves(tmp_path):
    # noise vs interference at high SNR is easy: a tiny run must beat chance clearly
    X, y, _ = make_dataset(per_class=60, n_samples=256, snrs=(20,),
                           classes=("noise", "interference"), seed=0)
    model, hist = cl.train_model(X, y, epochs=4, batch_size=32, device=torch.device("cpu"), verbose=False)
    assert len(hist) == 4
    path = cl.save_model(model, tmp_path / "m.pt")
    again = cl.load_model(path)
    p1 = cl.predict_proba(model, X[:10])
    p2 = cl.predict_proba(again, X[:10])
    assert np.allclose(p1[:, :2], p2[:, :2], atol=1e-5)
    pred = cl.predict(model, X)
    assert np.mean(pred == y) > 0.8


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    snr = np.array([0, 10, 0, 10])
    assert cl.accuracy_by_snr(y, p, snr) == {0.0: 1.0, 10.0: 0.5}
    cm = cl.confusion_matrix(y, p, 2)
    assert cm[0].tolist() == [0.5, 0.5] and cm[1].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("cfo", [0.0, 0.04])
def test_dataset_cfo_option_runs(cfo):
    X, y, _ = make_dataset(per_class=2, n_samples=128, seed=3, cfo_max=cfo)
    assert X.shape == (12, 2, 128)
