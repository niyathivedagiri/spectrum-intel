"""Deep-learning signal classifier (1-D CNN on raw I/Q).

Input:  (batch, 2, n_samples) float32 — row 0 = I, row 1 = Q, unit power
Output: scores for the 6 classes in signals.CLASSES

Architecture: four convolution blocks (conv -> batch-norm -> ReLU -> max-pool)
that learn local waveform patterns, global average pooling (so the decision does
not depend on where in the window a pattern appears), and a small dense head.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spectrum_intel.signals import CLASSES


def get_device() -> torch.device:
    """Apple Silicon GPU (MPS) if present, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SpectrumCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 32):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2),
                                 nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(
            block(2, width, 7),
            block(width, 2 * width, 5),
            block(2 * width, 4 * width, 5),
            block(4 * width, 4 * width, 3),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * width, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=-1, keepdim=True)            # global average pooling over time
        return self.head(h)


def complex_to_input(x: np.ndarray) -> np.ndarray:
    """Complex windows (..., n) -> unit-power float32 (..., 2, n) network input (AGC + split I/Q)."""
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
    x = x / np.sqrt(np.where(p == 0, 1, p))
    return np.stack([x.real, x.imag], axis=-2).astype(np.float32)


def train_val_split(n: int, val_frac: float, seed: int):
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    return order[n_val:], order[:n_val]


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 12, batch_size: int = 128,
                lr: float = 2e-3, val_frac: float = 0.15, seed: int = 0,
                device: torch.device | None = None, verbose: bool = True):
    """Train a SpectrumCNN. Returns (model, history). Keeps the epoch with best validation accuracy."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = SpectrumCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(
        len(y) * (1 - val_frac) / batch_size)))
    loss_fn = nn.CrossEntropyLoss()

    tr, va = train_val_split(len(y), val_frac, seed)
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    gen = torch.Generator().manual_seed(seed)
    history, best_acc, best_state = [], -1.0, None
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        val_acc = float(np.mean(predict(model, X[va], device) == y[va]))
        history.append({"epoch": ep + 1, "loss": total / len(yt), "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {ep + 1:2d}/{epochs}  loss {total / len(yt):.3f}  "
                  f"val acc {val_acc:.3f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device | None = None,
                  batch_size: int = 512) -> np.ndarray:
    """Class probabilities, shape (n, n_classes)."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, len(CLASSES)))


def predict(model: nn.Module, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    return np.argmax(predict_proba(model, X, device), axis=1)


def save_model(model: nn.Module, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": list(CLASSES), "meta": meta or {}}, path)
    return path


def load_model(path: str | Path, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SpectrumCNN(n_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def accuracy_by_snr(y_true, y_pred, snr) -> dict[float, float]:
    return {float(s): float(np.mean(y_pred[snr == s] == y_true[snr == s])) for s in np.unique(snr)}


def confusion_matrix(y_true, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Rows = true class, columns = predicted class, values = row fractions."""
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (y_true, y_pred), 1)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
