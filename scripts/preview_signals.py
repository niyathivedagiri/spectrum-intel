"""Draw the six signal classes so you can see what the generator produces.

Run from the project folder:
    python scripts/preview_signals.py
Figures are saved to outputs/ (not committed; recreate any time).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # lets the script import spectrum_intel

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from spectrum_intel import signals  # noqa: E402
from spectrum_intel.dsp import add_awgn  # noqa: E402

OUT = ROOT / "outputs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#222222", "#8a8a85"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def gallery(snr_db=15, n=2048, seed=3):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(signals.CLASSES), 3, figsize=(11, 13))
    for row, label in enumerate(signals.CLASSES):
        x = signals.generate(label, n, snr_db, rng)
        a_t, a_f, a_iq = axes[row]
        a_t.plot(x.real[:200], color=BLUE, lw=1, label="I")
        a_t.plot(x.imag[:200], color=ORANGE, lw=1, label="Q")
        a_t.set_ylabel(label, fontsize=11, color=INK, rotation=0, ha="right", va="center")
        f, p = welch(x, fs=1.0, nperseg=256, return_onesided=False)
        order = np.argsort(f)
        a_f.plot(f[order], 10 * np.log10(p[order]), color=BLUE, lw=1.2)
        a_f.set_xlim(-0.5, 0.5)
        a_iq.plot(x.real, x.imag, ".", ms=1.5, color=BLUE, alpha=0.4)
        a_iq.set_aspect("equal")
        lim = 1.1 * np.max(np.abs(x))
        a_iq.set_xlim(-lim, lim)
        a_iq.set_ylim(-lim, lim)
        if row == 0:
            a_t.set_title("I and Q over time (first 200 samples)")
            a_f.set_title("Power spectrum (dB)")
            a_iq.set_title("Raw I/Q samples")
            a_t.legend(loc="upper right", frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("sample")
    axes[-1, 1].set_xlabel("frequency (fraction of fs)")
    axes[-1, 2].set_xlabel("I")
    fig.suptitle(f"The six signal classes at SNR = {snr_db} dB", fontsize=12, color=INK)
    return fig


def matched_filter_constellations(snrs=(10, 0, -10), sps=8, beta=0.35, span=8, seed=4):
    """Transmit -> noise -> matched filter -> sample once per symbol."""
    rng = np.random.default_rng(seed)
    h = signals.rrc_taps(beta, sps, span)
    delay = len(h) - 1
    fig, axes = plt.subplots(2, len(snrs), figsize=(10, 6.6))
    for r, mod in enumerate(["qpsk", "qam16"]):
        tx = signals.pulse_shape(signals.random_symbols(mod, 600, rng), sps, beta, span)
        for c, snr in enumerate(snrs):
            rx = np.convolve(add_awgn(tx, snr, rng), h)
            sym = rx[delay:delay + 600 * sps:sps][20:-20]
            sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2))
            ax = axes[r, c]
            ax.plot(sym.real, sym.imag, ".", ms=3, color=BLUE, alpha=0.5)
            ideal = signals.constellation(mod)
            ax.plot(ideal.real, ideal.imag, "+", ms=10, mew=2, color=ORANGE)
            ax.set_aspect("equal")
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            es_n0 = snr + 10 * np.log10(sps)
            ax.set_title(f"{mod.upper()}  SNR {snr} dB  (Es/N0 ≈ {es_n0:.0f} dB)", fontsize=9)
    fig.suptitle("After the matched filter: received symbols (blue) vs ideal points (orange +)",
                 fontsize=11, color=INK)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    for name, fig in [("signal_gallery.png", gallery()),
                      ("constellations_vs_snr.png", matched_filter_constellations())]:
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=120)
        plt.close(fig)
        print(f"saved outputs/{name}")


if __name__ == "__main__":
    main()
