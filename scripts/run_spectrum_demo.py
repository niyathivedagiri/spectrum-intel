"""The full cognitive-radio loop on an 8-channel band.

    python scripts/run_spectrum_demo.py

1. Eight licensed (primary) users switch on and off over 60 time slots.
2. The radio records the whole band, draws its spectrum and spectrogram,
   and splits it into the 8 channels.
3. Sensing: energy detector and CNN each build an occupancy map.
4. A secondary user picks a channel every slot with different policies.
Figures come from the first run; the reported rates are averaged over N_RUNS runs.
Saves results/spectrum_*.png, results/occupancy_*.png, results/channel_selection.png.
"""
import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _common import BASELINE_MODEL, RESULTS, update_metrics
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from spectrum_intel import classifier as cl
from spectrum_intel import occupancy as oc
from spectrum_intel import spectrum as sp
from spectrum_intel.detector import EnergyDetector, energy
from spectrum_intel.plots import BLUE, CLASS_COLOURS, GREEN, INK, MUTED, ORANGE, save
from spectrum_intel.signals import CLASSES

N_SLOTS, N_NATIVE, N_RUNS = 60, 1024, 10
USERS = [
    oc.PrimaryUser("qpsk", 8, p_on=0.15, p_off=0.10),
    oc.PrimaryUser("ofdm", 12, p_on=0.10, p_off=0.08),
    oc.PrimaryUser("qam16", 15, p_on=0.08, p_off=0.15),
    oc.PrimaryUser("bpsk", 2, p_on=0.06, p_off=0.20),
    oc.PrimaryUser("interference", 10, p_on=0.05, p_off=0.25),
    oc.PrimaryUser("ofdm", 4, p_on=0.12, p_off=0.12),
    oc.PrimaryUser("qpsk", -2, p_on=0.10, p_off=0.15),
    oc.PrimaryUser("qam16", 18, p_on=0.20, p_off=0.06),
]


POLICIES = ["fixed channel (no sensing)", "random channel (no sensing)", "energy detector + hop",
            "CNN + hop", "perfect sensing + hop"]


def simulate(seed, model):
    """One run: users -> scene -> channelize -> sense (energy + CNN) -> pick channels."""
    rng = np.random.default_rng(seed)
    active = oc.markov_activity(USERS, N_SLOTS, rng)
    labels, snr = oc.activity_to_scene_inputs(USERS, active)
    scene = sp.build_scene(labels, snr, N_NATIVE, rng)
    truth = scene.occupied
    win = sp.scene_windows(scene)                                    # (slots, channels, n)
    floor = sp.noise_floor(scene.iq)
    ed = EnergyDetector(floor / sp.N_CHANNELS, N_NATIVE, pfa=0.01)
    sensed_ed = ed.detect(win)
    pred = cl.predict(model, cl.complex_to_input(win.reshape(-1, N_NATIVE))).reshape(N_SLOTS, sp.N_CHANNELS)
    sensed_cnn = pred != sp.NOISE_LABEL
    power = energy(win)
    choices = dict(zip(POLICIES, [
        oc.policy_fixed(truth, 0),
        oc.policy_random(truth, np.random.default_rng(seed + 1000)),
        oc.policy_sense_and_hop(sensed_ed, power),
        oc.policy_sense_and_hop(sensed_cnn, power),
        oc.policy_sense_and_hop(truth, power),
    ]))
    return dict(scene=scene, labels=labels, truth=truth, floor=floor, pred=pred,
                sensed_ed=sensed_ed, sensed_cnn=sensed_cnn, choices=choices)


def main(seed=7):
    model = cl.load_model(BASELINE_MODEL)
    runs = [simulate(seed + k, model) for k in range(N_RUNS)]
    r0 = runs[0]
    scene, labels, truth, floor, pred = r0["scene"], r0["labels"], r0["truth"], r0["floor"], r0["pred"]
    centres = sp.channel_centres()

    # ---- 1. Spectrum and spectrogram ----------------------------------------
    f, p = sp.psd(scene.slot(0), nperseg=1024)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(f, 10 * np.log10(p), color=BLUE, lw=1)
    ax.axhline(10 * np.log10(floor), color=ORANGE, ls="--", lw=1.2, label="estimated noise floor")
    for k in range(sp.N_CHANNELS + 1):
        ax.axvline(-0.5 + k / sp.N_CHANNELS, color=MUTED, lw=0.6, ls=":")
    for c, x0 in enumerate(centres):
        ax.text(x0, ax.get_ylim()[1], f"ch{c}\n{CLASSES[labels[0, c]]}", ha="center", va="top", fontsize=8, color=INK)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("frequency (fraction of the band's sample rate)")
    ax.set_ylabel("power density (dB)")
    ax.set_title("Power spectrum of the 8-channel band, first time slot")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, RESULTS / "spectrum_psd.png")

    t, fs, S = sp.spectrogram_db(scene.iq[: 30 * scene.slot_len], nperseg=256, noverlap=0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.pcolormesh(t / scene.slot_len, fs, S, shading="auto", cmap="Blues",
                  vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.7))
    ax.grid(False)
    for k in range(1, sp.N_CHANNELS):
        ax.axhline(-0.5 + k / sp.N_CHANNELS, color="white", lw=0.6)
    ax.set_yticks(centres, [f"ch{c}" for c in range(sp.N_CHANNELS)])
    ax.set_xlabel("time slot")
    ax.set_title("Spectrogram: users switching on and off in their channels (first 30 slots)")
    save(fig, RESULTS / "spectrum_spectrogram.png")

    # ---- 2-3. Sensing and channel selection, averaged over runs ----------------
    def mean_of(dicts):
        return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

    results = {name: mean_of([oc.evaluate_policy(r["choices"][name], r["truth"]) for r in runs])
               for name in POLICIES}
    sensing = {"energy": mean_of([oc.occupancy_accuracy(r["sensed_ed"], r["truth"]) for r in runs]),
               "cnn": mean_of([oc.occupancy_accuracy(r["sensed_cnn"], r["truth"]) for r in runs])}
    type_acc = float(np.mean(np.concatenate([r["pred"][r["truth"]] == r["labels"][r["truth"]] for r in runs])))
    print(f"  sensing over {N_RUNS} runs: energy {sensing['energy']}, CNN {sensing['cnn']}")
    print(f"  CNN names the signal correctly in {100 * type_acc:.1f}% of busy channel-slots")
    for name, r in results.items():
        print(f"  {name:28s} collisions {100 * r['collision_rate']:5.1f}%   changes {r['channel_changes']:.1f}")
    choices = r0["choices"]

    # ---- 4. Occupancy maps ----------------------------------------------------
    cmap = ListedColormap(CLASS_COLOURS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, title in [(axes[0], labels, "Ground truth"),
                            (axes[1], pred, f"Sensed by the CNN ({100 * type_acc:.0f}% of signals typed correctly)")]:
        ax.imshow(data.T, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5,
                  origin="lower", interpolation="nearest")
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel("time slot")
    cnn_choice = choices["CNN + hop"]
    axes[1].plot(np.arange(N_SLOTS), cnn_choice, color=INK, lw=1.5, drawstyle="steps-mid")
    axes[1].plot([], [], color=INK, lw=1.5, label="secondary user's channel")
    axes[0].set_ylabel("channel")
    axes[0].set_yticks(range(sp.N_CHANNELS))
    handles = [Patch(color=CLASS_COLOURS[i], label=c) for i, c in enumerate(CLASSES)]
    fig.legend(handles=handles + axes[1].get_legend_handles_labels()[0], loc="lower center",
               ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.2)
    save(fig, RESULTS / "occupancy_map.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(results)
    vals = [100 * results[n]["collision_rate"] for n in names]
    colours = [MUTED, MUTED, ORANGE, BLUE, GREEN]
    ax.barh(names[::-1], vals[::-1], color=colours[::-1], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("slots where the secondary user collided with a licensed user (%)")
    ax.set_title(f"Sensing before transmitting cuts collisions (mean of {N_RUNS} runs x {N_SLOTS} slots)")
    ax.set_xlim(0, max(vals) + 10)
    save(fig, RESULTS / "channel_selection.png")

    update_metrics("spectrum_demo", {
        "runs": N_RUNS, "slots_per_run": N_SLOTS, "channels": sp.N_CHANNELS,
        "true_occupancy": float(np.mean([r["truth"].mean() for r in runs])),
        "noise_floor_estimate_per_channel": float(np.mean([r["floor"] for r in runs]) / sp.N_CHANNELS),
        "sensing_energy": sensing["energy"], "sensing_cnn": sensing["cnn"],
        "cnn_type_accuracy_on_occupied": type_acc, "policies": results,
    })


if __name__ == "__main__":
    main()
