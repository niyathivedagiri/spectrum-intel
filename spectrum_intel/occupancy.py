"""Spectrum occupancy and channel selection (the "cognitive" part of the radio).

Primary users (the licensed owners of each channel) switch on and off over time.
A secondary user (our cognitive radio) senses every channel each time slot,
builds an occupancy map, and picks a channel to transmit on in the NEXT slot.
A collision = transmitting on a channel the primary user is using.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectrum_intel.signals import CLASSES
from spectrum_intel.spectrum import NOISE_LABEL


@dataclass
class PrimaryUser:
    """One channel's licensed user: its signal type, strength and on/off behaviour."""
    label: str            # one of CLASSES (not "noise")
    snr_db: float
    p_on: float           # chance per slot of switching on when off
    p_off: float          # chance per slot of switching off when on


def markov_activity(users: list[PrimaryUser], n_slots: int, rng: np.random.Generator) -> np.ndarray:
    """On/off pattern of each user over time (two-state Markov chain). Shape (n_slots, n_users)."""
    active = np.zeros((n_slots, len(users)), dtype=bool)
    for c, u in enumerate(users):
        on = rng.random() < u.p_on / (u.p_on + u.p_off)      # start in the long-run proportion
        for t in range(n_slots):
            active[t, c] = on
            on = (rng.random() >= u.p_off) if on else (rng.random() < u.p_on)
    return active


def activity_to_scene_inputs(users: list[PrimaryUser], active: np.ndarray):
    """Turn on/off activity into (labels, snr_db) arrays for spectrum.build_scene."""
    labels = np.full(active.shape, NOISE_LABEL)
    snr = np.zeros(active.shape)
    for c, u in enumerate(users):
        labels[active[:, c], c] = CLASSES.index(u.label)
        snr[:, c] = u.snr_db
    return labels, snr


# --------------------------------------------------------------------------
# Channel selection policies. Each takes the SENSED occupancy (n_slots, n_ch)
# and returns the channel chosen for every slot (decided from earlier slots only).
# --------------------------------------------------------------------------
def policy_fixed(sensed: np.ndarray, channel: int = 0) -> np.ndarray:
    """No sensing: always transmit on the same channel."""
    return np.full(sensed.shape[0], channel)


def policy_random(sensed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No sensing: a random channel every slot."""
    return rng.integers(0, sensed.shape[1], sensed.shape[0])


def policy_sense_and_hop(sensed: np.ndarray, power: np.ndarray | None = None) -> np.ndarray:
    """Cognitive policy.

    Slot t uses what was sensed in slot t-1:
      * stay on the current channel while it is still free (avoids needless hops);
      * otherwise move to the free channel that has been idle the longest
        (a channel that has been quiet for a while is likely to stay quiet);
      * ties broken by the lowest measured power (least interference).
    """
    n_slots, n_ch = sensed.shape
    power = np.zeros_like(sensed, dtype=float) if power is None else power
    idle = np.zeros(n_ch)
    choice = np.zeros(n_slots, dtype=int)
    current = 0
    for t in range(n_slots):
        if t > 0:
            free = ~sensed[t - 1]
            idle = np.where(free, idle + 1, 0)
            if not free[current]:
                if free.any():
                    score = np.where(free, idle - 1e-3 * power[t - 1], -np.inf)
                    current = int(np.argmax(score))
                # if nothing is free, stay put (and probably collide)
        choice[t] = current
    return choice


def evaluate_policy(choice: np.ndarray, truth_occupied: np.ndarray) -> dict:
    """Collision rate, success rate and number of channel changes."""
    slots = np.arange(len(choice))
    collided = truth_occupied[slots, choice]
    return {
        "collision_rate": float(np.mean(collided[1:])),     # slot 0 has no sensing history
        "success_rate": float(np.mean(~collided[1:])),
        "channel_changes": int(np.sum(np.diff(choice) != 0)),
    }


def occupancy_accuracy(sensed: np.ndarray, truth: np.ndarray) -> dict:
    """How well sensing matched the truth."""
    busy, free = truth, ~truth
    return {
        "accuracy": float(np.mean(sensed == truth)),
        "detection_rate": float(np.mean(sensed[busy])) if busy.any() else float("nan"),
        "false_alarm_rate": float(np.mean(sensed[free])) if free.any() else float("nan"),
    }
