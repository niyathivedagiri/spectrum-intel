"""LEO satellite extension: the same spectrum monitor flying in low Earth orbit.

A satellite listening to transmitters on the ground sees two big differences:

  1. Doppler shift — it moves at about 7.6 km/s, so every received frequency is
     shifted by  f_D = -f_c * (range rate) / c , up to ~ +/-48 kHz at 2 GHz.
  2. Distance — free-space path loss of 150-165 dB, changing through the pass,
     so the SNR rises and falls as the satellite flies over.

Geometry: a circular orbit whose ground track passes directly over the
transmitter (the worst case for Doppler rate); Earth's rotation is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS = 6371e3        # m
MU_EARTH = 3.986004418e14    # m^3/s^2
C = 299_792_458.0            # m/s


def orbital_speed(altitude_m: float) -> float:
    """Circular-orbit speed v = sqrt(mu / r)."""
    return float(np.sqrt(MU_EARTH / (EARTH_RADIUS + altitude_m)))


@dataclass
class Pass:
    time_s: np.ndarray
    elevation_deg: np.ndarray
    range_m: np.ndarray
    range_rate_mps: np.ndarray


def overhead_pass(altitude_m: float = 550e3, min_elevation_deg: float = 10.0, dt_s: float = 1.0) -> Pass:
    """Satellite pass straight over a ground point, from horizon (min elevation) to horizon."""
    r = EARTH_RADIUS + altitude_m
    omega = orbital_speed(altitude_m) / r                     # rad/s
    theta = np.linspace(-0.6, 0.6, 20001)                    # orbit angle, 0 = overhead
    sat = np.stack([r * np.sin(theta), r * np.cos(theta)])
    ground = np.array([[0.0], [EARTH_RADIUS]])
    d = sat - ground
    rng_m = np.linalg.norm(d, axis=0)
    elev = np.degrees(np.arcsin(d[1] / rng_m))               # local vertical is the y axis
    visible = elev >= min_elevation_deg
    t_all = theta / omega
    t = np.arange(t_all[visible].min(), t_all[visible].max(), dt_s)
    th = omega * t
    sat = np.stack([r * np.sin(th), r * np.cos(th)])
    d = sat - ground
    rng_m = np.linalg.norm(d, axis=0)
    vel = r * omega * np.stack([np.cos(th), -np.sin(th)])
    range_rate = np.sum(d * vel, axis=0) / rng_m
    elev = np.degrees(np.arcsin(d[1] / rng_m))
    return Pass(t - t[0], elev, rng_m, range_rate)


def doppler_hz(range_rate_mps, carrier_hz: float):
    """Doppler shift: approaching satellite (negative range rate) -> positive shift."""
    return -carrier_hz * np.asarray(range_rate_mps) / C


def fspl_db(distance_m, carrier_hz: float):
    """Free-space path loss 20*log10(4*pi*d*f/c)."""
    return 20 * np.log10(4 * np.pi * np.asarray(distance_m) * carrier_hz / C)


def link_snr_db(distance_m, carrier_hz: float = 2e9, eirp_dbm: float = 23.0,
                rx_gain_dbi: float = 30.0, bandwidth_hz: float = 1e6, noise_figure_db: float = 3.0):
    """SNR at the satellite for a ground transmitter (simple link budget).

    received = EIRP + receive antenna gain - path loss
    noise    = -174 dBm/Hz + 10 log10(bandwidth) + noise figure
    Defaults: a 23 dBm (200 mW) handset-class transmitter, a 30 dBi satellite antenna,
    1 MHz channel, 3 dB receiver noise figure.
    """
    rx = eirp_dbm + rx_gain_dbi - fspl_db(distance_m, carrier_hz)
    noise = -174 + 10 * np.log10(bandwidth_hz) + noise_figure_db
    return rx - noise
