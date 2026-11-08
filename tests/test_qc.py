"""Synthetic checks: known bar period and modulation, and border handling in the uniformity test."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gamma_camera_qc as qc  # noqa: E402


def bars(period, modulation, n=240, mean=150.0):
    x = np.arange(n)
    return mean * (1 + modulation * np.cos(2 * np.pi * x / period))


def test_period_and_modulation_of_a_clean_bar_pattern():
    lines = np.tile(bars(6.0, 0.2), (50, 1))
    period, _ = qc.dominant_period(lines)
    assert abs(period - 6.0) < 0.2
    m = qc.profile_modulation(lines[0], period)[0]
    assert abs(m - 0.2) < 0.02


def test_phantom_quadrants_get_frequencies_by_period():
    rng = np.random.default_rng(0)
    img = np.full((200, 260), 150.0)
    img[:, 127:133] = 60  # dark cross
    img[97:103, :] = 60
    q = {(0, 0): (5.2, "h"), (0, 1): (4.6, "v"), (1, 0): (6.5, "v"), (1, 1): (4.1, "h")}
    for (r, c), (p, d) in q.items():
        ys = slice(0, 97) if r == 0 else slice(103, 200)
        xs = slice(0, 127) if c == 0 else slice(133, 260)
        h, w = img[ys, xs].shape
        pat = bars(p, 0.15, w) if d == "h" else bars(p, 0.15, h)[:, None]
        img[ys, xs] = np.broadcast_to(pat, (h, w)) + rng.normal(0, 1, (h, w))
    quads, _ = qc.analyse_phantom(img, margin=6)
    got = {x["quadrant"]: x["lp_per_mm"] for x in quads}
    assert got == {"bottom-left": 4, "top-left": 5, "top-right": 6, "bottom-right": 7}
    assert all(x["modulation_mean"] > 3 * x["noise_floor_mean"] for x in quads)


def test_uniformity_ignores_border_artefacts():
    img = np.full((60, 80), 100.0)
    img[0, 0] = 5000.0  # single hot corner pixel: only reaches windows that are cut by the border
    res = qc.analyse_flood(img, size=13)
    assert res["including_border_pixels"]["Amax"] > res["valid_pixels_only"]["Amax"]
