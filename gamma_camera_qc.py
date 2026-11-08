"""Gamma-camera quality control from two phantom images.

A. Bar-phantom modulation (MTF points)
   The quadrant phantom has four bar patterns of 4, 5, 6 and 7 line pairs per mm. For each
   quadrant the script
   - finds the quadrant borders (the dark cross between the patterns),
   - detects the bar orientation and period with a Fourier transform of the profiles,
   - takes EVERY profile perpendicular to the bars (not one line) and computes
     M = (Pmax - Pmin) / (Pmax + Pmin) from the mean of its maxima and minima,
   - reports the mean and standard deviation of M over all profiles,
   - measures a noise floor with the same formula on profiles running ALONG the bars (no bar
     signal there), so a modulation close to it means the bars are not resolved.
   The task does not say which quadrant holds which frequency, so frequencies are assigned by the
   measured bar period: the finest bars are 7 lp/mm, the coarsest 4 lp/mm.

B. Field uniformity
   A(p) = mean of the flat-field image over a 13 x 13 window around each pixel.
   Range non-uniformity = (Amax - Amin) / Amax, plus the mean relative deviation |A - mean| / mean.
   Only pixels whose whole 13 x 13 window lies inside the image are used: near the border the
   filter would otherwise average mirrored pixels and bias Amax / Amin.

Usage:  python gamma_camera_qc.py      (writes results/ and figures/)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from skimage import color, io

ROOT = Path(__file__).resolve().parent
PHANTOM = ROOT / "phantom_C9FF1H.gif"
FLOOD = ROOT / "homogeneity_image_C9FF1A.gif"
FREQUENCIES = [4, 5, 6, 7]  # lp/mm, from the task description
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def load_gray(path: Path) -> np.ndarray:
    """Image as float intensities 0-255, whatever the file's channel layout."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3]) * 255.0
    return img.astype(float)  # a 2-D uint8 image is already on the 0-255 scale


# ----------------------------------------------------------------------------- A. modulation
def quadrant_borders(img: np.ndarray) -> tuple[int, int]:
    h, w = img.shape
    col_mean, row_mean = img.mean(axis=0), img.mean(axis=1)
    x = int(np.argmin(col_mean[w // 3: 2 * w // 3]) + w // 3)
    y = int(np.argmin(row_mean[h // 3: 2 * h // 3]) + h // 3)
    return x, y


def dominant_period(lines: np.ndarray) -> tuple[float, float]:
    """Bar period (px) and its spectral power, from profiles stacked as rows."""
    n = lines.shape[1]
    spectrum = (np.abs(np.fft.rfft((lines - lines.mean(axis=1, keepdims=True)) * np.hanning(n), axis=1)) ** 2).mean(0)
    freqs = np.fft.rfftfreq(n)
    band = (freqs > 1 / 12) & (freqs < 1 / 3)  # bar periods between 3 and 12 px
    i = int(np.argmax(np.where(band, spectrum, 0)))
    # parabolic interpolation around the peak for a sub-bin estimate
    if 0 < i < len(spectrum) - 1:
        a, b, c = np.log(spectrum[i - 1: i + 2] + 1e-12)
        shift = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        shift = 0.0
    return 1.0 / ((i + shift) / n), float(spectrum[i])


def profile_modulation(profile: np.ndarray, period: float):
    distance = max(1, int(0.6 * period))  # at most one maximum (minimum) per bar pair
    peaks, _ = find_peaks(profile, distance=distance)
    troughs, _ = find_peaks(-profile, distance=distance)
    if len(peaks) < 2 or len(troughs) < 2:
        return None
    pmax, pmin = profile[peaks].mean(), profile[troughs].mean()
    return (pmax - pmin) / (pmax + pmin), pmax, pmin, peaks, troughs


def analyse_phantom(img: np.ndarray, margin: int = 12) -> tuple[list[dict], tuple[int, int]]:
    x0, y0 = quadrant_borders(img)
    regions = {
        "top-left": (slice(0, y0 - margin), slice(0, x0 - margin)),
        "top-right": (slice(0, y0 - margin), slice(x0 + margin, None)),
        "bottom-left": (slice(y0 + margin, None), slice(0, x0 - margin)),
        "bottom-right": (slice(y0 + margin, None), slice(x0 + margin, None)),
    }
    out = []
    for name, (ys, xs) in regions.items():
        roi = img[ys, xs]
        horiz_period, horiz_power = dominant_period(roi)      # profiles along rows
        vert_period, vert_power = dominant_period(roi.T)      # profiles along columns
        if horiz_power >= vert_power:
            lines, along_bars, period, direction = roi, roi.T, horiz_period, "horizontal"
        else:
            lines, along_bars, period, direction = roi.T, roi, vert_period, "vertical"
        trim = 4  # avoid the blurred edge rows next to the quadrant border
        values = [r for r in (profile_modulation(l, period) for l in lines[trim:-trim]) if r is not None]
        m = np.array([v[0] for v in values])
        # Noise floor: the same measure on profiles running ALONG the bars, where there is no bar signal.
        floor = [r[0] for r in (profile_modulation(l, period) for l in along_bars[trim:-trim]) if r is not None]
        out.append({"quadrant": name, "profile_direction": direction, "bar_period_px": period,
                    "modulation_mean": float(m.mean()), "modulation_std": float(m.std()),
                    "noise_floor_mean": float(np.mean(floor)), "noise_floor_std": float(np.std(floor)),
                    "n_profiles": int(len(m)), "pmax_mean": float(np.mean([v[1] for v in values])),
                    "pmin_mean": float(np.mean([v[2] for v in values])),
                    "_example": lines[len(lines) // 2], "_slices": (ys, xs)})
    # finest bars = highest frequency
    for q, f in zip(sorted(out, key=lambda q: -q["bar_period_px"]), FREQUENCIES):
        q["lp_per_mm"] = f
        q["implied_pixel_mm"] = 1.0 / (f * q["bar_period_px"])
    return sorted(out, key=lambda q: q["lp_per_mm"]), (x0, y0)


def plot_phantom(img, quads, border):
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, cmap="gray")
    for q in quads:
        ys, xs = q["_slices"]
        y = (ys.start + (ys.stop or img.shape[0])) / 2
        x = (xs.start + (xs.stop or img.shape[1])) / 2
        ax.text(x, y, f"{q['lp_per_mm']} lp/mm\nM = {q['modulation_mean']:.3f}", color="white", ha="center",
                va="center", fontsize=9, bbox=dict(facecolor="black", alpha=0.55, lw=0))
    ax.set_title("Quadrant phantom\n(frequency assigned by bar period)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1:])
    f = [q["lp_per_mm"] for q in quads]
    floor = np.array([q["noise_floor_mean"] for q in quads])
    floor_sd = np.array([q["noise_floor_std"] for q in quads])
    ax.fill_between(f, floor - 2 * floor_sd, floor + 2 * floor_sd, color=MUTED, alpha=0.18, lw=0,
                    label="noise floor (profiles along the bars, ±2 SD)")
    ax.errorbar(f, [q["modulation_mean"] for q in quads], yerr=[q["modulation_std"] for q in quads],
                color=BLUE, marker="o", ms=7, lw=2, capsize=4, label="bar modulation (across the bars)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for q in quads:
        ax.annotate(f"{q['modulation_mean']:.3f}", (q["lp_per_mm"], q["modulation_mean"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=MUTED)
    ax.set(xlabel="Spatial frequency (lp/mm)", ylabel="Modulation M", xticks=f,
           title="Modulation falls to the noise level at 6-7 lp/mm")
    ax.grid(alpha=0.3)

    for k, q in enumerate([quads[0], quads[-1]]):
        ax = fig.add_subplot(gs[1, 1 + k])
        p = q["_example"]
        r = profile_modulation(p, q["bar_period_px"])
        ax.plot(p, color=MUTED, lw=1)
        if r:
            ax.plot(r[3], p[r[3]], "o", color=ORANGE, ms=4, label="Pmax")
            ax.plot(r[4], p[r[4]], "o", color=BLUE, ms=4, label="Pmin")
        ax.set(title=f"Example profile, {q['lp_per_mm']} lp/mm ({q['quadrant']})", xlabel="pixel", ylabel="intensity")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "bar_phantom_modulation.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- B. uniformity
def analyse_flood(img: np.ndarray, size: int = 13) -> dict:
    A = uniform_filter(img, size=size, mode="reflect")
    h = size // 2
    valid = np.zeros_like(A, dtype=bool)
    valid[h:-h, h:-h] = True  # the whole window lies inside the image

    def stats(mask):
        a = np.where(mask, A, np.nan)
        imax, imin = np.nanargmax(a), np.nanargmin(a)
        amax, amin = float(a.flat[imax]), float(a.flat[imin])
        mean = float(img[mask].mean()) if mask.all() else float(img.mean())
        return {"Amax": amax, "Amin": amin,
                "Amax_rc": [int(v) for v in np.unravel_index(imax, A.shape)],
                "Amin_rc": [int(v) for v in np.unravel_index(imin, A.shape)],
                "range_nonuniformity": (amax - amin) / amax,
                "mean_relative_deviation": float(np.nanmean(np.abs(a - img.mean()) / img.mean()))}

    return {"window": size, "valid_pixels_only": stats(valid), "including_border_pixels": stats(np.ones_like(valid))}


def plot_flood(img, res):
    v = res["valid_pixels_only"]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    im = ax.imshow(img, cmap="gray")
    fig.colorbar(im, ax=ax, label="Counts (grey level)")
    ax.plot(v["Amax_rc"][1], v["Amax_rc"][0], "x", color=ORANGE, ms=14, mew=2.5, label=f"Amax = {v['Amax']:.1f}")
    ax.plot(v["Amin_rc"][1], v["Amin_rc"][0], "+", color=BLUE, ms=16, mew=2.5, label=f"Amin = {v['Amin']:.1f}")
    ax.set_title(f"Flood image: range non-uniformity (Amax - Amin) / Amax = {v['range_nonuniformity']:.3f}")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "flood_uniformity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    phantom, flood = load_gray(PHANTOM), load_gray(FLOOD)

    quads, border = analyse_phantom(phantom)
    plot_phantom(phantom, quads, border)
    uni = analyse_flood(flood)
    plot_flood(flood, uni)

    clean = [{k: v for k, v in q.items() if not k.startswith("_")} for q in quads]
    (ROOT / "results" / "results.json").write_text(json.dumps({"bar_phantom": clean, "quadrant_border_xy": border,
                                                               "uniformity": uni}, indent=2), encoding="utf-8")
    print(f"Quadrant borders at x = {border[0]}, y = {border[1]}")
    print(f"{'lp/mm':>5} {'quadrant':>13} {'profile':>10} {'period px':>9} {'M mean':>7} {'M std':>6} {'floor':>6} {'n':>4} {'px mm':>6}")
    for q in clean:
        print(f"{q['lp_per_mm']:>5} {q['quadrant']:>13} {q['profile_direction']:>10} {q['bar_period_px']:>9.2f} "
              f"{q['modulation_mean']:>7.3f} {q['modulation_std']:>6.3f} {q['noise_floor_mean']:>6.3f} "
              f"{q['n_profiles']:>4} {q['implied_pixel_mm']:>6.4f}")
    for key in ("valid_pixels_only", "including_border_pixels"):
        u = uni[key]
        print(f"uniformity ({key}): Amax {u['Amax']:.1f} at {u['Amax_rc']}, Amin {u['Amin']:.1f} at {u['Amin_rc']}, "
              f"range NU {u['range_nonuniformity']:.3f}, mean rel. dev. {u['mean_relative_deviation']:.3f}")


if __name__ == "__main__":
    main()
