#!/usr/bin/env python3
"""
Replaces the panel titles of the equivalence figures (S1a, S1b) with
descriptive titles.

The title band is the topmost run of image rows containing ink; within it,
column clusters separated by wide gaps give one span per title. Each old title
is painted over and the new one is centred on the same span, at a font size
derived from the band height.

    python relabel_equivalence_titles.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

DST = Path(__file__).resolve().parents[2] / "figures_for_manuscript"

JOBS = [
    (DST / "S1a_equivalence_rat.png", [
        "Rat, room air versus hyperoxia",
        "Rat, hyperoxia versus hyperoxia + azithromycin",
    ]),
    (DST / "S1b_equivalence_human.png", [
        "Human, gestational week 24 versus acute injury, week 26",
        "Human, term-born versus BPD at 7 months",
        "Human, term-born versus BPD-PH at 7 months",
    ]),
]

SIZE_FACTOR = {"S1b_equivalence_human.png": 0.86}
INK = 0.6
GAP_FRAC = 0.02


def title_band(dark: np.ndarray) -> tuple[int, int]:
    """Rows of the topmost ink block (the titles)."""
    rows = np.where(dark.any(axis=1))[0]
    start = int(rows[0])
    end = start
    for r in rows[1:]:
        if r - end > 3:
            break
        end = int(r)
    return start, end


def title_spans(dark_band: np.ndarray, width: int) -> list[tuple[int, int]]:
    """One (left, right) column span per title in the band."""
    cols = np.where(dark_band.any(axis=0))[0]
    gap = max(int(GAP_FRAC * width), 8)
    spans, lo, prev = [], int(cols[0]), int(cols[0])
    for c in cols[1:]:
        if c - prev > gap:
            spans.append((lo, prev))
            lo = int(c)
        prev = int(c)
    spans.append((lo, prev))
    return spans


def relabel(path: Path, titles: list[str]) -> None:
    im = mpimg.imread(path)
    h, w = im.shape[:2]
    dark = im[..., :3].mean(-1) < INK

    top, bot = title_band(dark)
    spans = title_spans(dark[top:bot + 1], w)
    if len(spans) != len(titles):
        raise SystemExit(f"{path.name}: found {len(spans)} titles, given {len(titles)}")

    rgb = im[..., :3].copy()
    pad = 4
    rgb[max(top - pad, 0):bot + pad + 1, :, :] = 1.0

    dpi = 200.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(rgb, interpolation="none")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    pt = (bot - top + 1) / dpi * 72.0 * 1.05 * SIZE_FACTOR.get(path.name, 1.0)
    y = (top + bot) / 2.0
    for (lo, hi), text in zip(spans, titles):
        ax.text((lo + hi) / 2.0, y, text, ha="center", va="center",
                fontsize=pt, fontweight="normal", color="#1a1a1a", family=["Arial", "Helvetica", "DejaVu Sans"])

    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{path.name}: band rows {top}-{bot}, {len(spans)} titles, {pt:.1f} pt")


def main() -> None:
    for path, titles in JOBS:
        relabel(path, titles)


if __name__ == "__main__":
    main()
