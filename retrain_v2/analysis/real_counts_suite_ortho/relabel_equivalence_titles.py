#!/usr/bin/env python3
"""
relabel_equivalence_titles.py — replaces the panel titles on the finished
equivalence figures with titles a reader can parse.

The titles as drawn are pipeline keys: "RA -> HO (injury) (t1.0)",
"Acute26 (t1.0)", "BPDPH7mo (t1.0)". Nothing else on those figures is wrong, and
recomputing them means re-running the model over the test cells to redraw one
line of text, so this edits the raster instead: it finds the title band, paints
it out, and writes the new titles centred exactly where the old ones sat.

HOW THE POSITIONS ARE FOUND, rather than hard-coded. The title band is the run
of image rows above the plot frame that contain ink. Inside that band, columns
holding ink cluster into one group per title, separated by wide empty gaps; the
centre of each cluster is where that title was centred, so the replacement lands
in the same place whatever the figure's width or panel count. The band's ink
height sets the font size, so the new titles come out at the size the old ones
were.

    python relabel_equivalence_titles.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

DST = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/figures_for_manuscript")

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
    """Rows of the topmost ink block -- the titles, above the plot frame."""
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
