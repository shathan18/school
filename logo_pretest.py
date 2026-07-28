"""Logo contact sheet: what each mark looks like, and its INK AREA in both polarities.

Ink area is the number that decides everything here. From the face campaign:
    ink 44% -> 2.1% of the wall crossed by the other image -> 2/14 planes serve both
    ink 69% -> 9.0%
    ink 85% -> 19.1% -> 8/10 planes serve both
White paper = clear perspex = NO SHARD, and bare wall cannot intersect the other wall.
A normal logo is dark-mark-on-white, i.e. ink ~10-25% -- BELOW the case we already
proved cannot intersect. Inverting it (white mark on dark ground) pushes ink to ~75-90%
at ZERO cost in contrast, which is exactly what the GRAY_L paper tint bought for faces
but without giving up the black/white extremes.

This script only measures and displays. Run it before committing to a pair.

Run:  python logo_pretest.py
Out:  out_logos/pretest.png, out_logos/pretest.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import face_pretest as FP

SRC = Path("examples/logos")
OUT = Path("out_logos")
WORK = 256
WHITE_THR = 0.90          # scene.white_threshold: above this the solver cuts NOTHING


@dataclass
class Logo:
    key: str
    who: str
    trim_pad: float = 0.04     # white margin to keep after trimming to the mark


LOGOS = [
    Logo("technion", "Technion (official mark)"),
    Logo("technion5", "Technion (alt mark)"),
    Logo("huji", "Hebrew University"),
    Logo("tau", "Tel Aviv University"),
    Logo("bgu", "Ben-Gurion University"),
]


def load_square(key: str, pad: float, size: int = WORK) -> np.ndarray:
    """Trim to the mark's bounding box, pad to a SQUARE by edge-replication, greyscale.

    Same padding discipline as `face_pretest.Cand.crop`: never clamp/crop to force a
    square, or the mark loses limbs. Trimming first matters because these files carry
    a lot of dead white margin, which would otherwise dominate the ink-area figure.
    """
    rgb = np.asarray(Image.open(SRC / f"{key}.png").convert("RGB"), np.float32) / 255.0
    g = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    mark = g < 0.90
    if mark.any():
        ys, xs = np.where(mark)
        py = int(pad * g.shape[0])
        px = int(pad * g.shape[1])
        y0, y1 = max(0, ys.min() - py), min(g.shape[0], ys.max() + 1 + py)
        x0, x1 = max(0, xs.min() - px), min(g.shape[1], xs.max() + 1 + px)
        g = g[y0:y1, x0:x1]

    h, w = g.shape
    side = max(h, w)
    g = np.pad(g, (((side - h) // 2, side - h - (side - h) // 2),
                   ((side - w) // 2, side - w - (side - w) // 2)), mode="edge")
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    out = np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255.0
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def ink_area(g: np.ndarray) -> float:
    """Fraction of the wall the solver will actually put material on."""
    return float((g < WHITE_THR).mean() * 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, recs = [], []
    for lg in LOGOS:
        g = load_square(lg.key, lg.trim_pad)
        two = FP.posterize_gray(g, 2)
        inv = 1.0 - two
        rec = dict(key=lg.key, who=lg.who,
                   ink_normal=ink_area(two), ink_inverted=ink_area(inv))
        recs.append(rec)
        rows.append((lg, g, two, inv, rec))

    fig, ax = plt.subplots(len(rows), 3, figsize=(8.2, 2.75 * len(rows)))
    for i, (lg, g, two, inv, rec) in enumerate(rows):
        for j, (img, ttl) in enumerate((
                (g, "greyscale"),
                (two, f"NORMAL 2-tone\nink {rec['ink_normal']:.1f}%"),
                (inv, f"INVERTED 2-tone\nink {rec['ink_inverted']:.1f}%"))):
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(ttl, fontsize=8)
        ax[i, 0].set_ylabel(lg.who, fontsize=8)
    fig.suptitle("Ink area decides whether the piece can intersect at all.\n"
                 "Reference: ink 44% -> 2/14 planes served both images; ink 85% -> 8/10.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pretest.png", dpi=110, bbox_inches="tight")
    (OUT / "pretest.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    print(f"{'logo':12s}{'ink NORMAL':>12s}{'ink INVERTED':>14s}   verdict")
    print("-" * 62)
    for r in recs:
        v = ("inverted lands in the proven-intersecting band"
             if r["ink_inverted"] >= 70 else "inverted still thin -- check")
        print(f"{r['key']:12s}{r['ink_normal']:>11.1f}%{r['ink_inverted']:>13.1f}%   {v}")
    print(f"\nwrote {OUT}/pretest.png")


if __name__ == "__main__":
    main()
