"""Diagnostic ONLY: draw the current `Cand.head` / `Cand.face` boxes on the full source
image next to the square crop they produce, so a bad crop is visible instead of guessed at.

Run:  python face_crop_check.py
Out:  out_faces_hc/crop/crop_check.png   (one row per candidate)
      out_faces_hc/crop/full_<key>.png   (full image + boxes, for close inspection)

Why this exists: the head/face boxes in face_pretest.CANDIDATES are hand-typed fractions.
Nothing in the pipeline ever validated them, so a box that clips the skull or the chin
silently propagates into every downstream render.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import face_pretest as FP

OUT = Path("out_faces_hc/crop")
KEYS = ["poe", "dostoevsky", "mallarme", "wagner", "ibsen", "mona", "pearl"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = {c.key: c for c in FP.CANDIDATES}
    rows = [cands[k] for k in KEYS if k in cands]

    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3.0 * len(rows)))
    for i, c in enumerate(rows):
        img = Image.open(c.path).convert("RGB")
        W, H = img.size
        l, t, r, b = c.head
        cl, ct, cr, cb = c.crop

        ax[i, 0].imshow(np.asarray(img))
        ax[i, 0].add_patch(mpatches.Rectangle(
            (l * W, t * H), (r - l) * W, (b - t) * H,
            fill=False, ec="red", lw=1.6, ls="--"))
        ax[i, 0].add_patch(mpatches.Rectangle(
            (cl * W, ct * H), (cr - cl) * W, (cb - ct) * H,
            fill=False, ec="yellow", lw=2))
        ax[i, 0].set_title(f"{c.key}: head(--) vs squared crop(-)", fontsize=8)
        ax[i, 0].axis("off")

        head = FP.load_head(c)
        ax[i, 1].imshow(head, cmap="gray", vmin=0, vmax=1)
        fl, ft, fr, fb = c.face
        n = head.shape[0]
        ax[i, 1].add_patch(mpatches.Rectangle(
            (fl * n, ft * n), (fr - fl) * n, (fb - ft) * n,
            fill=False, ec="lime", lw=2))
        ax[i, 1].set_title("head crop + face box", fontsize=8)
        ax[i, 1].axis("off")

        ax[i, 2].imshow(FP.posterize_gray(head, 2), cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_title("what the solver aims at (2-tone)", fontsize=8)
        ax[i, 2].axis("off")

        # standalone full image + labelled fractional grid, so a corrected box can be
        # READ OFF instead of guessed a second time.
        f2, a2 = plt.subplots(figsize=(7, 7 * H / W))
        a2.imshow(np.asarray(img))
        for f in np.arange(0.0, 1.001, 0.05):
            major = abs(f * 20 - round(f * 20)) < 1e-6 and round(f * 20) % 2 == 0
            a2.axvline(f * W, color="cyan", lw=1.0 if major else 0.4,
                       alpha=0.9 if major else 0.45)
            a2.axhline(f * H, color="cyan", lw=1.0 if major else 0.4,
                       alpha=0.9 if major else 0.45)
            if major:
                a2.text(f * W, -0.012 * H, f"{f:.1f}", color="blue", fontsize=7,
                        ha="center", va="bottom")
                a2.text(-0.012 * W, f * H, f"{f:.1f}", color="blue", fontsize=7,
                        ha="right", va="center")
        a2.add_patch(mpatches.Rectangle(
            (l * W, t * H), (r - l) * W, (b - t) * H,
            fill=False, ec="red", lw=1.8, ls="--"))
        a2.add_patch(mpatches.Rectangle(
            (cl * W, ct * H), (cr - cl) * W, (cb - ct) * H,
            fill=False, ec="yellow", lw=2.2))
        fl, ft, fr, fb = c.face_img
        a2.add_patch(mpatches.Rectangle(
            (fl * W, ft * H), (fr - fl) * W, (fb - ft) * H,
            fill=False, ec="lime", lw=1.8))
        a2.set_title(f"{c.key}  head(--)=({l:.2f},{t:.2f},{r:.2f},{b:.2f})  "
                     f"crop(-)=({cl:.2f},{ct:.2f},{cr:.2f},{cb:.2f})", fontsize=8)
        a2.set_xlim(min(0, cl * W) - 0.02 * W, max(W, cr * W) + 0.02 * W)
        a2.set_ylim(max(H, cb * H) + 0.02 * H, min(0, ct * H) - 0.02 * H)
        a2.axis("off")
        f2.tight_layout()
        f2.savefig(OUT / f"full_{c.key}.png", dpi=110)
        plt.close(f2)

        print(f"{c.key:12s} {W:5d}x{H:<5d} head={tuple(round(v,3) for v in c.head)}  "
              f"crop={tuple(round(v,3) for v in c.crop)}  "
              f"face={tuple(round(v,3) for v in c.face)}")

    fig.tight_layout()
    fig.savefig(OUT / "crop_check.png", dpi=95)
    plt.close(fig)
    print(f"\nwrote {OUT/'crop_check.png'} and {len(rows)} full_*.png")


if __name__ == "__main__":
    main()
