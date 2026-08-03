"""Score how badly a target image's subject matte is damaged.

The 30x30 build reproduced white voids in the back view faithfully enough that they looked
like an optimiser artefact; they were in the source PNG all along. This measures the thing
the eye was noticing, so target sets can be compared without squinting at them:

  holes   -- white pixels ENCLOSED by the subject (a matte that ate into the figure).
             `binary_fill_holes` minus the mask itself.
  ragged  -- perimeter divided by the perimeter of a circle of equal area. A clean cutout
             sits near 1-2; a matte with torn, noisy edges climbs fast. Catches the damage
             `holes` cannot see: bites that open to the image border and so are not enclosed.
  pieces  -- connected components >=0.5% of the subject area. Should be 1. More means the
             cutout shattered, and `crop_mode='largest'` will silently discard the extras.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from shadowart.targets.color import subject_mask


def damage(path, white_thr=0.90):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    sub = ndimage.binary_opening(subject_mask(arr, white_thr), iterations=2)
    if not sub.any():
        return None
    filled = ndimage.binary_fill_holes(sub)
    area = float(sub.sum())
    per = float((ndimage.convolve(sub.astype(np.uint8), np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                                  mode="constant") > 0).sum())
    lbl, n = ndimage.label(sub)
    counts = np.bincount(lbl.ravel())[1:] if n else np.array([])
    return {
        "holes": (filled.sum() - area) / area,
        "ragged": per / (2.0 * np.sqrt(np.pi * area)),
        "pieces": int((counts >= 0.005 * area).sum()),
        "size": Image.open(path).size,
    }


def main(argv):
    paths = argv or sorted(glob.glob("examples/girl3_*.png") + glob.glob("examples/pearl*_*.png"))
    print(f"{'image':38s} {'size':>11s} {'holes%':>7s} {'ragged':>7s} {'pieces':>6s}")
    for p in paths:
        d = damage(p)
        if d is None:
            print(f"{p:38s}  -- empty subject mask --")
            continue
        flag = "  <-- damaged" if (d["holes"] > 0.01 or d["ragged"] > 3.0 or d["pieces"] > 1) else ""
        print(f"{p:38s} {str(d['size']):>11s} {d['holes']*100:7.2f} "
              f"{d['ragged']:7.2f} {d['pieces']:6d}{flag}")


if __name__ == "__main__":
    main(sys.argv[1:])
