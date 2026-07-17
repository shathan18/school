"""Fabrication constraints on the binary mask: minimum feature size and kerf.

- enforce_min_feature: morphological open (kill islands < min feature) then close
  (kill gaps/holes < min feature), using a disk the size of the smallest cuttable
  detail. This is where the depth->resolution limit is made physical.
- kerf_offset: grow/shrink a polygon by half the cut width so the finished part is
  nominal size (uses pyclipper for robust offsetting).
"""
from __future__ import annotations

import numpy as np
import pyclipper
from scipy import ndimage

_SCALE = 1e6          # metres -> integer units for pyclipper


def _disk(radius_px):
    r = int(max(1, round(radius_px)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def enforce_min_feature(mask, min_feature_m, pixel_size_m):
    """Remove solid islands and holes smaller than min_feature."""
    radius_px = min_feature_m / max(pixel_size_m, 1e-9) / 2.0
    if radius_px < 0.5:
        return np.asarray(mask, bool)
    st = _disk(radius_px)
    m = ndimage.binary_opening(np.asarray(mask, bool), structure=st)
    m = ndimage.binary_closing(m, structure=st)
    return m


def kerf_offset(polygon_xy, delta_m):
    """Offset one polygon ring (list of (x,y) metres) by delta_m. Returns list of rings.

    delta_m > 0 grows the part (compensating a laser that eats `delta_m` off each edge).
    """
    pco = pyclipper.PyclipperOffset(2.0, 0.25)
    pts = [(int(round(x * _SCALE)), int(round(y * _SCALE))) for x, y in polygon_xy]
    pco.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    sol = pco.Execute(delta_m * _SCALE)
    return [[(x / _SCALE, y / _SCALE) for x, y in ring] for ring in sol]
