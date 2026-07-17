"""Point-spread function (penumbra / focus blur) helpers.

A real "point" light has a finite emitting area, so shadows are not perfectly sharp:
each panel's contribution is blurred by a gaussian whose width grows with distance from
the wall (worst near the light). That is the true resolution limit of the piece — a
piece near the light can only render coarse blobs. We model it as a separable gaussian
applied to each panel's warped contribution *before* compositing.
"""
from __future__ import annotations

import numpy as np


def gaussian_kernel1d(sigma_px, truncate=3.0):
    """1D gaussian kernel (normalised) for a given sigma in pixels."""
    sigma_px = float(max(sigma_px, 1e-6))
    radius = max(1, int(truncate * sigma_px + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma_px ** 2))
    return (k / k.sum()).astype(np.float32)


def sigma_m_to_px(sigma_m, wall_extent_m, wall_res_px):
    """Convert a blur sigma in wall metres to wall pixels."""
    return sigma_m * wall_res_px / max(wall_extent_m, 1e-9)
