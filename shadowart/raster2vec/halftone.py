"""Continuous opacity field -> binary coverage (what actually gets cut).

Monochrome silhouettes threshold cleanly; for grey targets, Floyd-Steinberg error
diffusion turns continuous opacity into a black/white dot pattern whose *local mean*
matches the opacity. Either way the result is a boolean mask per panel.
"""
from __future__ import annotations

import numpy as np


def threshold(opacity, level=0.5):
    return np.asarray(opacity) >= level


def error_diffusion(opacity):
    """Floyd-Steinberg dithering. opacity [H,W] in [0,1] -> bool mask."""
    a = np.asarray(opacity, dtype=np.float64).copy()
    Hn, Wn = a.shape
    out = np.zeros_like(a, dtype=bool)
    for y in range(Hn):
        for x in range(Wn):
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0.5
            err = old - new
            if x + 1 < Wn:
                a[y, x + 1] += err * 7 / 16
            if y + 1 < Hn:
                if x > 0:
                    a[y + 1, x - 1] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if x + 1 < Wn:
                    a[y + 1, x + 1] += err * 1 / 16
    return out
