"""Group output by material layer.

v1 is monochrome: a single layer named 'mono'. This is the seam where CMYK (Phase 4)
plugs in: separate into ['C','M','Y','K1','K2'], solve/vectorise each independently
(each at its own depth), and return one drawing set per layer. Keeping the grouping here
means the exporters and CLI don't change when color arrives.
"""
from __future__ import annotations

from typing import Dict, List

from .joints import PanelDrawing

MONO = "mono"
CMYK_LAYERS = ["C", "M", "Y", "K1", "K2"]     # for Phase 4


def group_mono(drawings: List[PanelDrawing]) -> Dict[str, List[PanelDrawing]]:
    return {MONO: drawings}
