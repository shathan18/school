"""Simple shelf nesting of panel carriers onto stock sheets.

Greedy left-to-right shelf packing: place panels along a row until the sheet width is
exceeded, start a new shelf; start a new sheet when the height is exceeded. Panels larger
than the sheet are flagged (they must be tiled or a bigger sheet used). Returns a
placement (sheet index + offset) per panel — used only for the combined overview export;
per-panel files are exported at the origin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Placement:
    name: str
    sheet: int
    dx: float
    dy: float
    oversize: bool


def nest(drawings, sheet_size, margin=0.01):
    sw, sh = sheet_size
    placements: List[Placement] = []
    sheet = 0
    cx = margin
    cy = margin
    shelf_h = 0.0
    for d in drawings:
        w = d.u_range[1] - d.u_range[0]
        h = d.v_range[1] - d.v_range[0]
        oversize = (w > sw - 2 * margin) or (h > sh - 2 * margin)
        if cx + w > sw - margin:                 # wrap to next shelf
            cx = margin
            cy += shelf_h + margin
            shelf_h = 0.0
        if cy + h > sh - margin:                 # next sheet
            sheet += 1
            cx = margin; cy = margin; shelf_h = 0.0
        placements.append(Placement(d.name, sheet, cx - d.u_range[0], cy - d.v_range[0], oversize))
        cx += w + margin
        shelf_h = max(shelf_h, h)
    return placements
