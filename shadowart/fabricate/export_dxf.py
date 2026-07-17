"""Export panel drawings to DXF (one file per panel), coordinates in millimetres.

Layers:  OUTLINE = carrier rectangle to cut,  SLOT = cross-lap slots to cut,
         PIECE   = opaque shadow shapes (cut from opaque/frost material, or engrave).
"""
from __future__ import annotations

from pathlib import Path

import ezdxf

from ..raster2vec.contours import polygon_rings

_LAYERS = {"OUTLINE": 5, "SLOT": 1, "PIECE": 3}    # aci colors: blue, red, green


def _mm(ring, u0, v0):
    return [((u - u0) * 1000.0, (v - v0) * 1000.0) for u, v in ring]


def export_panel_dxf(drawing, path):
    doc = ezdxf.new()
    doc.units = ezdxf.units.MM
    for name, color in _LAYERS.items():
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    u0, v0 = drawing.u_range[0], drawing.v_range[0]

    msp.add_lwpolyline(_mm(drawing.outline, u0, v0), close=True, dxfattribs={"layer": "OUTLINE"})
    for s in drawing.slots:
        msp.add_lwpolyline(_mm(s, u0, v0), close=True, dxfattribs={"layer": "SLOT"})
    for poly in drawing.pieces:
        ext, holes = polygon_rings(poly)
        msp.add_lwpolyline(_mm(ext, u0, v0), close=True, dxfattribs={"layer": "PIECE"})
        for h in holes:
            msp.add_lwpolyline(_mm(h, u0, v0), close=True, dxfattribs={"layer": "PIECE"})
    doc.saveas(path)
    return path


def export_all_dxf(drawings, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return [export_panel_dxf(d, out_dir / f"panel_{d.name}.dxf") for d in drawings]
