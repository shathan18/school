"""Export panel drawings to SVG (one file per panel), coordinates in millimetres.

SVG y grows downward, so we flip v about the panel top. Groups mirror the DXF layers:
outline (blue stroke), slot (red stroke), piece (green stroke + light fill).
"""
from __future__ import annotations

from pathlib import Path

import svgwrite

from ..raster2vec.contours import polygon_rings


def _pts(ring, u0, v0, vtop):
    # u -> x (mm from panel left); v -> y flipped (mm from panel top)
    return [((u - u0) * 1000.0, (vtop - v) * 1000.0) for u, v in ring]


def export_panel_svg(drawing, path):
    u0, u1 = drawing.u_range
    v0, v1 = drawing.v_range
    w_mm = (u1 - u0) * 1000.0
    h_mm = (v1 - v0) * 1000.0
    dwg = svgwrite.Drawing(str(path), size=(f"{w_mm}mm", f"{h_mm}mm"),
                           viewBox=f"0 0 {w_mm} {h_mm}")
    g_out = dwg.g(id="outline", stroke="blue", fill="none", stroke_width=0.3)
    g_slot = dwg.g(id="slot", stroke="red", fill="none", stroke_width=0.3)
    g_piece = dwg.g(id="piece", stroke="green", fill="#66bb66", fill_opacity=0.25,
                    stroke_width=0.3)

    g_out.add(dwg.polygon(points=_pts(drawing.outline, u0, v0, v1)))
    for s in drawing.slots:
        g_slot.add(dwg.polygon(points=_pts(s, u0, v0, v1)))
    for poly in drawing.pieces:
        ext, holes = polygon_rings(poly)
        g_piece.add(dwg.polygon(points=_pts(ext, u0, v0, v1)))
        for h in holes:
            g_piece.add(dwg.polygon(points=_pts(h, u0, v0, v1), fill="white"))

    for g in (g_out, g_slot, g_piece):
        dwg.add(g)
    dwg.save()
    return path


def export_all_svg(drawings, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return [export_panel_svg(d, out_dir / f"panel_{d.name}.svg") for d in drawings]
