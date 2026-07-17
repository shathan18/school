"""Per-colour cut files: group shards by the perspex colour they must be cut from.

Each shard's colour is read from the panel colour-id map. For each colour we lay that
colour's shards out in per-panel columns (so nothing overlaps and the operator can see
which panel each piece belongs to) and write one SVG + DXF -> "cut these from <colour>".
Structural clear-perspex carriers (panel outline + weave slots) are exported separately by
the normal DXF/SVG path with empty pieces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import svgwrite
import ezdxf
from shapely import affinity

from ..targets import color as _color


def _piece_color(panel, poly, colorid_gi, names):
    """Palette name for a shard = majority non-clear colour-id under it."""
    Hp, Wp = colorid_gi.shape
    (u0, u1), (v0, v1) = panel.u_range, panel.v_range
    minx, miny, maxx, maxy = poly.bounds
    c0 = int((minx - u0) / (u1 - u0) * Wp); c1 = int((maxx - u0) / (u1 - u0) * Wp) + 1
    r0 = int((miny - v0) / (v1 - v0) * Hp); r1 = int((maxy - v0) / (v1 - v0) * Hp) + 1
    patch = colorid_gi[max(0, r0):max(1, r1), max(0, c0):max(1, c1)]
    nz = patch[patch > 0]
    if nz.size == 0:
        cx, cy = poly.centroid.x, poly.centroid.y
        cc = min(Wp - 1, max(0, int((cx - u0) / (u1 - u0) * Wp)))
        rr = min(Hp - 1, max(0, int((cy - v0) / (v1 - v0) * Hp)))
        idx = int(colorid_gi[rr, cc])
    else:
        idx = int(np.bincount(nz).argmax())
    return names[idx] if idx > 0 else "K"


def classify(scene, pieces_by_panel, colorid, names):
    """{panel_name: [(poly, colour_name)]}."""
    out = {}
    for gi, panel in enumerate(scene.panels):
        out[panel.name] = [(p, _piece_color(panel, p, colorid[gi], names))
                           for p in pieces_by_panel.get(panel.name, [])]
    return out


def group_by_color(scene, classified):
    """{colour_name: [placed shapely polygons]} laid out in per-panel columns."""
    gap = 0.04
    col_w = max((p.u_range[1] - p.u_range[0]) for p in scene.panels) + gap
    order = {p.name: i for i, p in enumerate(scene.panels)}
    groups = {}
    for panel in scene.panels:
        dx = order[panel.name] * col_w - panel.u_range[0]
        dy = -panel.v_range[0]
        for poly, name in classified[panel.name]:
            groups.setdefault(name, []).append(affinity.translate(poly, xoff=dx, yoff=dy))
    return groups


def export_all_color(scene, pieces_by_panel, colorid, names, out_dir, formats=("svg", "dxf")):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_by_color(scene, classify(scene, pieces_by_panel, colorid, names))
    written = []
    for name, polys in sorted(groups.items()):
        if "svg" in formats:
            written.append(_svg(name, polys, out_dir / f"{name}.svg"))
        if "dxf" in formats:
            written.append(_dxf(name, polys, out_dir / f"{name}.dxf"))
    return written, {n: len(p) for n, p in groups.items()}


def _bounds(polys):
    xs = [b for p in polys for b in (p.bounds[0], p.bounds[2])]
    ys = [b for p in polys for b in (p.bounds[1], p.bounds[3])]
    return min(xs), min(ys), max(xs), max(ys)


def _svg(name, polys, path):
    minx, miny, maxx, maxy = _bounds(polys)
    w_mm = (maxx - minx) * 1000; h_mm = (maxy - miny) * 1000
    dwg = svgwrite.Drawing(str(path), size=(f"{w_mm}mm", f"{h_mm}mm"),
                           viewBox=f"0 0 {w_mm} {h_mm}")
    fill = _color.hex_color(name)
    g = dwg.g(id=name, stroke="black", fill=fill, fill_opacity=0.85, stroke_width=0.2)
    for poly in polys:
        pts = [((x - minx) * 1000, (maxy - y) * 1000) for x, y in poly.exterior.coords]
        g.add(dwg.polygon(points=pts))
        for r in poly.interiors:
            g.add(dwg.polygon(points=[((x - minx) * 1000, (maxy - y) * 1000) for x, y in r.coords],
                              fill="white"))
    dwg.add(g); dwg.save()
    return str(path)


def _dxf(name, polys, path):
    doc = ezdxf.new(); doc.units = ezdxf.units.MM
    doc.layers.add(name)
    msp = doc.modelspace()
    minx, miny = _bounds(polys)[:2]
    for poly in polys:
        msp.add_lwpolyline([((x - minx) * 1000, (y - miny) * 1000) for x, y in poly.exterior.coords],
                           close=True, dxfattribs={"layer": name})
        for r in poly.interiors:
            msp.add_lwpolyline([((x - minx) * 1000, (y - miny) * 1000) for x, y in r.coords],
                               close=True, dxfattribs={"layer": name})
    doc.saveas(path)
    return str(path)
