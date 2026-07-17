"""Binary mask -> shapely polygons (with holes) in panel metres.

Uses contourpy marching squares at level 0.5 on a zero-padded mask (so every contour
is closed), then combines the rings with an even-odd XOR so nested rings become holes.
Coordinates are mapped from pixels to panel-local metres.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from contourpy import contour_generator, LineType
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def _rings_from_mask(mask):
    m = np.pad(np.asarray(mask, dtype=np.float64), 1)     # zero border -> closed loops
    cg = contour_generator(z=m, line_type=LineType.Separate)
    lines = cg.lines(0.5)
    rings = []
    for ln in lines:
        pts = np.asarray(ln, dtype=float)
        if pts.shape[0] >= 4:
            rings.append(pts - 1.0)                        # undo the pad offset
    return rings


def mask_to_polygons(mask, u_range, v_range, simplify_px=0.6):
    """Return a list of shapely Polygons (metres) with holes, for one panel mask.

    mask indexed [row(v), col(u)]; row/col are y/x for contourpy (x=col, y=row).
    """
    Hn, Wn = mask.shape
    rings = _rings_from_mask(mask)
    if not rings:
        return []
    su = (u_range[1] - u_range[0]) / Wn                    # metres per pixel (u = x = col)
    sv = (v_range[1] - v_range[0]) / Hn                    # metres per pixel (v = y = row)

    polys = []
    for r in rings:
        p = Polygon(r)
        if not p.is_valid:
            p = p.buffer(0)
        if p.area > 0:
            if simplify_px:
                p = p.simplify(simplify_px, preserve_topology=True)
            if p.area > 0:
                polys.append(p)
    if not polys:
        return []

    geom = reduce(lambda a, b: a.symmetric_difference(b), polys)   # even-odd -> holes
    geom = geom.buffer(0)
    # pixel (x=col, y=row) -> panel metres: u = u0 + col*su ; v = v0 + row*sv
    geom = affinity.affine_transform(geom, [su, 0, 0, sv, u_range[0], v_range[0]])
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.area > 0]
    # GeometryCollection fallback
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and g.area > 0]


def polygon_rings(poly):
    """(exterior_coords, [hole_coords,...]) as lists of (x,y)."""
    ext = list(poly.exterior.coords)
    holes = [list(r.coords) for r in poly.interiors]
    return ext, holes
