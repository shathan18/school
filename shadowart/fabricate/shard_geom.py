"""Shared 3D shard geometry: extrude shard polygons into prisms with per-vertex colour.

Used by both the interactive preview (Plotly Mesh3d) and the PLY export so they stay in
sync. `color_of(panel, poly) -> (r,g,b) in [0,1]` supplies each shard's colour.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point


def triangulate(poly):
    """(verts_uv [N,2], faces [M,3]) for a shapely polygon, respecting holes/concavity."""
    pts = list(poly.exterior.coords)[:-1]
    for r in poly.interiors:
        pts += list(r.coords)[:-1]
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    try:
        tri = Delaunay(pts)
    except Exception:
        return None
    faces = [s for s in tri.simplices if poly.contains(Point(pts[s].mean(axis=0)))]
    if not faces:
        return None
    return pts, np.asarray(faces, int)


def _extrude(panel, uv, faces, poly, thickness, center_offset=0.0):
    """Return (verts [V,3], tris [T,3] local indices) for one extruded shard.

    `center_offset` shifts the whole prism along the panel normal (metres) without changing
    its (u,v) footprint -- used to laminate several channel layers at the same spot but a
    different micro depth, so overlapping shards stack instead of z-fighting."""
    n = panel.normal
    N = len(uv)
    front = panel.uv_to_xyz(uv) + n * (center_offset - thickness / 2)
    back = panel.uv_to_xyz(uv) + n * (center_offset + thickness / 2)
    blocks = [front, back]
    tris = []
    for f in faces:                                        # both caps
        tris.append((f[0], f[1], f[2]))
        tris.append((N + f[0], N + f[2], N + f[1]))
    v = 2 * N
    rings = [np.asarray(poly.exterior.coords)[:-1]] + \
            [np.asarray(r.coords)[:-1] for r in poly.interiors]
    for R in rings:                                        # side walls
        M = len(R)
        blocks.append(panel.uv_to_xyz(R) + n * (center_offset - thickness / 2))
        blocks.append(panel.uv_to_xyz(R) + n * (center_offset + thickness / 2))
        for e in range(M):
            a, b = v + e, v + (e + 1) % M
            c, d = v + M + (e + 1) % M, v + M + e
            tris.append((a, b, c)); tris.append((a, c, d))
        v += 2 * M
    return np.vstack(blocks), np.asarray(tris, int)


def build_shard_mesh(scene, pieces_by_panel, thickness, color_of):
    """Concatenate all shards -> (verts [N,3], faces [M,3], colors [N,3] in [0,1])."""
    V, F, C = [], [], []
    base = 0
    for panel in scene.panels:
        for poly in pieces_by_panel.get(panel.name, []):
            tri = triangulate(poly)
            if tri is None:
                continue
            uv, faces = tri
            verts, tris = _extrude(panel, uv, faces, poly, thickness)
            V.append(verts)
            F.append(tris + base)
            C.append(np.tile(np.asarray(color_of(panel, poly), float), (len(verts), 1)))
            base += len(verts)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3))
    return np.vstack(V), np.vstack(F), np.vstack(C)


def build_stack_mesh(scene, stack_pieces, thickness, n_slots, color_of, channel_order):
    """Concatenate laminated shard stacks -> (verts, faces, colors, groups).

    `stack_pieces`: {panel_name: [(poly, channel, slot)]} from
    `decompose.panel_stack_pieces`. Each slot is laid at an even sub-thickness offset within
    the panel's total `thickness` so a stack of channels occupies the same (u,v) footprint
    but a different micro depth -- overlapping shards laminate instead of z-fighting.

    Faces are grouped and emitted in `channel_order` (contiguous per channel) so exporters
    can tag material groups (e.g. .obj `g`/`usemtl`) without a separate per-face label array.
    `groups` maps channel -> (face_start, face_end) into the returned `faces` array.
    """
    sub_t = thickness / n_slots
    V, F, C = [], [], []
    groups = {}
    base = 0
    for ch in channel_order:
        f_start = sum(len(f) for f in F)
        for panel in scene.panels:
            for poly, pch, slot in stack_pieces.get(panel.name, []):
                if pch != ch:
                    continue
                tri = triangulate(poly)
                if tri is None:
                    continue
                uv, faces = tri
                center_offset = (slot - (n_slots - 1) / 2.0) * sub_t
                verts, tris = _extrude(panel, uv, faces, poly, sub_t, center_offset)
                V.append(verts)
                F.append(tris + base)
                C.append(np.tile(np.asarray(color_of(ch), float), (len(verts), 1)))
                base += len(verts)
        f_end = sum(len(f) for f in F)
        groups[ch] = (f_start, f_end)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)), {}
    return np.vstack(V), np.vstack(F), np.vstack(C), groups
