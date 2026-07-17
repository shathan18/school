"""Export the Stochastic Shard Overlap cloud as a single dense .obj (+ .mtl).

Every shard keeps its footprint's (u,v) position but channels sharing a footprint are
laminated at a micro depth-offset (see `shard_geom.build_stack_mesh`) rather than spread
across separate physical planes -- so C/M/Y/K fragments that need to mix subtractively
really do occupy (to fabrication tolerance) the same spatial coordinate. Faces are grouped
per channel (`g`/`usemtl`) so the single mesh still tells a slicer or an assembler which
material each triangle belongs to.

Optionally also appends a full calibration rig in the SAME file, for aligning against an
external renderer (e.g. Rhino) that doesn't share our internal preview:
  - wall planes, panel planes, and light positions (mirrors `preview/render3d.py`'s
    `scene_3d.png`, as real mesh geometry instead of a flat image),
  - an origin RGB axis gizmo (+X red, +Y green, +Z blue) -- OBJ has no formal up-axis
    convention, so this makes ours (Z-up, origin at the wall-corner/floor -- see
    `config/scene.py`) unambiguous on import instead of relying on the importer's guess,
  - wall-corner anchor points and a reference grid on each wall + the floor, for measuring
    scale/alignment directly with the target application's own tools.
This is strictly additive: the shard vertices/faces are built and written first, exactly
as before; every rig/calibration element gets its own new vertex range appended
afterward with its own index offset, so none of it can renumber, reposition, or otherwise
touch a single shard vertex.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .shard_geom import build_stack_mesh
from ..targets import color as _color


_OCTA_FACES = ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
              (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5))
_QUAD_FACES = ((0, 1, 2), (0, 2, 3))


def _wall_quad(wall):
    o = wall.origin
    return np.array([o, o + wall.axis_u * wall.width,
                     o + wall.axis_u * wall.width + wall.axis_v * wall.height,
                     o + wall.axis_v * wall.height])


def _panel_quad(panel):
    (u0, u1), (v0, v1) = panel.u_range, panel.v_range
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    return panel.uv_to_xyz(uv)


def _light_marker(pos, r):
    """Small octahedron centred on a point -- OBJ has no point primitive. Used for both
    light positions and wall-corner calibration anchors (same shape, different radius)."""
    x, y, z = pos
    return np.array([[x + r, y, z], [x - r, y, z],
                     [x, y + r, z], [x, y - r, z],
                     [x, y, z + r], [x, y, z - r]])


def _ribbon(p0, p1, width, up):
    """Flat quad ("line") of the given width from p0 to p1, lying in the plane spanned by
    the segment and `up` (a vector not parallel to it). OBJ has no width-less line
    primitive that reliably renders as a visible, materialed line in every importer, so
    the axis gizmo and reference grid are built from these instead."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return np.tile(p0, (4, 1))
    d = d / length
    side = np.cross(d, np.asarray(up, float))
    sn = np.linalg.norm(side)
    if sn < 1e-9:                          # segment parallel to `up` -> pick any other axis
        alt = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        side = np.cross(d, alt); sn = np.linalg.norm(side)
    side = side / sn * (width / 2.0)
    return np.array([p0 - side, p1 - side, p1 + side, p0 + side])


def _merge_meshes(meshes):
    """[(verts[N,3], local 0-based faces), ...] -> one (verts, faces) with correct offsets."""
    verts, faces = [], []
    base = 0
    for v, f in meshes:
        v = np.asarray(v, float)
        verts.append(v)
        faces += [(base + a, base + b, base + c) for a, b, c in f]
        base += len(v)
    if not verts:
        return np.zeros((0, 3)), []
    return np.vstack(verts), faces


def _wall_gridlines(wall, step):
    """[(p0, p1), ...] world-space endpoints tiling the wall face at `step` spacing."""
    n_u = max(1, int(round(wall.width / step)))
    n_v = max(1, int(round(wall.height / step)))
    lines = []
    for i in range(n_u + 1):
        a = min(i * step, wall.width)
        lines.append((wall.origin + wall.axis_u * a,
                      wall.origin + wall.axis_u * a + wall.axis_v * wall.height))
    for j in range(n_v + 1):
        b = min(j * step, wall.height)
        lines.append((wall.origin + wall.axis_v * b,
                      wall.origin + wall.axis_v * b + wall.axis_u * wall.width))
    return lines


def _floor_extent(scene):
    """(xmax, ymax) covering both lights and every panel's floor-plan footprint,
    ~matching preview's floor size. No family/coord to read -- every panel's own
    `floor_segment_xy()` endpoints are checked instead."""
    xs = [scene.lights["A"].xyz[0], 0.0]
    ys = [scene.lights["B"].xyz[1], 0.0]
    for p in scene.panels:
        for (x, y) in p.floor_segment_xy():
            xs.append(x); ys.append(y)
    return max(xs) * 1.1, max(ys) * 1.1


def _floor_gridlines(xmax, ymax, step):
    n_x = max(1, int(round(xmax / step)))
    n_y = max(1, int(round(ymax / step)))
    lines = []
    for i in range(n_x + 1):
        x = min(i * step, xmax)
        lines.append((np.array([x, 0.0, 0.0]), np.array([x, ymax, 0.0])))
    for j in range(n_y + 1):
        y = min(j * step, ymax)
        lines.append((np.array([0.0, y, 0.0]), np.array([xmax, y, 0.0])))
    return lines


def _rig_groups(scene, light_marker_radius, grid_step, anchor_radius, axis_length):
    """[(group_name, material_name, verts[N,3], faces), ...] for the full calibration rig."""
    groups = []
    for name, wall in scene.walls.items():
        groups.append((f"wall_{name}", f"wall_{name}", _wall_quad(wall), _QUAD_FACES))
    for panel in scene.panels:
        groups.append((f"panel_{panel.name}", "panel", _panel_quad(panel), _QUAD_FACES))
    for name, light in scene.lights.items():
        groups.append((f"light_{name}", "light",
                       _light_marker(light.xyz, light_marker_radius), _OCTA_FACES))

    for name, wall in scene.walls.items():
        v, f = _merge_meshes([(_light_marker(c, anchor_radius), _OCTA_FACES)
                              for c in _wall_quad(wall)])
        groups.append((f"anchor_wall{name}", "anchor", v, f))

        v, f = _merge_meshes([(_ribbon(p0, p1, grid_step * 0.05, wall.normal), _QUAD_FACES)
                              for p0, p1 in _wall_gridlines(wall, grid_step)])
        groups.append((f"grid_wall{name}", "grid", v, f))

    xmax, ymax = _floor_extent(scene)
    v, f = _merge_meshes([(_ribbon(p0, p1, grid_step * 0.05, (0.0, 0.0, 1.0)), _QUAD_FACES)
                          for p0, p1 in _floor_gridlines(xmax, ymax, grid_step)])
    groups.append(("grid_floor", "grid", v, f))

    axis_length = axis_length or 0.15 * max(xmax, ymax, 1.0)
    aw = axis_length * 0.03
    groups.append(("axis_x", "axis_x", _ribbon((0, 0, 0), (axis_length, 0, 0), aw, (0, 0, 1)), _QUAD_FACES))
    groups.append(("axis_y", "axis_y", _ribbon((0, 0, 0), (0, axis_length, 0), aw, (0, 0, 1)), _QUAD_FACES))
    groups.append(("axis_z", "axis_z", _ribbon((0, 0, 0), (0, 0, axis_length), aw, (1, 0, 0)), _QUAD_FACES))
    return groups


def export_obj(scene, stack_pieces, thickness, n_slots, path, channel_order=None,
              include_rig=True, light_marker_radius=0.03, grid_step=0.1,
              anchor_radius=0.015, axis_length=None):
    """Write the laminated shard cloud as a single dense .obj (+ .mtl).

    `include_rig=True` (default) also appends, as extra groups: wall planes (`wall_A`/
    `wall_B`), panel planes (`panel_<name>`), light positions (`light_A`/`light_B`), an
    origin axis gizmo (`axis_x`/`axis_y`/`axis_z`), wall-corner calibration anchors
    (`anchor_wallA`/`anchor_wallB`), and reference grids (`grid_wallA`/`grid_wallB`/
    `grid_floor`) -- see module docstring for why this can't disturb shard placement: it's
    new vertices/faces appended after the shard data, with their own index range.
    """
    channel_order = channel_order or list(_color.CMYK)
    verts, faces, _colors, groups = build_stack_mesh(
        scene, stack_pieces, thickness, n_slots,
        color_of=_color.transmit_rgb, channel_order=channel_order)
    rig = (_rig_groups(scene, light_marker_radius, grid_step, anchor_radius, axis_length)
          if include_rig else [])

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    with open(mtl_path, "w", encoding="ascii") as f:
        for ch in channel_order:
            r, g, b = _color.transmit_rgb(ch)
            f.write(f"newmtl {ch}\nKd {r:.4f} {g:.4f} {b:.4f}\nd 0.55\nillum 1\n\n")
        if include_rig:
            f.write("newmtl wall_A\nKd 0.60 0.60 0.60\nd 0.25\nillum 1\n\n")
            f.write("newmtl wall_B\nKd 0.50 0.50 0.50\nd 0.25\nillum 1\n\n")
            f.write("newmtl panel\nKd 0.55 0.55 0.60\nd 0.35\nillum 1\n\n")
            f.write("newmtl light\nKd 1.00 0.85 0.20\nd 1.00\nillum 2\n\n")
            f.write("newmtl anchor\nKd 1.00 0.00 0.85\nd 1.00\nillum 1\n\n")
            f.write("newmtl grid\nKd 0.70 0.70 0.70\nd 0.40\nillum 1\n\n")
            f.write("newmtl axis_x\nKd 0.90 0.10 0.10\nd 1.00\nillum 1\n\n")
            f.write("newmtl axis_y\nKd 0.10 0.80 0.10\nd 1.00\nillum 1\n\n")
            f.write("newmtl axis_z\nKd 0.10 0.30 0.95\nd 1.00\nillum 1\n\n")

    v_mm = verts * 1000.0
    with open(path, "w", encoding="ascii") as f:
        f.write(f"mtllib {mtl_path.name}\no shards\n")
        for x, y, z in v_mm:                                # -- shard vertices, untouched --
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for ch in channel_order:                             # -- shard faces, untouched --
            start, end = groups.get(ch, (0, 0))
            if end <= start:
                continue
            f.write(f"g {ch}\nusemtl {ch}\n")
            for a, b, c in faces[start:end]:
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")

        base = len(verts)                                    # -- rig: appended, own index range --
        for gname, mtl, gverts, gfaces in rig:
            gverts = np.asarray(gverts)
            if len(gverts) == 0:
                continue
            for x, y, z in gverts * 1000.0:
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            f.write(f"g {gname}\nusemtl {mtl}\n")
            for a, b, c in gfaces:
                f.write(f"f {base + a + 1} {base + b + 1} {base + c + 1}\n")
            base += len(gverts)
    return str(path)
