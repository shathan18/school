"""Export the shard cloud as a per-vertex-coloured PLY mesh (opens in MeshLab/Blender).

Coordinates in millimetres. Colour comes from `color_of(panel, poly) -> (r,g,b) in [0,1]`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .shard_geom import build_shard_mesh, build_stack_mesh


def _write_ply(verts, faces, colors, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    v_mm = verts * 1000.0
    rgb = np.clip(colors * 255, 0, 255).astype(int)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for (x, y, z), (r, g, b) in zip(v_mm, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")
    return str(path)


def export_ply(scene, pieces_by_panel, thickness, color_of, path):
    verts, faces, colors = build_shard_mesh(scene, pieces_by_panel, thickness, color_of)
    return _write_ply(verts, faces, colors, path)


def export_ply_stack(scene, stack_pieces, thickness, n_slots, color_of, channel_order, path):
    """Like `export_ply` but for a laminated (overlapping) shard stack -- see
    `shard_geom.build_stack_mesh` for how channels sharing a footprint are offset in depth."""
    verts, faces, colors, _ = build_stack_mesh(scene, stack_pieces, thickness, n_slots,
                                               color_of, channel_order)
    return _write_ply(verts, faces, colors, path)
