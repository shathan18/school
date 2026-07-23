"""Rough 3D preview of the installation (matplotlib; vedo is a nicer optional upgrade).

Draws the two walls, the woven panels (colored by orientation), the two floor lights,
and the corner. Purely for intuition/orientation — not the physical simulator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _panel_quad(panel):
    (u0, u1), (v0, v1) = panel.u_range, panel.v_range
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    return panel.uv_to_xyz(uv)


def _panel_color(panel):
    """Colour by floor-plan orientation -- there's no family to switch on, so hue
    cycles over `angle` instead: panels facing the same direction read as the same
    colour (e.g. today's old "family A" panels, all angle=pi/2, still all render the
    same blue-ish hue; old "family B", angle=0, still all the same orange-ish hue),
    and anything in between gets its own distinct shade."""
    hue = (panel.angle % np.pi) / np.pi
    r, g, b = hsv_to_rgb((hue, 0.65, 0.85))
    return (r, g, b, 0.35)


def _box_faces(x0, x1, y0, y1, z0, z1):
    """6 quad faces of an axis-aligned box."""
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [c[list(f)] for f in idx], c


def _draw_table(ax, table):
    """Draw scene.table (top slab + optional legs); returns the corner points so the
    caller can include them in the auto-fit bounds (otherwise the table gets clipped)."""
    cx, cy = table.center
    sx, sy = table.size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    boxes = [(x0, x1, y0, y1, table.top_z - table.thickness, table.top_z)]
    if table.legs:
        leg, inset = 0.05, 0.06
        for lx, ly in ((x0 + inset, y0 + inset), (x1 - inset - leg, y0 + inset),
                       (x0 + inset, y1 - inset - leg), (x1 - inset - leg, y1 - inset - leg)):
            boxes.append((lx, lx + leg, ly, ly + leg, 0.0, table.top_z - table.thickness))
    pts = []
    for b in boxes:
        faces, corners = _box_faces(*b)
        ax.add_collection3d(Poly3DCollection(faces, facecolors=[(0.6, 0.45, 0.3, 0.55)],
                                             edgecolors="gray", linewidths=0.3))
        pts.append(corners)
    return np.concatenate(pts)


def save_scene_3d(scene, out_path, elev=22, azim=-60, title=None):
    """`elev`/`azim` set the matplotlib 3D camera (degrees). The default (elev=22, azim=-60) is
    the front view a viewer standing in the room sees; azim rotated by 180 (e.g. azim=120) looks
    at the installation FROM BEHIND -- the back of the woven shard body, facing away from the two
    image walls. `title` overrides the caption."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    quads, colors = [], []
    for p in scene.panels:
        quads.append(_panel_quad(p))
        colors.append(_panel_color(p))
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors="k", linewidths=0.3))

    # walls (image regions) -- also collected into the auto-fit bounds: with a compact
    # far-from-wall sculpture (tabletop scene) the panel+light bound alone would crop
    # the walls out of frame entirely, hiding the whole point of the long throw.
    wall_quads = []
    for w, col in (("A", (0.6, 0.6, 0.6, 0.25)), ("B", (0.5, 0.5, 0.5, 0.25))):
        wall = scene.walls[w]
        o = wall.origin
        quad = np.array([o, o + wall.axis_u * wall.width,
                         o + wall.axis_u * wall.width + wall.axis_v * wall.height,
                         o + wall.axis_v * wall.height])
        wall_quads.append(quad)
        ax.add_collection3d(Poly3DCollection([quad], facecolors=[col], edgecolors="gray"))

    for name, light in scene.lights.items():
        L = light.xyz
        ax.scatter(*L, c="gold", s=120, marker="*", edgecolors="k")
        ax.text(L[0], L[1], L[2], f" L{name}", fontsize=9)

    table_pts = []
    if getattr(scene, "table", None) is not None:
        table_pts.append(_draw_table(ax, scene.table))

    allpts = np.concatenate([q for q in quads] + wall_quads
                            + [np.array([l.xyz for l in scene.lights.values()])]
                            + table_pts)
    lo = allpts.min(axis=0); hi = allpts.max(axis=0); span = (hi - lo).max()
    mid = (hi + lo) / 2
    for setlim, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        setlim(m - span / 2, m + span / 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title or "ShadowArt scene (panel colour = floor-plan orientation)")
    ax.view_init(elev=elev, azim=azim)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
