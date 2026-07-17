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


def save_scene_3d(scene, out_path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    quads, colors = [], []
    for p in scene.panels:
        quads.append(_panel_quad(p))
        colors.append(_panel_color(p))
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors="k", linewidths=0.3))

    # walls (image regions)
    for w, col in (("A", (0.6, 0.6, 0.6, 0.25)), ("B", (0.5, 0.5, 0.5, 0.25))):
        wall = scene.walls[w]
        o = wall.origin
        quad = np.array([o, o + wall.axis_u * wall.width,
                         o + wall.axis_u * wall.width + wall.axis_v * wall.height,
                         o + wall.axis_v * wall.height])
        ax.add_collection3d(Poly3DCollection([quad], facecolors=[col], edgecolors="gray"))

    for name, light in scene.lights.items():
        L = light.xyz
        ax.scatter(*L, c="gold", s=120, marker="*", edgecolors="k")
        ax.text(L[0], L[1], L[2], f" L{name}", fontsize=9)

    allpts = np.concatenate([q for q in quads] + [np.array([l.xyz for l in scene.lights.values()])])
    lo = allpts.min(axis=0); hi = allpts.max(axis=0); span = (hi - lo).max()
    mid = (hi + lo) / 2
    for setlim, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        setlim(m - span / 2, m + span / 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("ShadowArt scene (panel colour = floor-plan orientation)")
    ax.view_init(elev=22, azim=-60)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
