"""Greedy back-projection initialiser (bootstrap + sanity check).

For each panel, sample its primary wall's target (`geometry.projection.primary_wall_of`
-- no family label to read this from) at the spot every panel pixel projects to, then
set opacity so that the n panels primary to that wall *composite* to the target:
    combined darkness = 1 - (1 - a)^n = t   =>   a = 1 - (1 - t)^(1/n).
Non-primary contribution is left near zero. The optimiser refines from here; this alone
already gives a recognisable (if ghosty) result, which is a useful sanity check.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..geometry import homography as H
from ..geometry.projection import primary_wall_of


def _panel_pixel_uv(panel, panel_res):
    Hp, Wp = panel_res
    u = panel.u_range[0] + (np.arange(Wp) + 0.5) / Wp * panel.u_size
    v = panel.v_range[0] + (np.arange(Hp) + 0.5) / Hp * panel.v_size
    uu = np.broadcast_to(u[None, :], (Hp, Wp))
    vv = np.broadcast_to(v[:, None], (Hp, Wp))
    return np.stack([uu, vv], axis=-1)                    # [Hp,Wp,2] metres


def back_project(scene, table, targets):
    """Return opacities [P,Hp,Wp] in [0,1]."""
    Hp, Wp = scene.solve.panel_res
    P = len(scene.panels)
    op = np.zeros((P, Hp, Wp), dtype=np.float32)
    primary_wall = {p.name: primary_wall_of(scene, table, p) for p in scene.panels}
    nfam = {w: sum(1 for pw in primary_wall.values() if pw == w) for w in scene.walls}
    for pi, panel in enumerate(scene.panels):
        wall_name = primary_wall[panel.name]
        wall = scene.walls[wall_name]
        target = targets[wall_name]
        Ht, Wt = target.shape
        m = table[(panel.name, wall_name)]
        uv = _panel_pixel_uv(panel, (Hp, Wp))
        ab = H.apply_homography(m.H_pw, uv)               # panel metres -> wall metres
        col = ab[..., 0] / max(wall.width, 1e-9) * Wt - 0.5
        row = ab[..., 1] / max(wall.height, 1e-9) * Ht - 0.5
        t = map_coordinates(target, [row.ravel(), col.ravel()], order=1,
                            mode="constant", cval=0.0).reshape(Hp, Wp)
        t = np.clip(t, 0.0, 0.999)
        n = max(nfam[wall_name], 1)
        op[pi] = 1.0 - (1.0 - t) ** (1.0 / n)
    return op
