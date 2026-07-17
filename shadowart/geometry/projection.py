"""Precompute the projection between every panel and both walls.

Each panel casts a shadow on *both* walls: on whichever it's more face-on to (the
image it mainly builds there) and on the other (edge-on / oblique -> cross-talk).
There's no per-panel label saying which is which -- every panel is checked against
both walls, and `wall_coverage_area`/`primary_wall_of` below compute which wall a
panel is currently more relevant to from its actual projected geometry, not a
declared family. Wall A is always lit by Light A and Wall B by Light B, so for a
given wall we use that wall's light for all panels. We cache the homography, its
inverse, the magnification, and a penumbra sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import homography as H


@dataclass
class PanelWallMap:
    panel_name: str
    wall_name: str
    H_pw: np.ndarray          # panel-local (u,v) metres -> wall-local (a,b) metres
    H_wp: np.ndarray          # inverse
    magnification: float      # shadow size / panel feature size, at panel centre
    penumbra_sigma_m: float   # gaussian blur std on the wall (metres) from finite source


def _penumbra_sigma_m(panel, light_xyz, wall, source_radius):
    """Penumbra half-width ~ source_radius * dist(panel,wall)/dist(light,panel).

    Evaluated along the ray through the panel centre. Returned as a gaussian sigma
    (metres on the wall); ~0.5x the geometric half-width reads as a sensible blur.
    """
    c_uv = panel.corners_uv().mean(axis=0)
    P = panel.uv_to_xyz(c_uv[None, :])[0]
    X, t = H.ray_plane_intersect(light_xyz, P, wall.origin, wall.normal)
    if X is None:
        return 0.0
    d_lp = np.linalg.norm(P - light_xyz)
    d_pw = np.linalg.norm(X - P)
    if d_lp < 1e-9:
        return 0.0
    return 0.5 * source_radius * d_pw / d_lp


def build_projection_table(scene):
    """Return {(panel_name, wall_name): PanelWallMap} for all panels x both walls."""
    table = {}
    for wall_name, wall in scene.walls.items():
        light = scene.light_for_wall(wall_name).xyz
        for panel in scene.panels:
            H_pw = H.homography_panel_to_wall(panel, light, wall)
            H_wp = np.linalg.inv(H_pw)
            m = H.magnification(panel, light, wall)
            sigma = _penumbra_sigma_m(panel, light, wall, scene.source_radius)
            table[(panel.name, wall_name)] = PanelWallMap(
                panel_name=panel.name, wall_name=wall_name,
                H_pw=H_pw, H_wp=H_wp, magnification=float(m),
                penumbra_sigma_m=float(sigma),
            )
    return table


def wall_coverage_area(table, panel, wall_name) -> float:
    """Projected area (wall-space m^2) of this panel's full footprint on `wall_name`.

    Accounts for foreshortening naturally: warping a panel's corners through the exact
    projective homography for a wall it's nearly edge-on to collapses them to a thin
    sliver (small area), correctly reading as low relevance -- unlike magnification
    alone, which is only a single-point scale factor and can be misleading (an edge-on
    panel's magnification isn't necessarily small even though its usable coverage is).
    """
    pts = H.apply_homography(table[(panel.name, wall_name)].H_pw, panel.corners_uv())
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def primary_wall_of(scene, table, panel) -> str:
    """Which wall this panel currently contributes the most projected area to --
    dynamic replacement for the old static `panel.family` label. An axis-aligned-style
    panel (angle 0 or pi/2) recovers its old assignment exactly: it's nearly edge-on to
    the other wall, so that wall's coverage area comes out close to zero."""
    return max(scene.walls, key=lambda w: wall_coverage_area(table, panel, w))


def report_table(scene, table):
    """Human-readable summary (magnification gradient + cross-talk blur). "role" is
    computed here from actual projected geometry (`primary_wall_of`), not read from a
    stored label -- whichever wall a panel projects more area onto is its "primary"
    wall for display purposes."""
    panels_by_name = {p.name: p for p in scene.panels}
    lines = ["panel  wall  role       magnif   penumbra(mm)"]
    for (pn, wn), m in sorted(table.items()):
        role = "primary" if wn == primary_wall_of(scene, table, panels_by_name[pn]) else "crosstalk"
        lines.append(f"{pn:5s}  {wn:4s}  {role:9s}  {m.magnification:6.2f}   "
                     f"{m.penumbra_sigma_m * 1000:6.2f}")
    return "\n".join(lines)
