"""Build DIAGONAL panels (family 'D') and place them in the room.

A diagonal is a still-vertical acrylic plane rotated by an angle about +z, anchored at
(x0, y0). Unlike the axis-aligned families it is oblique to *both* walls, so a single
shard on it casts a wanted shadow on Wall A *and* Wall B at once — forcing a joint
decision instead of the two walls being solved independently.

Two placement modes (both required, so "random vs deliberate" is itself a result):
  - random:            seeded uniform angle + anchor within room bounds (reproducible).
  - deliberate:        angle ~45 deg on the corner bisector at spread depths.
  - deliberate_search: coarse grid search maximising joint-intersection against targets.

Placement is validated by reject-and-resample: a candidate is rejected unless every
corner ray reaches both walls (homography solvable) and at least one corner lands inside
each wall's image rectangle (so the panel genuinely contributes to both images).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..config.scene import Panel
from . import homography as H


def _wall_overlap_fraction(panel, scene, wall_name):
    """Fraction of a coarse (u,v) sample of the panel whose shadow lands inside the wall's
    image rectangle. 0 if any ray misses the wall plane entirely."""
    wall = scene.walls[wall_name]
    light = scene.light_for_wall(wall_name).xyz
    u = np.linspace(panel.u_range[0], panel.u_range[1], 9)
    v = np.linspace(panel.v_range[0], panel.v_range[1], 9)
    uu, vv = np.meshgrid(u, v)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    ab = H.project_uv_to_wall(panel, light, wall, uv)
    if not np.all(np.isfinite(ab)):
        return 0.0
    inside = ((ab[:, 0] >= 0) & (ab[:, 0] <= wall.width) &
              (ab[:, 1] >= 0) & (ab[:, 1] <= wall.height))
    return float(inside.mean())


def _corners_hit_both_walls(panel, scene, require_inside=True, min_frac=0.05) -> bool:
    """Valid placement iff every corner ray reaches both wall planes (homography solvable)
    and (if `require_inside`) a non-trivial fraction of the panel's shadow lands inside
    each wall's image rectangle, so the diagonal genuinely contributes to BOTH images."""
    for wall_name, wall in scene.walls.items():
        light = scene.light_for_wall(wall_name).xyz
        ab = H.project_uv_to_wall(panel, light, wall, panel.corners_uv())
        if not np.all(np.isfinite(ab)):
            return False                                    # a ray misses this wall plane
    if require_inside:
        if _wall_overlap_fraction(panel, scene, "A") < min_frac:
            return False
        if _wall_overlap_fraction(panel, scene, "B") < min_frac:
            return False
    return True


def _make_panel(name, angle, x0, y0, u_range, v_range, thickness) -> Panel:
    return Panel(name, "D", 0.0, tuple(u_range), tuple(v_range), thickness,
                 angle=float(angle), x0=float(x0), y0=float(y0))


def _bounds(scene):
    """Anchor sampling box: near the corner, between the corner and the lights. A diagonal
    anchored too far from the corner overshoots both walls (strong magnification), so we
    keep anchors in the near band where the shadow still lands on the image regions."""
    lax = scene.lights["A"].xyz[0]        # Light A x -> family-A panels need 0 < x < lax
    lby = scene.lights["B"].xyz[1]        # Light B y -> family-B panels need 0 < y < lby
    lo, hi = 0.12, 0.32                   # fraction of the light distance (near-corner band)
    return (lo * lax, hi * lax), (lo * lby, hi * lby)


def _random_panels(scene, spec, rng, u_range, v_range, thickness) -> List[Panel]:
    n = int(spec.get("count", 2))
    a0, a1 = spec.get("angle_range_deg", [15.0, 75.0])
    # minimum angular separation so multiple diagonals don't cluster at one angle
    min_sep = math.radians(float(spec.get("min_angle_sep_deg", 12.0)))
    (x_lo, x_hi), (y_lo, y_hi) = _bounds(scene)
    out, angles, attempts, max_attempts = [], [], 0, 400 * max(n, 1)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        angle = math.radians(rng.uniform(a0, a1))
        if any(abs(angle - a) < min_sep for a in angles):    # too close to an existing one
            continue
        x0 = rng.uniform(x_lo, x_hi)
        y0 = rng.uniform(y_lo, y_hi)
        cand = _make_panel(f"D{len(out)}", angle, x0, y0, u_range, v_range, thickness)
        if _corners_hit_both_walls(cand, scene):
            out.append(cand); angles.append(angle)
    if len(out) < n:
        raise ValueError(
            f"could not place {n} diagonal panels (min_angle_sep_deg="
            f"{math.degrees(min_sep):.0f}) after {attempts} attempts "
            f"(angle_range_deg={[a0, a1]}, bounds x{[round(x_lo,2), round(x_hi,2)]} "
            f"y{[round(y_lo,2), round(y_hi,2)]}). Widen angle_range_deg / lower "
            "min_angle_sep_deg / move the lights.")
    return out


def _deliberate_panels(scene, spec, rng, u_range, v_range, thickness,
                       targets=None, table_fn=None) -> List[Panel]:
    """Diagonals SPREAD across a range of angles (not all parallel) on the corner bisector
    at evenly spread depths, so multiple diagonals don't cluster at one angle.

    In `deliberate_search` mode (targets provided) a coarse (angle, depth) grid is scored
    by how much of each candidate's footprint lands where *both* walls want dark, keeping
    the best `count` subject to a minimum angular separation."""
    n = int(spec.get("count", 2))
    (x_lo, x_hi), (y_lo, y_hi) = _bounds(scene)
    d_lo = max(x_lo, y_lo); d_hi = min(x_hi, y_hi)         # shared bisector range
    a0, a1 = spec.get("angle_range_deg", [30.0, 60.0])    # spread band around the 45deg bisector

    if targets is None:                                    # plain deliberate: spread angles
        # angles fanned across [a0,a1]; depths staggered so panels don't overlap in space
        angles = ([45.0] if n == 1 else list(np.linspace(a0, a1, n)))
        depths = ([0.5 * (d_lo + d_hi)] if n == 1
                  else list(np.linspace(d_lo, d_hi, n + 2)[1:-1]))
        out = []
        for i, (ang, d) in enumerate(zip(angles, depths)):
            cand = _make_panel(f"D{i}", math.radians(ang), d, d,
                               u_range, v_range, thickness)
            if _corners_hit_both_walls(cand, scene):
                out.append(cand)
        if len(out) < n:                                   # fall back to resampling (spread)
            return _random_panels(scene, {**spec, "angle_range_deg": [a0, a1],
                                          "min_angle_sep_deg": spec.get("min_angle_sep_deg", 12.0)},
                                  rng, u_range, v_range, thickness)
        return out

    # deliberate_search: rank candidates by joint-dark overlap against the targets.
    angles = [math.radians(a) for a in (35.0, 45.0, 55.0)]
    depths = np.linspace(d_lo, d_hi, 4)
    scored = []
    for angle in angles:
        for d in depths:
            cand = _make_panel("Dtmp", angle, d, d, u_range, v_range, thickness)
            if not _corners_hit_both_walls(cand, scene):
                continue
            scored.append((_joint_dark_score(cand, scene, targets), angle, d))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for i, (_, angle, d) in enumerate(scored[:n]):
        out.append(_make_panel(f"D{i}", angle, d, d, u_range, v_range, thickness))
    if len(out) < n:
        raise ValueError("deliberate_search could not place enough valid diagonals; "
                         "widen the room or reduce count.")
    return out


def _joint_dark_score(panel, scene, targets) -> float:
    """Fraction of the panel that, projected to both walls, lands where BOTH want dark.

    A cheap placement heuristic (no solve): sample a coarse (u,v) grid, project to each
    wall, look up whether each wall's target is dark there, and count pixels dark on both.
    """
    Hp = Wp = 48
    u = panel.u_range[0] + (np.arange(Wp) + 0.5) / Wp * panel.u_size
    v = panel.v_range[0] + (np.arange(Hp) + 0.5) / Hp * panel.v_size
    uu, vv = np.meshgrid(u, v)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    dark = {}
    for wall_name, wall in scene.walls.items():
        light = scene.light_for_wall(wall_name).xyz
        ab = H.project_uv_to_wall(panel, light, wall, uv)
        tgt = targets[wall_name]
        tgt = tgt if tgt.ndim == 2 else tgt.mean(axis=-1)   # colour -> intensity proxy
        Ht, Wt = tgt.shape
        col = np.clip((ab[:, 0] / max(wall.width, 1e-9) * Wt).astype(int), 0, Wt - 1)
        row = np.clip((ab[:, 1] / max(wall.height, 1e-9) * Ht).astype(int), 0, Ht - 1)
        onwall = np.isfinite(ab).all(axis=1)
        d = np.zeros(uv.shape[0], bool)
        d[onwall] = tgt[row[onwall], col[onwall]] > 0.5
        dark[wall_name] = d
    both = dark["A"] & dark["B"]
    return float(both.sum()) / float(uv.shape[0])


def build_diagonal_panels(scene, spec, rng=None, targets=None) -> List[Panel]:
    """Return a list of diagonal Panels per `spec` (the parsed `panels.diagonals` block).

    `spec` keys: count, placement ('random'|'deliberate'|'deliberate_search'),
    angle_range_deg, u_range, v_range, seed, or an explicit `panels:` list of
    {angle_deg, x0, y0}.  `targets` (optional) enables deliberate_search scoring.
    """
    thickness = scene.material_thickness
    # diagonals are narrower than the full-wall axis-aligned panels: a diagonal near the
    # corner throws a strongly magnified shadow, so a full 1.2 m width overshoots both walls.
    u_range = tuple(spec.get("u_range", (0.0, 0.60)))
    v_range = tuple(spec.get("v_range", (0.10, 1.55)))

    if "panels" in spec:                                   # explicit list
        out = []
        for i, d in enumerate(spec["panels"]):
            cand = _make_panel(f"D{i}", math.radians(float(d["angle_deg"])),
                               float(d["x0"]), float(d["y0"]), u_range, v_range, thickness)
            if not _corners_hit_both_walls(cand, scene, require_inside=False):
                raise ValueError(f"explicit diagonal D{i} (angle {d['angle_deg']} deg at "
                                 f"({d['x0']},{d['y0']})) does not project onto both walls.")
            out.append(cand)
        return out

    if rng is None:
        rng = np.random.default_rng(int(spec.get("seed", scene.solve.seed)))
    placement = str(spec.get("placement", "random"))
    if placement == "random":
        return _random_panels(scene, spec, rng, u_range, v_range, thickness)
    if placement in ("deliberate", "deliberate_search"):
        t = targets if placement == "deliberate_search" else None
        return _deliberate_panels(scene, spec, rng, u_range, v_range, thickness, targets=t)
    raise ValueError(f"unknown diagonals.placement '{placement}' "
                     "(expected random|deliberate|deliberate_search).")
