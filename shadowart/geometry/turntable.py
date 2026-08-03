"""Turntable rig: N views from ONE lamp and ONE wall plus a rotating base.

The physical piece is a single lamp, a single wall, and a turntable the shard assembly
stands on. To show view *k* you rotate the assembly by `stops_deg[k]` about the vertical
axis through `center`. Nothing about the rig moves.

The solver, however, wants every view present *simultaneously* so one opacity field is
optimised against all of them at once. The two are related by a change of frame:

    rotating the ASSEMBLY by +theta under a fixed rig
      ==  rotating the RIG by -theta around a fixed assembly

so each turntable stop is emitted as its own `Wall(plane='n', ...)` + `Light` pair,
obtained by rotating the one physical wall and the one physical lamp by `-theta` about
the turntable axis. The existing renderer already loops over `scene.walls`, so an N-stop
turntable needs no new physics -- only this coordinate change. `tests/test_turntable.py`
asserts the equivalence numerically.

Layout, looking down on the floor plane (default `view_azimuth_deg = 0`):

        lamp  ----->  [ assembly on turntable ]  ----->  wall
      (azimuth+180)            (center)                (azimuth)

The lamp sits behind the assembly and the wall in front, collinear through `center`, so
the shadow lands on the wall. Wall normals point back toward the assembly, matching the
convention of the fixed corner walls A/B (whose normals point into the room).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config.scene import Light, Panel, TurntableSpec, Wall


def _rot2(pt: Sequence[float], center: Sequence[float], deg: float) -> np.ndarray:
    """Rotate a 2D point about `center` by `deg` (counter-clockwise, degrees)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    return np.array([center[0] + c * dx - s * dy,
                     center[1] + s * dx + c * dy])


def rig_geometry(spec: TurntableSpec) -> Tuple[np.ndarray, np.ndarray, float]:
    """The ONE physical lamp position and wall centre, in world coordinates.

    Returns (lamp_xyz, wall_center_xyz, wall_normal_deg) for the rig at rest, i.e. what
    you would actually build in the room. `stops_deg` never enters here -- the rig is
    the same for every view; only the assembly turns.
    """
    cx, cy = spec.center
    az = math.radians(spec.view_azimuth_deg)
    ux, uy = math.cos(az), math.sin(az)                     # centre -> wall direction
    wall_c = np.array([cx + spec.wall_distance * ux,
                       cy + spec.wall_distance * uy,
                       0.0])
    lamp = np.array([cx - spec.lamp_distance * ux,
                     cy - spec.lamp_distance * uy,
                     spec.lamp_height])
    # Normal points from the wall back toward the assembly (into the "room"), matching
    # the A/B convention where the normal faces the light.
    return lamp, wall_c, spec.view_azimuth_deg + 180.0


def stop_names(spec: TurntableSpec) -> List[str]:
    if spec.names is not None:
        if len(spec.names) != len(spec.stops_deg):
            raise ValueError(
                f"turntable: {len(spec.names)} names for {len(spec.stops_deg)} stops -- "
                f"they must correspond one-to-one.")
        return list(spec.names)
    return [f"V{k}" for k in range(len(spec.stops_deg))]


def build_walls_and_lights(spec: TurntableSpec) -> Tuple[Dict[str, Wall], Dict[str, Light]]:
    """Emit one `Wall` + one `Light` per turntable stop (the -theta rig frames).

    Wall and Light for a stop share a name, so `Scene.light_for_wall` resolves without
    any special casing.
    """
    lamp, wall_c, wall_n_deg = rig_geometry(spec)
    names = stop_names(spec)
    walls: Dict[str, Wall] = {}
    lights: Dict[str, Light] = {}
    for name, theta in zip(names, spec.stops_deg):
        # Rotate the rig by -theta about the turntable axis.
        n_deg = wall_n_deg - theta
        t = math.radians(n_deg)
        n_hat = np.array([math.cos(t), math.sin(t)])
        u_hat = np.array([-math.sin(t), math.cos(t)])        # == Wall.axis_u for plane='n'
        w_xy = _rot2(wall_c[:2], spec.center, -theta)
        l_xy = _rot2(lamp[:2], spec.center, -theta)
        walls[name] = Wall(
            name=name, plane="n",
            offset=float(n_hat @ w_xy),          # signed distance of the plane from the origin
            width_axis="",                       # unused for plane='n'
            width=spec.wall_width, height=spec.wall_height, z0=spec.wall_z0,
            normal_deg=float(n_deg),
            # Centre the image rectangle on where the wall actually is, not on the
            # world origin -- otherwise a turntable placed away from (0,0) images
            # the wrong patch of wall.
            lateral_offset=float(u_hat @ w_xy),
        )
        lights[name] = Light(name=name, pos=(float(l_xy[0]), float(l_xy[1]), float(lamp[2])))
    return walls, lights


def rotate_panels(panels: Sequence[Panel], center: Sequence[float], deg: float) -> List[Panel]:
    """The assembly turned by `deg` about the vertical axis through `center`.

    This is the *physical* motion (used by previews, by per-stop 3D exports, and by the
    equivalence test); the solver never needs it, because `build_walls_and_lights`
    already folds it into the rig frames.
    """
    out = []
    for p in panels:
        ax, ay = _rot2(p.anchor, center, deg)
        out.append(replace(p, angle=p.angle + math.radians(deg), anchor=(float(ax), float(ay))))
    return out


def footprint_bbox(panels: Sequence[Panel]) -> Tuple[float, float, float, float]:
    """Axis-aligned floor-plan bounding box (x0, y0, x1, y1) of the assembly.

    Measured on panel EXTENTS (both endpoints of every footprint segment), not on panel
    centres -- the 30x30 cm installation limit is a limit on the real occupied floor
    area, and a centre-based box silently understates it by half a panel length.
    """
    if not panels:
        raise ValueError("footprint_bbox: no panels")
    pts = np.array([xy for p in panels for xy in p.floor_segment_xy()])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def footprint_size(panels: Sequence[Panel]) -> Tuple[float, float]:
    """(width_x, width_y) of the assembly footprint, in metres. See `footprint_bbox`."""
    x0, y0, x1, y1 = footprint_bbox(panels)
    return (x1 - x0, y1 - y0)


def swept_radius(panels: Sequence[Panel], center: Sequence[float]) -> float:
    """Radius of the disc the assembly sweeps as the turntable rotates.

    Because the piece turns, the footprint that must fit the installation limit is this
    swept CIRCLE, not the bounding box at one stop. `2 * swept_radius` is the number to
    compare against the 0.30 m constraint.
    """
    pts = np.array([xy for p in panels for xy in p.floor_segment_xy()])
    d = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    return float(d.max())
