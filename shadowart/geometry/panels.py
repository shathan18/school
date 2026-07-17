"""Woven-lattice geometry: where panels physically cross.

Rigid flat panels cannot literally weave, so "warp and weft" means an egg-crate:
wherever two panel footprints cross in the floor plan, we cut a half-depth slot into
each so they slide together. Every panel is independently positioned/angled (no
family grouping), so crossing detection is general 2D line-segment intersection
(`_segment_intersection`) on each panel's own `floor_segment_xy()` -- what used to be
called "family A x family B" (always exactly perpendicular by construction) is simply
the theta=90 special case of the same general computation, reproduced exactly
(verified: identical crossing count/positions on the existing baseline scene both
before and after this generalised).

fabricate/joints.py turns `Crossing`s into cut geometry; today it only wires up
`uA`/`uB` for the top/bottom slot split -- `angle_deg`/`slot_width` are computed here
but not yet consumed there for the actual notch shape (still a fixed-width rectangular
slot regardless of angle; cutting a properly angle-dependent parallelogram slot is
follow-on work).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Crossing:
    panelA_name: str
    panelB_name: str
    x: float
    y: float
    z_range: tuple          # overlap in z where both panels exist
    uA: float               # local u on panel A where the slot goes
    uB: float               # local u on panel B where the slot goes
    angle_deg: float = 90.0     # angle between the two panels' floor-plan directions
                                # (90 = perpendicular, today's only case; smaller = more oblique)
    slot_width: float = 0.0     # in-plane notch width this crossing needs (thickness/sin(angle));
                                # equals material_thickness exactly when angle_deg=90


def _segment_intersection(p0, p1, q0, q1) -> Optional[Tuple[float, float]]:
    """(x,y) intersection of two line SEGMENTS p0->p1 and q0->q1, or None if they
    don't cross within both segments' extents (this also naturally rejects parallel
    segments, e.g. two family-A panels, without needing a special case for it)."""
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _direction_xy(panel) -> np.ndarray:
    return np.array([np.cos(panel.angle), np.sin(panel.angle)])


def _crossing_angle_rad(panel_a, panel_b) -> float:
    """Angle between two panels' floor-plan directions, folded into (0, pi/2] -- 90
    degrees for perpendicular (today's only case in the example scene), shrinking as
    they become more parallel (which is exactly where a physical slot gets
    impractically wide)."""
    da, db = _direction_xy(panel_a), _direction_xy(panel_b)
    cos_theta = float(np.clip(abs(np.dot(da, db)), 0.0, 1.0))
    return float(np.arccos(cos_theta))


def _local_u(panel, point_xy) -> float:
    """Local u-coordinate on `panel` of a floor-plan point known to lie on its line."""
    ax, ay = panel.anchor
    dx, dy = np.cos(panel.angle), np.sin(panel.angle)
    return (point_xy[0] - ax) * dx + (point_xy[1] - ay) * dy


def weave_crossings(scene) -> List[Crossing]:
    """Every pair of panels is checked for a crossing (`_segment_intersection` already
    rejects parallel pairs on its own, so there's no family-based short-circuit needed
    -- two panels that happen to share an angle are simply never found to cross,
    correctly, without a special case for it)."""
    out = []
    panels = scene.panels
    for i, pa in enumerate(panels):
        for pb in panels[i + 1:]:
            p0, p1 = pa.floor_segment_xy()
            q0, q1 = pb.floor_segment_xy()
            pt = _segment_intersection(p0, p1, q0, q1)
            if pt is None:
                continue
            z0 = max(pa.v_range[0], pb.v_range[0])
            z1 = min(pa.v_range[1], pb.v_range[1])
            if z1 <= z0:
                continue
            theta = _crossing_angle_rad(pa, pb)
            slot_w = scene.material_thickness / max(np.sin(theta), 1e-6)
            out.append(Crossing(pa.name, pb.name, pt[0], pt[1], (z0, z1),
                                uA=_local_u(pa, pt), uB=_local_u(pb, pt),
                                angle_deg=float(np.degrees(theta)), slot_width=slot_w))
    return out


def slot_depth(scene) -> float:
    """Half-lap: each panel is notched half of the *other* panel's thickness deep.

    With equal thickness this is simply thickness/2 on each side of the crossing plane,
    i.e. a slot half the panel height tall centred on the shared z-range. This is the
    notch's DEPTH (into the panel's own thickness axis) -- independent of crossing
    angle, unlike `Crossing.slot_width` (the notch's WIDTH, in-plane, which does depend
    on angle). An oblique crossing needs a wider notch, not a deeper one.
    """
    return scene.material_thickness / 2.0
