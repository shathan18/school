"""Cross-lap (egg-crate) joints + the per-panel drawing bundle handed to exporters.

Each panel is cut from a clear perspex rectangle (the carrier). Wherever two panels
cross we cut a half-height slot into each so they slide together; one of the pair
slots from the top down to mid-height, the other from the bottom up -- with no family
concept left to decide which is which, `slots_for_panel` uses a deterministic
lexical tie-break on panel name instead (reproduces the old family-A-notches-top
behaviour exactly for today's A0..A3/B0..B3 panels, and stays well-defined for any
two arbitrarily-named/angled panels). The opaque shadow shapes ('pieces') ride on
the carrier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from shapely.geometry import Polygon

from ..geometry.panels import weave_crossings


@dataclass
class PanelDrawing:
    name: str
    u_range: tuple
    v_range: tuple
    outline: List[tuple]              # carrier rectangle ring, (u,v) metres
    slots: List[List[tuple]] = field(default_factory=list)     # rings to cut
    pieces: List[Polygon] = field(default_factory=list)        # opaque shadow shapes


def _slot_ring(u_center, half_w, v0, v1):
    return [(u_center - half_w, v0), (u_center + half_w, v0),
            (u_center + half_w, v1), (u_center - half_w, v1), (u_center - half_w, v0)]


def slots_for_panel(scene, panel, crossings):
    half_w = (scene.material_thickness + scene.fab.joint_clearance) / 2.0
    v0, v1 = panel.v_range
    vmid = 0.5 * (v0 + v1)
    out = []
    for c in crossings:
        if c.panelA_name == panel.name:
            other_name, u = c.panelB_name, c.uA
        elif c.panelB_name == panel.name:
            other_name, u = c.panelA_name, c.uB
        else:
            continue
        if panel.name < other_name:
            out.append(_slot_ring(u, half_w, vmid, v1))        # notch from the top
        else:
            out.append(_slot_ring(u, half_w, v0, vmid))        # notch from the bottom
    return out


def build_panel_drawings(scene, pieces_by_panel) -> List[PanelDrawing]:
    crossings = weave_crossings(scene)
    drawings = []
    for panel in scene.panels:
        u0, u1 = panel.u_range; v0, v1 = panel.v_range
        outline = [(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)]
        drawings.append(PanelDrawing(
            name=panel.name,
            u_range=panel.u_range, v_range=panel.v_range,
            outline=outline,
            slots=slots_for_panel(scene, panel, crossings),
            pieces=list(pieces_by_panel.get(panel.name, [])),
        ))
    return drawings
