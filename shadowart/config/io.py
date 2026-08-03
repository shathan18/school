"""Load + validate a scene from YAML into the dataclasses in scene.py."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from .scene import FabParams, Light, Panel, Scene, SolveParams, TableSpec, TurntableSpec, Wall


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise ValueError(f"scene: missing required key '{key}' in {ctx}")
    return d[key]


def _resolve_palette(col: dict) -> list:
    """Resolve the shard palette from the `color:` block.

    `color.preset` (e.g. 'cmyk'|'muted'|'noir') selects a named palette from
    `targets.color.PALETTES` and takes precedence over an explicit `color.palette` list.
    Every resolved colour must exist in `targets.color.PERSPEX` -- error early (with an
    actionable message) rather than failing deep in quantisation."""
    from ..targets import color as _color
    preset = col.get("preset")
    if preset is not None:
        key = str(preset).lower()
        if key not in _color.PALETTES:
            raise ValueError(
                f"scene: unknown color.preset '{preset}'. "
                f"Known presets: {sorted(_color.PALETTES)}")
        palette = list(_color.PALETTES[key])
    else:
        palette = list(col.get("palette", ["C", "M", "Y", "K"]))
    unknown = [n for n in palette if n not in _color.PERSPEX]
    if unknown:
        raise ValueError(
            f"scene: color palette references unknown perspex colour(s) {unknown}. "
            f"Known: {[n for n in _color.PERSPEX if n != 'clear']}")
    return palette


def _load_turntable(raw: dict):
    """Build walls+lights from a `turntable:` block, if present.

    Returns (walls, lights, spec) or (None, None, None). A scene declares EITHER the
    fixed `walls:`/`lights:` corner rig OR a `turntable:` rig -- never both, since the
    turntable derives its walls and lights from the single physical lamp/wall.
    """
    t = raw.get("turntable")
    if t is None:
        return None, None, None
    if "walls" in raw or "lights" in raw:
        raise ValueError(
            "scene: use either `turntable:` or `walls:`/`lights:`, not both -- the "
            "turntable block derives its per-stop walls and lights from one lamp and "
            "one wall.")
    from ..geometry.turntable import build_walls_and_lights
    stops = tuple(float(s) for s in _require(t, "stops_deg", "turntable"))
    if len(stops) < 1:
        raise ValueError("scene: turntable.stops_deg must list at least one view angle.")
    names = t.get("names")
    spec = TurntableSpec(
        stops_deg=stops,
        center=tuple(map(float, t.get("center", (0.0, 0.0)))),
        view_azimuth_deg=float(t.get("view_azimuth_deg", 0.0)),
        wall_distance=float(_require(t, "wall_distance", "turntable")),
        lamp_distance=float(_require(t, "lamp_distance", "turntable")),
        lamp_height=float(t.get("lamp_height", 0.5)),
        wall_width=float(_require(t, "wall_width", "turntable")),
        wall_height=float(_require(t, "wall_height", "turntable")),
        wall_z0=float(t.get("wall_z0", 0.0)),
        names=tuple(str(n) for n in names) if names is not None else None,
    )
    walls, lights = build_walls_and_lights(spec)
    return walls, lights, spec


def load_scene(path: Union[str, Path]) -> Scene:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # --- walls + lights ----------------------------------------------------
    # Two rigs are supported: the original fixed corner (Wall A at x=0, Wall B at y=0,
    # one light each) and a turntable (one lamp, one wall, N rotation stops).
    walls, lights, turntable = _load_turntable(raw)
    if walls is None:
        walls = {}
        for name in ("A", "B"):
            w = _require(raw["walls"], name, "walls")
            walls[name] = Wall(
                name=name,
                plane=w.get("plane", "x" if name == "A" else "y"),
                offset=float(w.get("offset", 0.0)),
                width_axis=w.get("width_axis", "y" if name == "A" else "x"),
                width=float(_require(w, "width", f"walls.{name}")),
                height=float(_require(w, "height", f"walls.{name}")),
                z0=float(w.get("z0", 0.0)),
            )
        lights = {name: Light(name=name, pos=tuple(map(float, _require(raw["lights"], name, "lights")["pos"])))
                  for name in ("A", "B")}
    source_radius = float(raw.get("lights", raw.get("turntable", {})).get("source_radius", 0.003))

    # --- panels (woven lattice) --------------------------------------------
    # Every panel is independent: its own angle + anchor, no grouping. A flat
    # list, not the old familyA/familyB(/diagonal) sections -- there's nothing
    # left to group by. `angle_deg=90` reproduces what used to be called
    # "family A" (parallel to Wall A); `angle_deg=0` reproduces "family B" --
    # just two particular angles now, not special cases.
    thickness = float(raw.get("material", {}).get("thickness", 0.003))
    panels = []
    for i, p in enumerate(raw["panels"]):
        panels.append(Panel(
            name=p.get("name", f"P{i}"),
            angle=math.radians(float(_require(p, "angle_deg", f"panels[{i}]"))),
            anchor=tuple(map(float, _require(p, "anchor", f"panels[{i}]"))),
            u_range=tuple(map(float, _require(p, "u_range", f"panels[{i}]"))),
            v_range=tuple(map(float, _require(p, "v_range", f"panels[{i}]"))),
            thickness=thickness,
        ))

    # --- solve / fabricate params -----------------------------------------
    s = raw.get("solve", {})
    solve = SolveParams(
        mode=str(s.get("mode", "partition")),
        fragment_size=float(s.get("fragment_size", 0.09)),
        fragment_max_area=float(s.get("fragment_max_area", 0.014)),
        fragment_min_area=float(s.get("fragment_min_area", 0.0012)),
        shard_gap=float(s.get("shard_gap", 0.0)),
        depth_bias=float(s.get("depth_bias", 0.0)),
        wall_res=tuple(s.get("wall_res", (300, 300))),
        panel_res=tuple(s.get("panel_res", (280, 280))),
        iters=int(s.get("iters", 300)),
        lr=float(s.get("lr", 0.05)),
        lambda_sparsity=float(s.get("lambda_sparsity", 0.002)),
        lambda_tv=float(s.get("lambda_tv", 0.001)),
        lambda_crosstalk=float(s.get("lambda_crosstalk", 0.0)),
        seed=int(s.get("seed", 0)),
        restarts=int(s.get("restarts", 1)),
        panel_restarts=int(s.get("panel_restarts", 1)),
        time_budget=float(s.get("time_budget", 0.0)),
        search_panels=bool(s.get("search_panels", False)),
        damage_weight=float(s.get("damage_weight", 0.0)),
        credit_weight=float(s.get("credit_weight", 0.0)),
        score_ssim_weight=float(s.get("score_ssim_weight", 1.0)),
        score_edge_weight=float(s.get("score_edge_weight", 0.5)),
        score_crosstalk_weight=float(s.get("score_crosstalk_weight", 0.25)),
        diagonal_frac=float(s.get("diagonal_frac", 0.0)),
        search_anchor_range=tuple(map(float, s.get("search_anchor_range", (0.5, 2.4)))),
        search_standoff=float(s.get("search_standoff", 0.5)),
        search_mag_cap=float(s.get("search_mag_cap", 3.0)),
        search_u_size_range=tuple(map(float, s.get("search_u_size_range", (0.25, 0.75)))),
        search_v_range=tuple(map(float, s.get("search_v_range", (0.03, 1.18)))),
    )
    f = raw.get("fabricate", {})
    fab = FabParams(
        kerf=float(f.get("kerf", 0.0002)),
        min_feature=float(f.get("min_feature", 0.004)),
        threshold=float(f.get("threshold", 0.5)),
        sheet_size=tuple(f.get("sheet_size", (0.6, 0.4))),
        formats=tuple(f.get("formats", ("dxf", "svg"))),
        joint_clearance=float(f.get("joint_clearance", 0.0001)),
    )

    t = raw.get("table")
    table = None
    if t is not None:
        table = TableSpec(
            top_z=float(_require(t, "top_z", "table")),
            center=tuple(map(float, _require(t, "center", "table"))),
            size=tuple(map(float, _require(t, "size", "table"))),
            thickness=float(t.get("thickness", 0.04)),
            legs=bool(t.get("legs", True)),
        )

    col = raw.get("color", {})
    palette = _resolve_palette(col)
    scene = Scene(walls=walls, lights=lights, panels=panels,
                  source_radius=source_radius, material_thickness=thickness,
                  solve=solve, fab=fab, table=table,
                  color_palette=palette,
                  white_threshold=float(col.get("white_threshold", 0.90)),
                  color_max_layers=int(col.get("max_layers", 2)),
                  color_max_stack=int(col.get("max_stack", 3)),
                  overlap_jitter=float(col.get("overlap_jitter", 0.0)),
                  overlap_shard_budget=int(col.get("shard_budget", 220)),
                  overlap_detail_bias=float(col.get("detail_bias", 2.0)),
                  overlap_penumbra_min_feature_sigmas=float(col.get("penumbra_min_feature_sigmas", 2.0)),
                  turntable=turntable)
    _sanity_check(scene)
    return scene


def _sanity_check(scene: Scene):
    """Cheap geometric sanity checks with actionable messages.

    Every panel may contribute to every wall (there's no family label saying which), so
    every endpoint of every panel's floor-plan footprint must sit between each light and
    that light's wall -- one rule, applied uniformly, rather than a separate rule per
    family or per rig. Expressed as a signed distance along the wall normal `n`: the wall
    plane sits at `n.x = offset` and its light at `n.L`, so a panel point `P` must satisfy
    `offset <= n.P < n.L` (with the inequalities flipped when the light lies on the
    negative side of the plane).

    The wall-plane side is a non-strict bound: a panel's extent legitimately touches the
    plane at the room's corner (every axis-aligned panel in the example scene runs its
    u_range from exactly 0.0). The light side stays strict -- a panel reaching or passing
    the lamp is a real physical impossibility, not a boundary case.

    This works unchanged for the fixed corner rig and for a turntable, whose per-stop
    walls and lamps are just more (wall, light) pairs in the same dicts.
    """
    for name, wall in scene.walls.items():
        n = wall.normal
        L = scene.light_for_wall(name).xyz
        s_wall = float(n @ wall.origin)          # exact plane position along the normal
        s_light = float(n @ L)
        if abs(s_light - s_wall) < 1e-9:
            raise ValueError(f"light {name} lies in the plane of wall {name}; it cannot light it.")
        lo, hi = (s_wall, s_light) if s_light > s_wall else (s_light, s_wall)
        for p in scene.panels:
            for (x, y) in p.floor_segment_xy():
                s = float(n @ np.array([x, y, 0.0]))
                inside = (s_wall <= s < s_light) if s_light > s_wall else (s_light < s <= s_wall)
                if not inside:
                    raise ValueError(
                        f"panel {p.name}: endpoint (x={x:.3f}, y={y:.3f}) is at {s:.3f} along "
                        f"wall {name}'s normal, outside [{lo:.3f}, {hi:.3f}] -- it must sit "
                        f"between light {name} (at {s_light:.3f}) and wall {name} (at "
                        f"{s_wall:.3f}) to cast a shadow on it.")
    if scene.table is not None:
        # A panel standing over the table must not dip below its top -- the shard body
        # would physically intersect the furniture. Judged at the footprint midpoint so a
        # panel merely brushing the table's edge isn't rejected on a technicality.
        t = scene.table
        x0, x1 = t.center[0] - t.size[0] / 2, t.center[0] + t.size[0] / 2
        y0, y1 = t.center[1] - t.size[1] / 2, t.center[1] + t.size[1] / 2
        for p in scene.panels:
            (ax, ay), (bx, by) = p.floor_segment_xy()
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if x0 <= mx <= x1 and y0 <= my <= y1 and p.v_range[0] < t.top_z:
                raise ValueError(
                    f"panel {p.name}: bottom v={p.v_range[0]:.3f} is below the table top "
                    f"z={t.top_z:.3f} it stands over -- the body would intersect the table.")
