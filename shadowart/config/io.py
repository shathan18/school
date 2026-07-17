"""Load + validate a scene from YAML into the dataclasses in scene.py."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import yaml

from .scene import FabParams, Light, Panel, Scene, SolveParams, Wall


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


def load_scene(path: Union[str, Path]) -> Scene:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # --- walls -------------------------------------------------------------
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

    # --- lights ------------------------------------------------------------
    lights = {name: Light(name=name, pos=tuple(map(float, _require(raw["lights"], name, "lights")["pos"])))
              for name in ("A", "B")}
    source_radius = float(raw["lights"].get("source_radius", 0.003))

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

    col = raw.get("color", {})
    palette = _resolve_palette(col)
    scene = Scene(walls=walls, lights=lights, panels=panels,
                  source_radius=source_radius, material_thickness=thickness,
                  solve=solve, fab=fab,
                  color_palette=palette,
                  white_threshold=float(col.get("white_threshold", 0.90)),
                  color_max_layers=int(col.get("max_layers", 2)),
                  color_max_stack=int(col.get("max_stack", 3)),
                  overlap_jitter=float(col.get("overlap_jitter", 0.0)),
                  overlap_shard_budget=int(col.get("shard_budget", 220)),
                  overlap_detail_bias=float(col.get("detail_bias", 2.0)),
                  overlap_penumbra_min_feature_sigmas=float(col.get("penumbra_min_feature_sigmas", 2.0)))
    _sanity_check(scene)
    return scene


def _sanity_check(scene: Scene):
    """Cheap geometric sanity checks with actionable messages.

    Every panel may contribute to either or both walls (there's no family label
    saying which), so both endpoints of every panel's floor-plan footprint must
    sit between each light and that light's wall's plane -- one check, applied
    uniformly, rather than a separate rule per family. The wall-plane side is a
    non-strict bound (a panel's along-wall extent legitimately touches x=0 or
    y=0 at the room's corner -- that's normal, e.g. every axis-aligned-style
    panel in the example scene runs its u_range from exactly 0.0); the light
    side stays strict (a panel must not reach or pass the light itself, which
    would be a real physical impossibility, not a boundary case).
    """
    la, lb = scene.lights["A"].xyz, scene.lights["B"].xyz
    if la[0] <= 0 or lb[1] <= 0:
        raise ValueError("Light A must have x>0 and Light B must have y>0.")
    for p in scene.panels:
        for (x, y) in p.floor_segment_xy():
            if not (0.0 <= x < la[0]):
                raise ValueError(
                    f"panel {p.name}: endpoint x={x:.3f} must satisfy "
                    f"0 <= x < light A x ({la[0]}) so it sits between Light A and Wall A.")
            if not (0.0 <= y < lb[1]):
                raise ValueError(
                    f"panel {p.name}: endpoint y={y:.3f} must satisfy "
                    f"0 <= y < light B y ({lb[1]}) so it sits between Light B and Wall B.")
