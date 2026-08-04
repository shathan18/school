"""Pearl Girl turntable build -- one script for the 60x60 reference and the 30x30 arm.

WHY THIS FILE EXISTS
--------------------
`out_final/pearl_girl3_turntable/` ships an assembly, cut sheets and a README claiming
IoU 0.83 / 646 shards -- but the script that produced it was never committed, and neither
was the rig it used (lamp distance, wall distance, image size are all absent from the
README). Nothing can be honestly described as "better than the 60x60 baseline" while the
baseline cannot be re-run. So this rebuilds it from the geometry the repo *does* record,
and the 30x30 arm from the same code path, so the two differ only in the parameters under
study.

THE RECONSTRUCTION IS NOT THE ORIGINAL. The optical construction is taken from
`scenes/tabletop60.yaml` / `scenes/tabletop.yaml`, which are the committed 0.60 m and
0.30 m bodies, and re-expressed as a turntable. Expect the reference numbers to land near
the README's, not on them; the point is a baseline measured by the same ruler as
everything that follows, not a bit-exact replica.

GEOMETRY
--------
Both arms use the committed construction: a lamp at `lamp_to_wall` from the wall, the body
centred so that magnification `m = lamp_to_wall / lamp_to_body` maps the body exactly onto
the imaged band. The lamp height is *solved*, not guessed, so the body's top and bottom
land on the band's top and bottom (see `_lamp_height`).

The 60 vs 30 difference is then exactly the roadmap's question: at a fixed 1.8 m image a
30 cm body must run at 6x where a 60 cm body runs at 3x, and every fabrication limit is
multiplied by that magnification on the wall.

Run:  .venv\\Scripts\\python.exe pearl3_baseline.py --arm 60
      .venv\\Scripts\\python.exe pearl3_baseline.py --arm 30
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from shadowart.config.scene import (
    FabParams, Panel, Scene, SolveParams, TurntableSpec,
)
from shadowart.forward.renderer import Renderer
from shadowart.forward.backend import DEVICE
from shadowart.geometry import turntable as TT
from shadowart.geometry.projection import build_projection_table
from shadowart.solve import decompose
from shadowart.targets import color as C
from shadowart.targets import engrave
from shadowart import metrics


# --- the piece -------------------------------------------------------------
# back -> side -> front is the natural reading order of a figure turning to face you.
#
# The `pearlN` set replaced the `girl3` set because girl3_back.png's matte is torn: it scores
# 5.08 on the raggedness measure in `check_targets.py` against 2.26 for pearlN_back, with
# large white bites taken out of the turban and shoulder. Those bites are open to the image
# border rather than enclosed, so they measure 0% on a hole test and no hole-fill repairs
# them -- the solver was faithfully reproducing missing content, and the renders showed
# chunks of the figure simply absent. pearlN is ~40% smaller in pixels, which costs nothing:
# resolution measured FLAT as a lever (600px = 300px), and the built piece resolves roughly
# 55 features across the image, far below either source.
VIEW_TARGETS = {
    "back": "examples/pearlN_back.png",
    "side": "examples/pearlN_side.png",
    "front": "examples/pearlN_front.png",
}

# The same rotation supervised every 60 deg instead of every 120, using the two three-quarter
# renders. The brief asks for "one continuous turning sequence" rather than three independent
# images, and the only way to hold the optimiser to that is to score it at angles BETWEEN the
# headline stops -- otherwise nothing constrains what the piece looks like while it turns, and
# the in-between angles are most of what a viewer walking around it actually sees.
SEQUENCE_STOPS = (15.0, 75.0, 135.0, 195.0, 255.0)
SEQUENCE_TARGETS = {
    "back": "examples/girl3_back.png",
    "q34back": "examples/girl3_q34back.png",
    "side": "examples/girl3_profile.png",
    "q34front": "examples/girl3_q34front.png",
    "front": "examples/girl3_front.png",
}


@dataclass
class BuildConfig:
    """Every free parameter of a build, in one place, so a sweep can vary one at a time."""
    name: str
    body: float = 0.60                  # sheet size AND floor footprint of the assembly (m)
    n_per_family: int = 2               # sheets per orientation
    pitch: float = 0.06                 # depth spacing between sheets of one family (m)
    # Orientation of each sheet family, degrees from +x. None = one family per turntable
    # stop, each parallel to that stop's wall (i.e. face-on to it). A 2-family egg-crate
    # CANNOT serve 3 views: `primary_wall_of` is winner-take-all, so two orientations win
    # two views and the third gets no panels at all and is never built. See the
    # `2family` arm, which is kept precisely to demonstrate that failure.
    family_angles_deg: Tuple[float, ...] | None = None
    image: float = 1.80                 # imaged square on the wall (m)
    image_z0: float = 0.30              # bottom of the imaged band (m)
    lamp_to_wall: float = 3.00          # lamp -> wall plane (m)
    body_z0: float = 0.78               # bottom of the sheets (m); they stand on a table
    source_radius: float = 0.0015       # LED die radius -> penumbra
    stops_deg: Tuple[float, ...] = (15.0, 135.0, 255.0)
    thickness: float = 0.003
    min_feature: float = 0.005          # smallest cuttable feature, PANEL space (m)
    # Smallest ENGRAVED feature, panel space (m). Not the same limit and not the same order
    # of magnitude. `min_feature` exists because a shard cut OUT of a sheet has to be big
    # enough to survive being a separate physical object; an engraved region is a mark on a
    # sheet that stays whole, so the only limit is the beam spot -- roughly 0.1-0.2 mm on a
    # CO2 laser, i.e. under half a panel pixel here. Conflating the two silently imposed a
    # 5 mm floor on tonal detail that no physical constraint asked for, and the brief
    # explicitly removed any minimum shard size.
    engrave_min_feature: float = 0.0002
    wall_res: int = 600
    panel_res: int = 600
    fragment_size: float = 0.09         # target shard size, WALL space (m)
    fragment_max_area: float = 0.014
    shard_budget: int = 260
    detail_bias: float = 2.0
    palette: str = "noir"               # 4-level grayscale engraving stand-in
    max_stack: int = 3
    penumbra_sigmas: float = 2.0
    seed: int = 0
    # Cross-talk-aware host selection. `damage_weight` prices the stray shadow a shard will
    # throw across the OTHER views when choosing which panel hosts it; `credit_weight`
    # scales the reward for stray light that lands where another view genuinely wanted it.
    # Both are off by default in the shipped solver, so the host greedy is currently blind
    # to cross-talk -- it maximises coverage on the shard's own view alone.
    damage_weight: float = 0.0
    credit_weight: float | None = None
    match_tol: float = 0.30
    # >0 picks each shard's TONE jointly with its host panel, blending toward what the
    # other views want where the stray shadow will land. Without it a shard's tone is fixed
    # from its own view before the host is chosen, so it can only ever help another view by
    # luck.
    colour_blend: float = 0.0
    colour_primary_tol: float = 0.10
    # Transmittance of the three engraved tones (the 4th tone is the unengraved sheet). When
    # set, this replaces the `palette` stock list -- the piece is clear acrylic frosted by
    # the laser, not coloured perspex, so the tones are chosen rather than bought.
    engrave_levels: tuple | None = None
    # Scales how strongly each shard is toned. The renders come out visibly darker than the
    # targets even when the silhouette matches, and IoU cannot see that -- it only scores the
    # mark's outline. This is the direct control over the tonal error SSIM and RMSE measure.
    intensity_gain: float = 1.0
    white_threshold: float = 0.90
    # Iterations of cross-talk pre-compensation (0 = off, the previous behaviour). See
    # `precompensate_targets`: the decomposer is re-aimed at a target that already
    # discounts the stray light the other views' panels will add. `_gain` damps the
    # correction; the undamped map oscillates (measured 0.690 -> 0.598 -> 0.665 -> 0.641).
    precompensate: int = 0
    precompensate_gain: float = 1.0
    # Low-pass the targets the SOLVER is aimed at, in wall pixels (0 = off). Scoring always
    # uses the untouched target, so this can only pay off if it is genuinely true that the
    # detail being removed was never reproducible.
    #
    # It is: at 600 px across a 1.8 m image one pixel is 3 mm, the penumbra is sigma 3.6-4.6 mm
    # (~1.3 px) and the smallest cuttable feature projects to 32 mm (~11 px). Everything the
    # target contains above that band is energy the optics will destroy no matter what the
    # solver does -- but the solver cannot know that, so it spends shards chasing it, and the
    # shards it spends are then unavailable for structure it could have got right. This is the
    # attenuator-display result (content beyond the stack's depth of field should be filtered
    # out, not fought) applied to the target instead of to the layers.
    target_blur_px: float = 0.0
    # If set, shrink the sheets until the assembly's SWEPT CIRCLE fits this diameter (m).
    # The piece turns, so the swept circle -- not the bounding box at one stop -- is what
    # has to fit the installation limit.
    footprint_limit: float | None = None
    views: Dict[str, str] = field(default_factory=lambda: dict(VIEW_TARGETS))

    @property
    def sheet(self) -> float:
        """Sheet length actually used, after honouring `footprint_limit`.

        A sheet of length L centred on the axis, offset by `o` across its own direction,
        reaches `hypot(L/2, o)` from the axis. Solving that for L at the outermost sheet
        is why a 30 cm limit does NOT permit a 30 cm sheet: at 6 cm pitch the corner sheets
        swing 31 cm and quietly break the constraint.
        """
        if self.footprint_limit is None:
            return self.body
        max_off = self.pitch * (self.n_per_family - 1) / 2.0
        room = (self.footprint_limit / 2.0) ** 2 - max_off ** 2
        if room <= 0:
            raise ValueError(
                f"footprint_limit {self.footprint_limit:.3f} m is too small for "
                f"{self.n_per_family} sheets at {self.pitch:.3f} m pitch (they alone span "
                f"{2*max_off:.3f} m). Reduce pitch or sheet count.")
        return min(self.body, 2.0 * math.sqrt(room))

    @property
    def magnification(self) -> float:
        """Image size / body size -- fixed by the two sizes alone, not by where things sit."""
        return self.image / self.sheet

    @property
    def lamp_to_body(self) -> float:
        return self.lamp_to_wall / self.magnification

    @property
    def body_to_wall(self) -> float:
        return self.lamp_to_wall - self.lamp_to_body


def _lamp_height(cfg: BuildConfig) -> float:
    """Lamp z such that the body's vertical extent maps EXACTLY onto the imaged band.

    A ray from the lamp at height L through body height z lands at `L + m*(z - L)`. Setting
    the body's bottom `body_z0` onto the band's bottom `image_z0` and solving for L gives
    one equation, and the top then lands on the top automatically because the body height
    is `image / m` by construction. Guessing this instead (and letting the figure ride high
    or low in the frame) would cost IoU for reasons that have nothing to do with the
    optimisation under study.
    """
    m = cfg.magnification
    return (m * cfg.body_z0 - cfg.image_z0) / (m - 1.0)

def _family_angles(cfg: BuildConfig) -> List[float]:
    """Sheet orientations, in degrees from +x -- one family per view.

    Natural choice: each family PARALLEL to its own stop's wall. With the wall at azimuth 0,
    the wall normal seen at stop `theta` sits at `180 - theta`, so a sheet parallel to it lies
    at `270 - theta`. A face-on sheet has the largest projected area on its own view and so
    wins it under `primary_wall_of`, which is what gives every view panels of its own.

    But sheet orientation lives mod 180 (a sheet has no front or back), so two stops 180 deg
    apart want the SAME family -- and since `primary_wall_of` is winner-take-all, one of them
    would then win both and the other would be left with no panels and never be built at all.
    That is the failure that silently deleted the side view from the two-family egg-crate. It
    bites as soon as the stops span more than 180 deg, which the five-stop turning sequence
    does. When the natural angles collide, fall back to spreading the families evenly over the
    180 deg of distinct orientations, phased to sit as close to the natural angles as it can.
    """
    if cfg.family_angles_deg is not None:
        return list(cfg.family_angles_deg)
    natural = [(270.0 - t) % 180.0 for t in cfg.stops_deg]
    n = len(natural)

    def sep(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    if all(sep(natural[i], natural[j]) > 1.0 for i in range(n) for j in range(i + 1, n)):
        return natural
    step = 180.0 / n
    best = min(range(int(step) * 4),
               key=lambda p: sum(sep((p / 4.0 - i * step) % 180.0, natural[i])
                                 for i in range(n)))
    return [(best / 4.0 - i * step) % 180.0 for i in range(n)]


def build_scene(cfg: BuildConfig) -> Scene:
    """Woven assembly on a turntable, sized and placed by `cfg`."""
    spec = TurntableSpec(
        stops_deg=cfg.stops_deg,
        center=(0.0, 0.0),
        view_azimuth_deg=0.0,
        wall_distance=cfg.body_to_wall,
        lamp_distance=cfg.lamp_to_body,
        lamp_height=_lamp_height(cfg),
        wall_width=cfg.image, wall_height=cfg.image, wall_z0=cfg.image_z0,
        names=tuple(cfg.views),
    )
    walls, lights = TT.build_walls_and_lights(spec)

    # Sheets are centred on the turntable axis so the swept circle stays as small as the
    # sheet length allows -- for a piece that turns, the swept circle is the footprint that
    # has to obey the installation limit, not the bounding box at one stop.
    half = cfg.sheet / 2.0
    v = (cfg.body_z0, cfg.body_z0 + cfg.sheet)
    offs = (np.arange(cfg.n_per_family) - (cfg.n_per_family - 1) / 2.0) * cfg.pitch
    panels: List[Panel] = []
    for fi, deg in enumerate(_family_angles(cfg)):
        a = math.radians(deg)
        d = np.array([math.cos(a), math.sin(a)])            # along the sheet
        n = np.array([-math.sin(a), math.cos(a)])           # across it: the pitch direction
        for i, o in enumerate(offs):
            anchor = -half * d + float(o) * n                # local u=0 end of the sheet
            panels.append(Panel(f"F{fi}_{i}", a, (float(anchor[0]), float(anchor[1])),
                                (0.0, cfg.sheet), v, cfg.thickness))

    # A shard may not be finer than the light can resolve OR than the laser can cut. Both
    # limits live in panel space; the wall sees them multiplied by the magnification.
    min_wall = cfg.min_feature * cfg.magnification
    if cfg.footprint_limit is not None:
        swept = 2 * TT.swept_radius(panels, (0.0, 0.0))
        if swept > cfg.footprint_limit + 1e-9:
            raise AssertionError(
                f"assembly sweeps {swept*100:.1f} cm, over the "
                f"{cfg.footprint_limit*100:.0f} cm limit")
    return Scene(
        walls=walls, lights=lights, panels=panels,
        source_radius=cfg.source_radius, material_thickness=cfg.thickness,
        solve=SolveParams(
            mode="partition",
            fragment_size=cfg.fragment_size,
            fragment_max_area=cfg.fragment_max_area,
            fragment_min_area=min_wall ** 2,
            wall_res=(cfg.wall_res, cfg.wall_res),
            panel_res=(cfg.panel_res, cfg.panel_res),
            seed=cfg.seed,
        ),
        fab=FabParams(min_feature=cfg.min_feature),
        color_palette=(list(engrave.register(cfg.engrave_levels))
                       if cfg.engrave_levels else list(C.PALETTES[cfg.palette])),
        color_max_stack=cfg.max_stack,
        overlap_shard_budget=cfg.shard_budget,
        overlap_detail_bias=cfg.detail_bias,
        overlap_penumbra_min_feature_sigmas=cfg.penumbra_sigmas,
        white_threshold=cfg.white_threshold,
        turntable=spec,
    )


def load_targets(cfg: BuildConfig) -> Dict[str, np.ndarray]:
    wr = (cfg.wall_res, cfg.wall_res)
    return {name: C.load_color_target(path, wr, white_thr=0.90)
            for name, path in cfg.views.items()}


def _solve_once(cfg, scene, table, renderer, names, solve_targets):
    """One decomposition pass aimed at `solve_targets`. Returns (products, panel_T, pred)."""
    out = decompose.fragment_shards_overlap(
        scene, table, solve_targets, names=names, white_thr=scene.white_threshold,
        max_stack=scene.color_max_stack, shard_budget=scene.overlap_shard_budget,
        detail_bias=scene.overlap_detail_bias,
        penumbra_min_feature_sigmas=scene.overlap_penumbra_min_feature_sigmas,
        damage_weight=cfg.damage_weight, credit_weight=cfg.credit_weight,
        match_tol=cfg.match_tol, colour_blend=cfg.colour_blend,
        colour_primary_tol=cfg.colour_primary_tol,
        intensity_gain=cfg.intensity_gain,
        seed=cfg.seed)
    stack_colorid, stack_intensity = out[0], out[6]
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    return out, panel_T, renderer.render_color_np(panel_T)


def band_limit_targets(targets, sigma_px):
    """Gaussian low-pass every target to the band the optics can actually deliver.

    Applied to what the solver is AIMED at only. `run_build` still scores against the
    original, so a gain here is real and not a redefinition of success.
    """
    from scipy.ndimage import gaussian_filter
    return {w: np.clip(gaussian_filter(t, sigma=(sigma_px, sigma_px, 0)), 0.0, 1.0)
            for w, t in targets.items()}


def precompensate_targets(scene, table, renderer, panel_T, targets, gain=1.0):
    """Re-aim each view's target so the stray light it will receive is already discounted.

    The renderer composites by TRANSMITTANCE: what reaches the wall is the product over
    all panels, so for one view

        T_seen = T_own * T_stray

    where `T_own` is that view's own panels and `T_stray` is everything else in the beam.
    The decomposer, however, fits `T_own` to the target directly and is simply never told
    about `T_stray` -- which is why the reconstruction comes out systematically too dark
    (measured: foreground area 0.30 against a target of 0.21). Asking it instead for

        T_own_wanted = T_target / T_stray

    makes the product land on the target. Since `T_stray` depends on the solution, this is
    applied as a fixed-point iteration: solve, measure the stray light, re-aim, re-solve.

    This is the standard pre-compensation used in projector-camera systems (radiometric
    compensation: measure what the surface adds, divide it out of what you send). Values
    above 1 are clipped -- they mean the stray light alone is already darker than the
    target asks for, which no arrangement of the view's own shards can undo.
    """
    from shadowart.geometry.projection import primary_wall_of
    prim = {p.name: primary_wall_of(scene, table, p) for p in scene.panels}
    adjusted = {}
    for w in scene.walls:
        q = panel_T.copy()
        for gi, p in enumerate(scene.panels):
            if prim[p.name] == w:
                q[gi] = 1.0                      # keep ONLY the stray contributors
        stray = np.clip(renderer.render_color_np(q)[w], 1e-3, 1.0)
        adjusted[w] = np.clip(targets[w] / stray ** gain, 0.0, 1.0)
    return adjusted


def run_build(cfg: BuildConfig, out: Path, verbose: bool = True) -> dict:
    """Solve, render, measure. Returns the metrics dict (also written to metrics.json)."""
    t_start = time.perf_counter()
    scene = build_scene(cfg)
    table = build_projection_table(scene)
    renderer = Renderer(scene, table)
    targets = load_targets(cfg)
    names = C.palette_names(scene.color_palette)

    if verbose:
        _report_geometry(cfg, scene, table)

    solve_targets = targets
    if cfg.target_blur_px > 0:
        solve_targets = band_limit_targets(targets, cfg.target_blur_px)
    trace, best = [], None
    for k in range(cfg.precompensate + 1):
        raw, panel_T, pred = _solve_once(cfg, scene, table, renderer, names, solve_targets)
        score = metrics.evaluate_multiview(targets, pred)["_summary"]
        trace.append({"pass": k, "mean_iou": score["mean_iou"], "min_iou": score["min_iou"]})
        if verbose and cfg.precompensate:
            print(f"  pre-compensation pass {k}: mean IoU {score['mean_iou']:.3f}  "
                  f"min {score['min_iou']:.3f}")
        # The fixed point is not guaranteed monotone, so keep the best pass rather than
        # the last -- shipping a solution worse than one already computed is indefensible.
        if best is None or score["mean_iou"] > best[0]:
            best = (score["mean_iou"], k, raw, panel_T, pred)
        if k < cfg.precompensate:
            solve_targets = precompensate_targets(scene, table, renderer, panel_T,
                                                  band_limit_targets(targets, cfg.target_blur_px)
                                                  if cfg.target_blur_px > 0 else targets,
                                                  cfg.precompensate_gain)

    _, best_pass, raw, panel_T, pred = best
    stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity = raw
    pred_primary = _primary_only_render(scene, table, renderer, panel_T)

    m = metrics.evaluate_multiview(targets, pred)
    m_primary = metrics.evaluate_multiview(targets, pred_primary)
    duty = metrics.shard_view_duty(scene, table, fragments, targets, scene.white_threshold)
    n_shards = len(fragments)

    result = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
        "geometry": {
            "magnification": cfg.magnification,
            "lamp_to_body_m": cfg.lamp_to_body,
            "body_to_wall_m": cfg.body_to_wall,
            "lamp_height_m": _lamp_height(cfg),
            "footprint_m": list(TT.footprint_size(scene.panels)),
            "swept_diameter_m": 2 * TT.swept_radius(scene.panels, (0.0, 0.0)),
            "min_feature_panel_mm": cfg.min_feature * 1000,
            "min_feature_wall_mm": cfg.min_feature * cfg.magnification * 1000,
            "penumbra_sigma_mm": {w: table[(scene.panels[0].name, w)].penumbra_sigma_m * 1000
                                  for w in scene.walls},
            "family_angles_deg": _family_angles(cfg),
            "panels_per_view": _panels_per_view(scene, table),
        },
        "views": {w: m[w] for w in scene.walls},
        "summary": m["_summary"],
        # The same piece measured the way the original README measured it: shards only,
        # cross-talk deleted. Kept so the published 0.83 can be checked against this
        # rebuild -- and so the gap between the two columns, which IS the projection
        # noise, is impossible to overlook.
        "views_shards_only": {w: m_primary[w] for w in scene.walls},
        "summary_shards_only": m_primary["_summary"],
        "crosstalk_cost_iou": m_primary["_summary"]["mean_iou"] - m["_summary"]["mean_iou"],
        "shards": {"n_fragments": n_shards, "stack_depths": stack_depths and
                   {str(d): int((np.asarray(stack_depths) == d).sum())
                    for d in sorted(set(stack_depths))},
                   "collisions_resolved": int(resolved), "budget": budget_stats},
        "duty": duty,
        "precompensation": {"iters": cfg.precompensate, "best_pass": best_pass,
                            "trace": trace},
        "runtime_s": time.perf_counter() - t_start,
        "device": DEVICE.type,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "opacity.npy", opacity)

    # Live objects for a downstream fabrication or verification pass. Attached after the
    # JSON is written, never before -- none of this is serialisable and metrics.json is the
    # file everything else in the project reads.
    result["_artifacts"] = {
        "scene": scene, "table": table, "renderer": renderer, "targets": targets,
        "names": names, "panel_T": panel_T, "pred": pred, "opacity": opacity,
        "stack_colorid": stack_colorid, "stack_intensity": stack_intensity,
        "fragments": fragments,
    }

    from shadowart.preview.wallview import save_color_comparison
    save_color_comparison(targets, pred, out / "preview_views.png")

    if verbose:
        print(metrics.format_multiview_report(m, f"{cfg.name}: ALL light (honest)"))
        s0, s1 = m_primary["_summary"], m["_summary"]
        print(f"\nshards-only (cross-talk deleted -- the README's convention): "
              f"mean IoU {s0['mean_iou']:.3f}, min {s0['min_iou']:.3f}")
        print(f"CROSS-TALK COSTS {s0['mean_iou'] - s1['mean_iou']:.3f} mean IoU "
              f"-- that gap is the projection noise, and it is the thing to attack")
        print(f"shards: {n_shards} laminated layers over "
              f"{len(scene.panels)} sheets   collisions resolved {resolved}")
        print(f"duty: {duty['frac_multi']*100:.0f}% of shards serve >=2 views, "
              f"{duty['frac_all']*100:.0f}% serve all {len(scene.walls)} "
              f"(mean {duty['mean_views']:.2f} views/shard)")
        print(f"runtime {result['runtime_s']:.1f}s on {DEVICE.type}   -> {out/'metrics.json'}")
    return result


def _primary_only_render(scene, table, renderer, panel_T):
    """Each view rendered from ONLY the panels primary to it -- i.e. cross-talk deleted.

    This is not a physical image: no such render can be built, because the other panels are
    still in the beam. It is the decomposition's own ceiling, and it is what
    `cli._print_fragment_stats` reports as "reconstruction IoU (shards-only, excl.
    cross-talk)" -- so it is almost certainly the convention behind the 0.83 in
    `out_final/pearl_girl3_turntable/README.txt`. Reported alongside the honest number so
    the two are never confused.
    """
    from shadowart.geometry.projection import primary_wall_of
    prim = {p.name: primary_wall_of(scene, table, p) for p in scene.panels}
    out = {}
    for w in scene.walls:
        q = panel_T.copy()
        for gi, p in enumerate(scene.panels):
            if prim[p.name] != w:
                q[gi] = 1.0                      # fully transparent = panel removed
        out[w] = renderer.render_color_np(q)[w]
    return out


def _panels_per_view(scene, table) -> Dict[str, int]:
    """How the sheets split across the views. A view with 0 panels is never built at all."""
    from shadowart.geometry.projection import primary_wall_of
    counts = {w: 0 for w in scene.walls}
    for p in scene.panels:
        counts[primary_wall_of(scene, table, p)] += 1
    return counts


def _report_geometry(cfg: BuildConfig, scene: Scene, table) -> None:
    fw, fh = TT.footprint_size(scene.panels)
    swept = 2 * TT.swept_radius(scene.panels, (0.0, 0.0))
    print(f"\n=== {cfg.name} ===")
    print(f"sheet {cfg.sheet*100:.1f} cm -> image {cfg.image*100:.0f} cm  "
          f"= magnification {cfg.magnification:.2f}x")
    print(f"lamp {cfg.lamp_to_body:.3f} m behind body, wall {cfg.body_to_wall:.3f} m in front, "
          f"lamp height {_lamp_height(cfg):.3f} m")
    print(f"assembly: {len(scene.panels)} sheets, {cfg.pitch*100:.0f} cm pitch, "
          f"footprint {fw*100:.0f} x {fh*100:.0f} cm, swept circle {swept*100:.0f} cm diameter")
    print(f"min feature {cfg.min_feature*1000:.1f} mm on the sheet "
          f"-> {cfg.min_feature*cfg.magnification*1000:.1f} mm on the wall "
          f"(magnification multiplies every fabrication limit)")
    sig = [table[(p.name, w)].penumbra_sigma_m * 1000 for p in scene.panels for w in scene.walls]
    print(f"penumbra sigma {min(sig):.1f}-{max(sig):.1f} mm on the wall")
    print(f"stops at {', '.join(f'{s:g}' for s in cfg.stops_deg)} deg -> "
          f"{', '.join(scene.walls)}")


ARMS = {
    "60": BuildConfig(name="60x60 reference (reconstructed)", body=0.60),
    "30": BuildConfig(name="30x30 arm (as previously handicapped)", body=0.30),
    # Kept as evidence, not as a candidate: the two-family egg-crate the README describes.
    # `primary_wall_of` is winner-take-all, so its two orientations win the back and front
    # stops and the side stop is left with no panels of its own and is never built.
    "2family": BuildConfig(name="60x60, 2-family egg-crate (side view starves)",
                           body=0.60, n_per_family=3, family_angles_deg=(90.0, 0.0)),
    # The 30 cm limit applies to the SWEPT circle, so the sheets come down to 29.4 cm.
    "30fit": BuildConfig(name="30x30, footprint-legal", body=0.30, footprint_limit=0.30),
    "30pc": BuildConfig(name="30x30, footprint-legal + cross-talk pre-compensation",
                        body=0.30, footprint_limit=0.30, precompensate=3),
    "60pc": BuildConfig(name="60x60 + cross-talk pre-compensation",
                        body=0.60, precompensate=3),
    # Damage-aware host selection. Measured +0.058 mean IoU and cut the cross-talk cost from
    # 0.137 to 0.076, and the result is identical for every weight from 0.01 to 2.0 -- the
    # choice is really just "on", so there is nothing here to tune or to overfit.
    "30v2": BuildConfig(name="30x30, footprint-legal + damage-aware hosts",
                        body=0.30, footprint_limit=0.30, damage_weight=0.25),
    "60v2": BuildConfig(name="60x60 + damage-aware hosts",
                        body=0.60, damage_weight=0.25),
    # Best layout found. Tightening the pitch from 6 cm to 2 cm and going to 6 sheets per
    # view turns the cross-talk CONSTRUCTIVE (cost -0.033: the all-light render now beats
    # the shards-only one). Past 18 sheets the curve is flat -- 30 sheets buys +0.005 -- so
    # this is the knee, not the maximum.
    "30v3": BuildConfig(name="30x30 candidate: tight stack + damage-aware hosts",
                        body=0.30, footprint_limit=0.30, damage_weight=0.25,
                        n_per_family=6, pitch=0.02),
    # Same piece, scored every 60 deg of rotation instead of every 120: the two extra stops
    # are the three-quarter views, so the turn itself is optimised rather than only its three
    # headline positions.
    "30seq": BuildConfig(name="30x30 turning sequence (5 stops, 60 deg apart)",
                         body=0.30, footprint_limit=0.30, damage_weight=0.25,
                         n_per_family=6, pitch=0.02,
                         engrave_levels=(0.60, 0.30, 0.10),
                         stops_deg=SEQUENCE_STOPS, views=dict(SEQUENCE_TARGETS)),
    # Best three-stop configuration found, with the four-level engrave alphabet.
    "30v4": BuildConfig(name="30x30 candidate + 4-level engrave",
                        body=0.30, footprint_limit=0.30, damage_weight=0.25,
                        n_per_family=6, pitch=0.02, engrave_levels=(0.60, 0.30, 0.10)),
    # 30v4 minus one sheet per family. Gate C (single-panel ablation) found sheet F0_5 doing
    # no measurable work in 30v4, so the question is whether 15 sheets buy what 18 do -- 17%
    # less material and three fewer parts to cut and slot. Ablation says "this sheet is idle",
    # not "this sheet is removable": drop it and the solver redistributes onto the rest, so
    # the only way to answer is to re-solve at 15 and compare through the same gates.
    "30v5": BuildConfig(name="30x30, 5 sheets per family",
                        body=0.30, footprint_limit=0.30, damage_weight=0.25,
                        n_per_family=5, pitch=0.02, engrave_levels=(0.60, 0.30, 0.10)),
    # SIX SHEETS TOTAL -- two per family. A hard build constraint, not a search result, and
    # far below the 18-sheet knee: the layout sweep measured cross-talk turning constructive
    # only past ~12 sheets, so at 6 the stray light is noise again and every parameter tuned
    # against 18 has to be re-examined. Two things do get easier. The footprint solve frees up
    # (one pitch gap instead of five, so the sheets swing 30 cm instead of 25 and run nearly
    # full size), and pitch is no longer boxed in at 20 mm.
    #
    # 50 mm pitch is measured, and it REVERSES the 18-sheet result that tighter was better at
    # every sheet count. That rule was found where pitch was capped at 20 mm by the footprint
    # solve, so "tighter is better" was only ever observed on one side of the optimum. With
    # room to open up, 6 sheets want 50 mm: mean IoU 0.815 -> 0.830, worst view 0.785 -> 0.816,
    # and the cross-talk cost falls from +0.015 to -0.002. Past that it collapses (0.780 at
    # 100 mm) as the outer sheets swing far enough off-axis to throw their stray shadows
    # somewhere else entirely.
    #
    # The engrave alphabet flips too, and harder. At 18 sheets the dark-biased set won and
    # (0.78,0.60,0.42) was in as a deliberately weak control; at 6 sheets that control comes
    # FIRST and dark-biased falls to sixth of seven. The probe explains why: it over-covered
    # every view (front fg 0.43 against a target 0.36) because with two sheets per family
    # instead of six there are far fewer transmittances to multiply, so a dark alphabet has no
    # light one to walk back with and the piece just goes muddy. Lighter levels restore the
    # tonal ladder: SSIM 0.661 -> 0.697 and RMSE 0.1862 -> 0.1642.
    #
    # (0.85,0.62,0.35) over the marginally better-scoring (0.78,0.60,0.42): they are inside
    # seed noise of each other (0.765 vs 0.769, sigma 0.002) and this one has the higher mean
    # and worst-view IoU, but the deciding argument is fabrication -- its levels are 23 and 27
    # points apart instead of 18, so the engraver has room to hold them apart. intensity_gain
    # swept flat over 0.8-1.0; left at 1.0.
    #
    # Shard granularity was flat at 18 sheets and is nearly flat here too -- every combination
    # lands inside 0.005 of every other, against a seed sigma of 0.002. It goes fine anyway:
    # the brief lifted the minimum shard size, edge fidelity is the one metric that separates
    # the runs (0.513 against 0.489-0.498 for everything coarser), and with six sheets carrying
    # what eighteen used to, resolution is the only currency left. 1168 shards, ~195 per sheet.
    "30v6": BuildConfig(name="30x30, 6 sheets (2 per family)",
                        body=0.30, footprint_limit=0.30, damage_weight=0.25,
                        n_per_family=2, pitch=0.05, engrave_levels=(0.85, 0.62, 0.35),
                        fragment_size=0.05, shard_budget=400),
}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="60")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-res", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = ARMS[a.arm]
    cfg.seed = a.seed
    if a.wall_res:
        cfg.wall_res = cfg.panel_res = a.wall_res
    out = Path(a.out or f"out_pearl3/baseline_{a.arm}")
    run_build(cfg, out)


if __name__ == "__main__":
    main()
