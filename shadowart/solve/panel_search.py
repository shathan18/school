"""Greedy incremental panel-layout construction (panel count/angle/position as free
search variables, replacing the old fixed 8-panel family layout).

Scoring: COVERAGE-AWARE marginal set-cover of the wall image content, not the earlier
`sqrt(areaA*areaB)` resolvable-detail score. That earlier score was measured to fail at
the rescaled installation: it rewarded a panel's total resolvable *area* regardless of
*where on the wall* that area landed, so the search happily picked panels whose shadows
fell entirely off the image subject -- 6-10% subject coverage, ~5% shard survival, walls
reconstructing as mostly-empty. See `build_panels_greedy` for the replacement; the key
measured facts that shaped it:

  - The two walls are lit from different directions (Light A at y=0.7, Light B at
    x=0.7), so a panel covering Wall A's centre must sit at LOW y (near Wall B) while a
    Wall-B-covering panel sits at LOW x -- a single compact anchor box near the far
    corner geometrically cannot cover both wall centres (confirmed: compact box tops out
    at ~6% subject coverage). The anchor range must stay WIDE; a geometric standoff
    (below) is what keeps panels off the glass, not a tight box.
  - With a wide range, a single panel close to a light blankets the whole wall at high
    magnification (93% coverage, one panel) -- degenerate and blurry. A magnification cap
    forces coverage to be assembled from several modest panels instead (the shard cloud
    we actually want).

So placement rests on three coordinated constraints, each validated numerically before
being coded (compact-box 6% / wide-open 1-panel-93% / wide+standoff+magcap ~90% via
several panels): a wide anchor range, a geometric wall standoff, and a magnification cap.
"""
from __future__ import annotations

import math

import numpy as np

from ..config.scene import Panel
from ..geometry import homography as H
from ..geometry.projection import _penumbra_sigma_m
from ..targets import color as _color


def _floor_segment_xy(angle, anchor, u_range):
    ax, ay = anchor
    dx, dy = math.cos(angle), math.sin(angle)
    u0, u1 = u_range
    return (ax + u0 * dx, ay + u0 * dy), (ax + u1 * dx, ay + u1 * dy)


def _sample_angle_deg(rng, angle_deg_range, angle_bands):
    """Sample a panel angle (degrees). `angle_bands`, if given, is a list of (lo,hi)
    degree bands sampled proportional to their width -- lets an angle-DIVERSITY sweep
    request e.g. a bimodal near-axis distribution ([(0,20),(70,90)]) instead of a single
    contiguous range. `angle_bands=None` falls back to the single `angle_deg_range`."""
    if not angle_bands:
        return rng.uniform(*angle_deg_range)
    widths = np.array([hi - lo for lo, hi in angle_bands], float)
    lo, hi = angle_bands[int(rng.choice(len(angle_bands), p=widths / widths.sum()))]
    return rng.uniform(lo, hi)


def _random_candidate(scene, rng, angle_deg_range, u_size_range, v_range, name,
                      anchor_range=(0.5, 2.4), standoff=0.5, mag_cap=3.0, max_tries=80,
                      angle_bands=None):
    """One random candidate panel, rejection-sampled against three constraints (each
    validated numerically before being chosen -- see module docstring):

      1. Room bounds (`config/io.py::_sanity_check`): both floor-segment endpoints
         strictly between each light and that light's wall plane.
      2. Geometric wall standoff: the panel's nearest floor-plan approach to EACH wall
         (Wall A at x=0, Wall B at y=0) is >= `standoff` metres. This is the real
         "don't smear the object against the glass" guarantee -- enforced on the actual
         projected floor segment, not hoped for via a tight anchor range. (A compact
         anchor box was tried and rejected: it can't reach both wall centres, which are
         lit from different directions.)
      3. Magnification cap: the panel's shadow magnification on BOTH walls is <= `mag_cap`.
         Without this, a single panel near a light blankets an entire wall at high
         magnification (measured: 93% coverage from one panel) -- degenerate and blurry.
         Capping it forces wall coverage to be assembled from several modest panels.

    `anchor_range` is deliberately WIDE (both coords), because the coverage-viable zones
    for the two walls sit in different parts of the room; the standoff, not a tight box,
    is what keeps panels off the walls."""
    la = scene.lights["A"].xyz[0]
    lb = scene.lights["B"].xyz[1]
    wallA, wallB = scene.walls["A"], scene.walls["B"]
    lightA, lightB = scene.light_for_wall("A").xyz, scene.light_for_wall("B").xyz
    for _ in range(max_tries):
        angle = math.radians(_sample_angle_deg(rng, angle_deg_range, angle_bands))
        u_size = rng.uniform(*u_size_range)
        u_range = (-u_size / 2, u_size / 2)
        anchor = (rng.uniform(*anchor_range), rng.uniform(*anchor_range))
        (x0, y0), (x1, y1) = _floor_segment_xy(angle, anchor, u_range)
        if not (0.0 <= x0 < la and 0.0 <= x1 < la and 0.0 <= y0 < lb and 0.0 <= y1 < lb):
            continue
        if min(x0, x1) < standoff or min(y0, y1) < standoff:      # standoff from both walls
            continue
        cand = Panel(name=name, angle=angle, anchor=anchor, u_range=u_range,
                     v_range=v_range, thickness=scene.material_thickness)
        magA = H.magnification(cand, lightA, wallA)
        magB = H.magnification(cand, lightB, wallB)
        if magA > mag_cap or magB > mag_cap:
            continue
        return cand
    return None


def _wall_coverage_area(panel, scene, wall_name):
    """Same measure as geometry.projection.wall_coverage_area, computed standalone (no
    full projection table needed -- a candidate isn't in the scene yet)."""
    wall = scene.walls[wall_name]
    light = scene.light_for_wall(wall_name).xyz
    try:
        Hpw = H.homography_panel_to_wall(panel, light, wall)
    except ValueError:
        return 0.0                                        # degenerate projection (e.g. edge-on)
    pts = H.apply_homography(Hpw, panel.corners_uv())
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _resolvable_detail(panel, scene, wall_name):
    """Coverage area discounted by penumbra blur (squared, since blur eats area in both
    directions on the wall) -- "how much actually-resolvable detail can this panel
    contribute to this wall", not just raw projected size. See module docstring for why
    this (not raw area) is the right quantity: it stays bounded as a panel approaches
    its light instead of blowing up, because penumbra blows up at almost the same rate
    area does there -- for MODERATE magnification. That approximation breaks down for a
    near-degenerate, highly keystoned projection (a panel sitting almost next to its
    light): raw `wall_coverage_area` (shoelace of the projected quad) can explode far
    faster than the penumbra normalisation compensates for, since keystoning stretches
    area non-uniformly rather than by a single scale factor. Confirmed empirically: two
    panels at ~7-8x magnification scored area 25,952 m^2 / 42,534 m^2 against a 1.44 m^2
    wall and still won the greedy search, dominating (and wrecking) that wall's
    reconstruction. A panel physically cannot contribute more resolvable information to
    a wall than the wall has surface for, so cap raw area at the wall's own physical
    area before the penumbra discount -- bounds the pathological case without changing
    the score for any panel in the normal (non-degenerate) regime."""
    area = _wall_coverage_area(panel, scene, wall_name)
    if area <= 0.0:
        return 0.0
    wall = scene.walls[wall_name]
    area = min(area, wall.width * wall.height)
    light = scene.light_for_wall(wall_name).xyz
    sigma = _penumbra_sigma_m(panel, light, wall, scene.source_radius)
    return area / max(sigma, 1e-4) ** 2


def _candidate_score(panel, scene):
    """LEGACY joint-detail score, kept as a fallback for `build_panels_greedy` when no
    targets are supplied (so callers that don't care about coverage still work). Measured
    to fail at the rescaled install (rewards resolvable area regardless of where it lands
    on the wall) -- the coverage-aware path below supersedes it. See module docstring."""
    a = _resolvable_detail(panel, scene, "A")
    b = _resolvable_detail(panel, scene, "B")
    return math.sqrt(a * b)


# ---------------------------------------------------------------------------
# coverage-aware scoring (the real placement driver -- see module docstring)
# ---------------------------------------------------------------------------
def _coverage_context(scene, targets, gsz=96):
    """Precompute, per wall, the coarse (gsz x gsz) wall-pixel centres in wall metres and
    the downsampled image-subject mask. Coarse res keeps footprint scoring cheap (a full
    450x450 back-projection per candidate x K x steps would be ~60M point-transforms);
    coverage FRACTIONS are accurate at coarse res."""
    ctx = {}
    for w, wall in scene.walls.items():
        a = (np.arange(gsz) + 0.5) / gsz * wall.width
        b = (np.arange(gsz) + 0.5) / gsz * wall.height
        aa, bb = np.meshgrid(a, b)
        pts = np.stack([aa.ravel(), bb.ravel()], axis=-1)          # [gsz*gsz, 2] wall metres
        subj_full = _color.subject_mask(targets[w], scene.white_threshold)
        Hn, Wn = subj_full.shape
        ri = (np.linspace(0, Hn - 1, gsz)).astype(int)
        ci = (np.linspace(0, Wn - 1, gsz)).astype(int)
        subj = subj_full[np.ix_(ri, ci)].ravel()                   # [gsz*gsz] bool
        ctx[w] = {"pts": pts, "subject": subj, "shape": (gsz, gsz)}
    return ctx


def _footprint_flat(panel, scene, wall_name, ctx):
    """Boolean [gsz*gsz] mask: which coarse wall-pixels this panel's shadow can reach
    (its projected footprint), computed by back-projecting each wall point through the
    panel->wall homography's inverse and testing against the panel's (u,v) extent -- the
    same geometry as `decompose._panel_reachable_mask`, on the precomputed coarse grid."""
    wall = scene.walls[wall_name]
    light = scene.light_for_wall(wall_name).xyz
    try:
        Hpw = H.homography_panel_to_wall(panel, light, wall)
        Hwp = np.linalg.inv(Hpw)
    except (ValueError, np.linalg.LinAlgError):
        return np.zeros(ctx[wall_name]["pts"].shape[0], dtype=bool)
    uv = H.apply_homography(Hwp, ctx[wall_name]["pts"])
    u0, u1 = panel.u_range
    v0, v1 = panel.v_range
    return ((uv[:, 0] >= u0) & (uv[:, 0] <= u1) &
            (uv[:, 1] >= v0) & (uv[:, 1] <= v1))


def _coverage_score(panel, scene, ctx, count, spill_weight=0.0):
    """Marginal coverage value of `panel` given what's already placed, MINUS a penalty for
    shadow that lands outside the image content.

    Gain term (unchanged): for each wall, sum a diminishing-returns weight 1/(1+count[pixel])
    over the panel's footprint intersected with the image subject -- so a panel is rewarded
    for reaching subject pixels that few (or no) placed panels reach, and barely at all for
    piling onto already-well-covered pixels. Walls are combined with a deficit weight (the
    wall with less accumulated coverage mass counts more), which self-balances the
    primary-wall split.

    Spill term (`spill_weight` > 0): subtracts the panel's footprint that falls OUTSIDE each
    wall's subject, normalised the same way as the gain (fraction of that wall's subject
    area), so `spill_weight` is dimensionless. This is the subject-aware bad-cross-talk
    measure -- shadow landing on a wall's empty background -- and it is what the gain term
    alone is blind to: a candidate that covers Wall A's subject beautifully while splattering
    Wall B's background scored identically to one that didn't. Since every panel necessarily
    casts on BOTH walls, penalising outside-subject footprint makes the greedy prefer
    candidates whose cross-wall shadow either lands on the other wall's real content (GOOD
    cross-talk -- deliberately preserved, this is the joint-intersection effect) or is small.
    Deliberately NOT the area-balance `joint_ratio`, which is subject-blind and rates a
    background-splattering panel as perfectly "joint".

    `spill_weight=0.0` reproduces the previous (spill-blind) scorer exactly -- kept as the
    default so it doubles as the control arm when measuring whether the penalty helps.

    Returns (score, footprintA_flat, footprintB_flat) so the caller can update `count` with
    the chosen panel's footprints without recomputing them."""
    fpA = _footprint_flat(panel, scene, "A", ctx)
    fpB = _footprint_flat(panel, scene, "B", ctx)
    subjA, subjB = ctx["A"]["subject"], ctx["B"]["subject"]
    valA = (fpA & subjA) / (1.0 + count["A"])
    valB = (fpB & subjB) / (1.0 + count["B"])
    nA = max(subjA.sum(), 1)
    nB = max(subjB.sum(), 1)
    margA = valA.sum() / nA
    margB = valB.sum() / nB
    massA, massB = float(count["A"].sum()), float(count["B"].sum())
    wA = (massB + 1.0) / (massA + massB + 2.0)                     # under-covered wall weighted up
    wB = 1.0 - wA
    gain = wA * margA + wB * margB
    if spill_weight <= 0.0:
        return gain, fpA, fpB
    spillA = float((fpA & ~subjA).sum()) / nA
    spillB = float((fpB & ~subjB).sum()) / nB
    spill = spillA + spillB
    # MULTIPLICATIVE, bounded in [0, gain] -- never negative. An ADDITIVE penalty
    # (gain - lambda*spill) was tried and is broken by construction: a panel aimed at
    # nothing (footprint entirely off-wall) has gain=0 AND spill=0 -> score 0, while any
    # genuinely useful panel necessarily spills some shadow onto background (the subject is
    # only ~54-64% of each wall) and goes NEGATIVE once lambda*spill > gain. So 0 beats
    # negative and the greedy picks 14 panels that miss the wall entirely -- measured: 0%
    # bad cross-talk on every seed because image-completeness was 0% (blank walls).
    # Here a zero-gain panel scores exactly 0 and therefore cannot beat a real panel.
    # (A clamped-linear form max(0, 1 - lambda*spill_frac) also avoids negatives but
    # collapses most candidates to a 0 tie at high lambda, losing all ranking; the power
    # form degrades smoothly instead.)
    spill_frac = spill / (spill + gain) if (spill + gain) > 0 else 0.0   # in [0,1]
    return gain * (1.0 - spill_frac) ** spill_weight, fpA, fpB


def build_panels_greedy(scene, count, mode="deliberate", K=12,
                        angle_deg_range=(20, 70), u_size_range=(0.25, 0.75),
                        v_range=(0.03, 1.18), anchor_range=(0.5, 2.4),
                        standoff=0.5, mag_cap=3.0, targets=None, seed=0,
                        angle_bands=None, spill_weight=0.0):
    """Greedily construct `count` panels, one at a time.

    mode="random": K=1, accept the next valid random candidate each step (no scoring).
    mode="deliberate": sample K candidates per step, keep the best-scoring one.

    Scoring (deliberate mode): if `targets` is given, use COVERAGE-AWARE marginal
    set-cover of the wall image content (`_coverage_score`) -- the real placement driver
    that fixes the survival/coverage collapse (see module docstring). Without `targets`,
    fall back to the legacy joint-detail score. Returns (panels, scores) where scores[i]
    is the score the i-th panel was chosen with (for count-sweep-from-prefixes / diags)."""
    rng = np.random.default_rng(seed)
    k = 1 if mode == "random" else K
    use_coverage = (targets is not None) and (mode != "random")
    ctx = _coverage_context(scene, targets) if use_coverage else None
    count_masks = ({"A": np.zeros(ctx["A"]["subject"].shape, np.float32),
                    "B": np.zeros(ctx["B"]["subject"].shape, np.float32)}
                   if use_coverage else None)
    panels, scores = [], []
    for i in range(count):
        # -inf, not -1.0: with a spill penalty (`spill_weight`>0) a candidate's score can be
        # legitimately negative (spill outweighs marginal gain). A -1.0 floor would silently
        # reject every such candidate and truncate the layout to fewer panels than requested.
        best, best_score, best_fp = None, float("-inf"), None
        for _ in range(k):
            cand = _random_candidate(scene, rng, angle_deg_range, u_size_range, v_range,
                                     f"S{i}", anchor_range=anchor_range,
                                     standoff=standoff, mag_cap=mag_cap, angle_bands=angle_bands)
            if cand is None:
                continue
            if use_coverage:
                s, fpA, fpB = _coverage_score(cand, scene, ctx, count_masks, spill_weight)
                if s > best_score:
                    best, best_score, best_fp = cand, s, (fpA, fpB)
            else:
                s = _candidate_score(cand, scene)
                if s > best_score:
                    best, best_score = cand, s
        if best is None:
            break                                          # couldn't find any valid candidate
        panels.append(best)
        scores.append(best_score)
        if use_coverage:                                   # accumulate chosen panel's footprints
            count_masks["A"] += best_fp[0]
            count_masks["B"] += best_fp[1]
    return panels, scores


def joint_intersection_pct(fragments, table, panels, thresholds=(0.1, 0.2, 0.3)):
    """Stage-1-plan section 4 metric: fraction of total shard PHYSICAL area whose
    contribution to both walls is within `threshold` of each other (i.e. neither wall's
    contribution is negligible next to the other's), reported at each of `thresholds`
    rather than a single picked cutoff.

    Uses each shard's host panel's PROJECTED COVERAGE AREA to each wall
    (`geometry.projection.wall_coverage_area`, foreshortening-aware) scaled by the
    shard's own fraction of that panel's physical area, as a proxy for "how big is this
    shard's shadow on that wall". An earlier version used magnification^2 instead --
    confirmed wrong, not just a rounding difference: magnification is a single-point
    scale factor and doesn't capture foreshortening, so it understated how much more
    relevant an axis-aligned panel is to its own wall than the other (measured on the
    baseline layout: area ratio ~0.03-0.32 per panel vs. magnification ratio ~0.69-0.97
    for the *same* panels -- magnification made near-edge-on panels look almost as
    relevant to the wall they're edge-on to as to their own, which is exactly the
    foreshortening effect wall_coverage_area exists to capture).
    """
    from ..geometry.projection import wall_coverage_area
    panels_by_name = {p.name: p for p in panels}
    ratio_cache = {}

    def area_ratio(panel_name, wall_name):
        key = (panel_name, wall_name)
        if key not in ratio_cache:
            p = panels_by_name[panel_name]
            phys_area = max(p.u_size * p.v_size, 1e-9)
            ratio_cache[key] = wall_coverage_area(table, p, wall_name) / phys_area
        return ratio_cache[key]

    total_area = 0.0
    per_thresh_area = {t: 0.0 for t in thresholds}
    for f in fragments:
        area = f["phys_mm2"]
        total_area += area
        dA = area * area_ratio(f["panel"], "A")
        dB = area * area_ratio(f["panel"], "B")
        ratio = min(dA, dB) / max(dA, dB, 1e-9)
        for t in thresholds:
            if ratio > t:
                per_thresh_area[t] += area
    if total_area <= 0:
        return {t: 0.0 for t in thresholds}
    return {t: 100.0 * per_thresh_area[t] / total_area for t in thresholds}


def run_layout(scene, panels, targets, names, seed=0):
    """Run the existing, unmodified overlap pipeline (projection table -> render ->
    shard generation -> fidelity metrics) against a specific panel layout, and add the
    joint-intersection metric on top. Reuses `solve.decompose.fragment_shards_overlap`
    as-is -- it already works with any panel set via `primary_wall_of`, no changes
    needed there for this to work.
    """
    import dataclasses
    from ..geometry.projection import build_projection_table
    from ..forward.renderer import Renderer
    from ..targets import color as C
    from . import decompose
    from .. import metrics as _metrics

    test_scene = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(test_scene)
    renderer = Renderer(test_scene, table)
    stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity = \
        decompose.fragment_shards_overlap(test_scene, table, targets, names=names,
                                          white_thr=test_scene.white_threshold,
                                          max_stack=test_scene.color_max_stack, seed=seed)
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    pred_rgb = renderer.render_color_np(panel_T)
    return {
        "accuracy": _metrics.evaluate_wall_accuracy(targets, pred_rgb),
        "joint_intersection_pct": joint_intersection_pct(fragments, table, panels),
        "shard_regions": {w: budget_stats.get(w, {}).get("achieved", 0) for w in scene.walls},
        "laminated_layers": len(fragments),
        "panel_count": len(panels),
        "collisions_resolved": resolved,
    }
