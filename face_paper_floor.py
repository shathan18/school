"""THE FIX: give the paper a GRAY_L tint so it becomes material that can intersect.

DIAGNOSIS RECAP (`face_crosstalk_anatomy.py`). Cross-talk scales with ink area, steeply:

    pair                     ink(subj%)   cross-talk on wall   good%
    Poe / Dostoevsky            44%              2.1%           0.34
    Mallarme / Wagner           69%              9.0%           0.43
    Mona / Pearl                85%             19.1%          11.39
    Hokusai control (cmyk)     ~full           (good 11.5 + bad 15.2 = 26.7% of subject)

The two-tone woodcuts win on face legibility for exactly one reason -- white = clear = NO
SHARD, so all 300 shards land on the 44% that is ink -- and that is the same reason they
cannot intersect. There is no material in the other 56% of the piece to be crossed. The
property that made them survive is the property that makes them a bad sculpture. Those are
not two findings, they are one finding with two signs.

THE FIX. Stop spending the paper on nothing. Scale the target so the light tone sits at
GRAY_L (~0.75 transmittance, i.e. below the 0.90 white threshold) instead of paper-white:

    target_out = target_gray * floor          floor=1.0 -> clear paper (the old behaviour)
                                              floor=0.75 -> GRAY_L paper (all material)

This is a pure global scaling of the light tone, so the ink geometry -- which is what
carries the identity in a flat-mass woodcut -- is untouched. What changes is that the light
regions now demand shards, so the whole wall is perspex and there is something everywhere
for the other image to cross.

WHY GRAYSCALE HELPS HERE (correcting my earlier caveat). I flagged that grayscale discards
hue and therefore forfeits the colour-compatibility result. For the DOUBLE-DUTY question
that flag was backwards. Under the `noir` palette both walls want the SAME four neutral
greys, so a light-grey shard is a tonally correct answer on either wall -- this is the
maximally palette-compatible case, better than any colour pair can be. The caveat still
stands for the separate claim that colour pairs can be chosen for compatibility; it does
not stand for whether these pieces can intersect.

THE COST, STATED UP FRONT. Tinting the paper raises ink area from 0.44 to ~1.0, so the same
300 shards now cover the whole wall instead of 44% of it -- roughly 2.3x less resolution per
shard. That is precisely what turned the oil paintings to mush. The `face_dense` arm exists
to pay for it: flat paper is cheap to tile with a few huge shards (`bg_coarsen`), so the
budget can be concentrated back into the face box (`face_density`). Whether that actually
works is the question this script answers -- it is not assumed.

RANKING. On duty ABSOLUTE (good%), not on the good/bad ratio. The ratio read a healthy 0.72
for a layout with 2.1% cross-talk; a ratio describes the quality of whatever overlap exists
and is blind to there being almost none. Fidelity (face_det, SSIM) is reported alongside so
a layout cannot win by destroying the images.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.solve.search import colour_agreeing_duty
from shadowart.targets import color as C

import face_pretest as FP
import face_render300 as FR
import face_duty_sweep as DS
from face_crosstalk_anatomy import anatomy

OUT = Path("out_faces_hc/duty")
TGT = Path("out_faces_hc/targets_floor")

PAPER_FLOORS = [1.0, 0.85, 0.75]        # 1.0 = clear paper (old), 0.75 = GRAY_L paper
ARMS = ["uniform", "face_dense"]
PANEL_COUNT = 10                        # duty sweep: 10 @ 43-47 deg was the best clean config
ANGLE_RANGE = (43, 47)                  # ~45 deg = equally face-on to both walls


def write_floor_targets(floor: float) -> dict:
    """Two-tone head crops with the light tone scaled to `floor` instead of paper-white."""
    TGT.mkdir(parents=True, exist_ok=True)
    tag = f"f{int(round(floor * 100)):03d}"
    by_key = {}
    for c in FP.CANDIDATES:
        if not Path(c.path).exists():
            continue
        two = FP.posterize_gray(FP.load_head(c), 2) * floor
        Image.fromarray((np.stack([two] * 3, -1) * 255).astype(np.uint8)).save(
            TGT / f"{c.key}_{tag}.png")
        by_key[c.key] = c
    return by_key


def face_mask_floor(target_rgb: np.ndarray, face: tuple, floor: float) -> np.ndarray:
    """Face-box mask for a tinted target.

    `FR.face_mask_wall` locates the head by thresholding at the 0.90 white threshold, which
    on a tinted target matches EVERY pixel (the paper is now 0.75) and would return the whole
    frame. Tinting is a pure global scale, so the ink geometry is unchanged -- threshold at
    half the paper level to recover the same ink bounding box."""
    img = np.flipud(target_rgb)
    ink = img.mean(-1) < 0.5 * floor
    ys, xs = np.where(ink)
    if not len(ys):
        return np.zeros(target_rgb.shape[:2], dtype=bool)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = r1 - r0 + 1, c1 - c0 + 1
    l, t, r, b = face
    m = np.zeros(target_rgb.shape[:2], dtype=bool)
    m[int(r0 + t * h):int(r0 + b * h), int(c0 + l * w):int(c0 + r * w)] = True
    return np.flipud(m)


def build_floor(pair, seed, floor, arm, cands, panel_count=PANEL_COUNT,
                angle_range=ANGLE_RANGE, density=1.0):
    """`density` < 1 shrinks the shards, which is the ONLY way to actually raise the count.

    `shard_budget` is a fabrication CEILING: `_autotune_spacing` coarsens when the natural
    count overshoots and otherwise leaves it alone, so raising the budget from 300 to 2750
    changes nothing (measured: 343 shards at every budget). The natural count is set by
    `solve.fragment_size`, with `fragment_min_area` as a hard floor that DROPS shards below
    it -- so both must move together, areas as the square of the linear size."""
    label, ka, kb = pair
    tag = f"f{int(round(floor * 100)):03d}"
    scene = load_scene(FR.SCENE)
    scene = dataclasses.replace(scene, color_palette=C.PALETTES["noir"])
    if density != 1.0:
        sp = scene.solve
        scene = dataclasses.replace(scene, solve=dataclasses.replace(
            sp, fragment_size=sp.fragment_size * density,
            fragment_min_area=sp.fragment_min_area * density ** 2,
            fragment_max_area=sp.fragment_max_area * density ** 2))
    names = C.palette_names(scene.color_palette)
    wr = scene.solve.wall_res
    targets = {
        "A": C.load_color_target(str(TGT / f"{ka}_{tag}.png"), wr, white_thr=scene.white_threshold),
        "B": C.load_color_target(str(TGT / f"{kb}_{tag}.png"), wr, white_thr=scene.white_threshold),
    }
    panels, _ = build_panels_greedy(scene, count=panel_count, mode="deliberate",
                                    K=FR.K_CANDIDATES, targets=targets, seed=seed,
                                    angle_deg_range=angle_range)
    layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(layout)
    renderer = Renderer(layout, table)

    fmask = {w: face_mask_floor(targets[w], f, floor)
             for w, f in (("A", cands[ka].face), ("B", cands[kb].face))}
    _ = label
    extra = {}
    if arm == "face_dense":
        # flat paper is cheap to tile with a few huge shards, so the budget can be pushed
        # back into the face box -- this is what is meant to pay for the tint's resolution cost
        extra = dict(face_masks=fmask, face_density=4.0, bg_coarsen=2.2)

    sc, opacity, fragments, resolved, sd, bs, si = decompose.fragment_shards_overlap(
        layout, table, targets, names=names, white_thr=layout.white_threshold,
        max_stack=layout.color_max_stack, seed=seed,
        shard_budget=FR.SHARD_BUDGET_PER_WALL,
        damage_weight=FR.DAMAGE_WEIGHT, credit_weight=FR.CREDIT_WEIGHT,
        match_metric=FR.MATCH_METRIC, match_tol=FR.MATCH_TOL, **extra)
    panel_T = C.stack_transmit_lut(names, sc, si)
    b = dict(layout=layout, table=table, renderer=renderer, panels=panels,
             targets=targets, panel_T=panel_T, names=names, fragments=fragments,
             opacity=opacity, stack_colorid=sc, budget=bs, _fmask=fmask)
    return b


def evaluate(b, pair, seed, floor, arm, density=1.0):
    base, rows = DS.ablate(b)
    prim = {p.name: primary_wall_of(b["layout"], b["table"], p) for p in b["panels"]}
    good, bad = colour_agreeing_duty(
        b["renderer"], b["panel_T"], b["panels"], b["targets"], b["layout"].white_threshold,
        prim=prim, match_tol=FR.MATCH_TOL_REPORT, match_metric="lab")
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)
    fd = {w: FR.detail_retention_wall(b["targets"][w], base[w], b["_fmask"][w])
          for w in ("A", "B")}
    an = anatomy(b)
    gm, bm = 0.5 * (good["A"] + good["B"]), 0.5 * (bad["A"] + bad["B"])
    n_a = int(b["budget"].get("A", {}).get("achieved", 0))
    n_b = int(b["budget"].get("B", {}).get("achieved", 0))
    return dict(
        label=pair[0], seed=seed, floor=floor, arm=arm, density=density,
        n_both=sum(r["serves_A"] and r["serves_B"] for r in rows),
        n_dead=sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows),
        n_panels=len(rows),
        subj_pct=0.5 * (an["A"]["subj_pct"] + an["B"]["subj_pct"]),
        xt_wall_pct=0.5 * (an["A"]["xt_wall_pct"] + an["B"]["xt_wall_pct"]),
        good_mean=gm, bad_mean=bm, good_bad=(gm / bm if bm else None),
        ssim=0.5 * (acc["A"]["ssim"] + acc["B"]["ssim"]),
        face_det=0.5 * (fd["A"] + fd["B"]),
        shards=n_a + n_b,
        over_budget=bool(max(n_a, n_b) > FR.SHARD_BUDGET_PER_WALL * 1.05),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 108)
    print("PAPER-FLOOR EXPERIMENT -- can tinting the paper GRAY_L buy a real intersection?")
    print(f"  {PANEL_COUNT} panels @ {ANGLE_RANGE[0]}-{ANGLE_RANGE[1]} deg, "
          f"{FR.SHARD_BUDGET_PER_WALL}/wall budget, seed 2")
    print("=" * 108)
    print(f"{'pair':26s} {'floor':>5s} {'arm':>10s} | {'ink%':>5s} {'xt%':>5s} "
          f"{'good%':>6s} {'bad%':>6s} {'g/b':>5s} | {'both':>4s} {'dead':>4s} | "
          f"{'SSIM':>5s} {'face':>5s} {'shards':>6s}")
    print("-" * 108)
    rows = []
    for floor in PAPER_FLOORS:
        cands = write_floor_targets(floor)
        for pair, arm in itertools.product(FR.PAIRS[:1], ARMS):
            b = build_floor(pair, 2, floor, arm, cands)
            m = evaluate(b, pair, 2, floor, arm)
            rows.append(m)
            ob = "!" if m["over_budget"] else " "
            print(f"{m['label'][:26]:26s} {floor:>5.2f} {arm:>10s} | {m['subj_pct']:>5.1f} "
                  f"{m['xt_wall_pct']:>5.1f} {m['good_mean']:>6.2f} {m['bad_mean']:>6.2f} "
                  f"{(m['good_bad'] or 0):>5.2f} | {m['n_both']:>4d} {m['n_dead']:>4d} | "
                  f"{m['ssim']:>5.3f} {m['face_det']:>5.3f} {m['shards']:>6d}{ob}")
    (OUT / "paper_floor.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/paper_floor.json")


if __name__ == "__main__":
    main()
