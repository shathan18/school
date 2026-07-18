"""
OLD vs NEW run-selection score -- does the colour-aware fix change what actually gets built?

  OLD:  ssim + 0.5*edge - 0.25*joint_intersection_pct[0.3]
        `joint_intersection_pct` is a COLOUR-BLIND area-overlap proxy, so this SUBTRACTS double
        duty -- fighting the per-shard signed credit that rewards it (corrections_note.md 3).
  NEW:  ssim + 0.5*edge + 0.5*mean(duty) - 0.5*mean(bleed)
        `duty`/`bleed` split cross-talk into the half that arrives in a WANTED colour (bonus)
        and the half that arrives wrong (penalty). Both levels now optimise the same thing.

For each pair we solve N seeds once, score every seed under BOTH functions, and keep the bundle
each function would pick. If they pick the same seed the change is inert for that pair -- which
is a real result and is reported as such.

Run:  py out_thickness_test/score_compare.py [n_seeds]
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy, joint_intersection_pct
from shadowart.targets import color as C
from shadowart import metrics as _metrics

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MATCH_TOL = 0.30
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)
PAIRS = [
    ("cut sunflowers x oranges", "examples/sf_surface_nobg.png", "examples/oranges_nobg.png"),
    ("vase x oranges",           "examples/sunflowers_clean_nobg.png", "examples/oranges_nobg.png"),
    ("vase x cut sunflowers",    "examples/sunflowers_clean_nobg.png", "examples/sf_surface_nobg.png"),
]
scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)


def one_seed(targets, seed):
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=targets,
                                    seed=seed, angle_deg_range=(5, 85),
                                    anchor_range=SP.search_anchor_range, standoff=SP.search_standoff,
                                    mag_cap=SP.search_mag_cap, u_size_range=SP.search_u_size_range,
                                    v_range=SP.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=NAMES, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed,
        damage_weight=0.5, credit_weight=0.5, match_tol=MATCH_TOL)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    jp = joint_intersection_pct(fr, table, ts.panels)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pT, ts.panels, targets,
                                              ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    old = search.score_layout(acc, jp)                                   # legacy path
    new = search.score_layout(acc, jp, duty=duty, bleed=bleed)           # colour-aware path
    return dict(seed=seed, pred=pred, acc=acc, duty=duty, bleed=bleed, old=old, new=new,
                clarity=acc["A"]["ssim"] + acc["B"]["ssim"])


results = {}
for label, pa, pb in PAIRS:
    targets = {"A": C.load_color_target(pa, WR, white_thr=scene0.white_threshold),
               "B": C.load_color_target(pb, WR, white_thr=scene0.white_threshold)}
    best_old = best_new = None
    print(f"\n{label}  ({N} seeds)")
    print(f"{'seed':>4} {'clarity':>8} {'duty A/B':>14} {'bleed A/B':>14} {'OLD':>8} {'NEW':>8}")
    print("-" * 62)
    for s in range(1, N + 1):
        r = one_seed(targets, s)
        print(f"{s:>4} {r['clarity']:8.3f} {r['duty']['A']:6.1f}/{r['duty']['B']:6.1f} "
              f"{r['bleed']['A']:6.1f}/{r['bleed']['B']:6.1f} {r['old']:8.4f} {r['new']:8.4f}")
        if best_old is None or r["old"] > best_old["old"]:
            best_old = r
        if best_new is None or r["new"] > best_new["new"]:
            best_new = r
    results[label] = dict(targets=targets, old=best_old, new=best_new)
    same = best_old["seed"] == best_new["seed"]
    print(f"  OLD picks seed {best_old['seed']} (clarity {best_old['clarity']:.3f}) | "
          f"NEW picks seed {best_new['seed']} (clarity {best_new['clarity']:.3f})"
          f"{'   <-- SAME seed: change is inert for this pair' if same else ''}")

# ---------------- visual comparison ----------------
rows = len(results)
fig, ax = plt.subplots(rows, 4, figsize=(15, 3.9 * rows))
if rows == 1:
    ax = ax[None, :]
for i, (label, r) in enumerate(results.items()):
    for ci, (which, wall) in enumerate([("old", "A"), ("old", "B"), ("new", "A"), ("new", "B")]):
        b = r[which]
        ax[i, ci].imshow(np.clip(b["pred"][wall], 0, 1), origin="lower", aspect="auto")
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
        if i == 0:
            ax[i, ci].set_title(f"{which.upper()} score — Wall {wall}", fontsize=11,
                                fontweight="bold", color=("#b03030" if which == "old" else "#207040"))
    same = r["old"]["seed"] == r["new"]["seed"]
    ax[i, 0].set_ylabel(f"{label}\nOLD seed {r['old']['seed']} | NEW seed {r['new']['seed']}"
                        + ("\n(identical)" if same else ""), fontsize=9)
plt.suptitle("Run-selection score: OLD (penalises colour-blind overlap) vs NEW (rewards genuine "
             "colour-agreeing double duty)", fontsize=13, y=0.999)
plt.tight_layout(); plt.savefig(f"{OUT}/score_compare.png", dpi=105, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/score_compare.png")

summary = {lab: {"old_seed": r["old"]["seed"], "new_seed": r["new"]["seed"],
                 "old_clarity": r["old"]["clarity"], "new_clarity": r["new"]["clarity"],
                 "old_duty": r["old"]["duty"], "new_duty": r["new"]["duty"],
                 "old_bleed": r["old"]["bleed"], "new_bleed": r["new"]["bleed"]}
           for lab, r in results.items()}
with open(f"{OUT}/score_compare.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {OUT}/score_compare.json\n")
print(f"{'pair':26s} {'OLD seed':>9} {'NEW seed':>9} {'OLD clarity':>12} {'NEW clarity':>12}")
print("-" * 74)
for lab, s in summary.items():
    print(f"{lab:26s} {s['old_seed']:>9} {s['new_seed']:>9} "
          f"{s['old_clarity']:12.3f} {s['new_clarity']:12.3f}")
