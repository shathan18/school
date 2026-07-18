"""
Does JOINT (depth + colour) shard design actually pay off?

OFF (colour_blend=0): today's pipeline -- a shard's colour is fixed from its OWN wall before the
host depth-plane is chosen, so it can only help the other wall by luck.
ON  (colour_blend=w): the colour becomes a FUNCTION of the candidate host -- for each panel we
read what the other wall wants where the shard would land and blend toward it, then choose the
(panel, colour) pair together. Voronoi is untouched (measured straddle ~0.95 => shape is not the
limiter). A compromise is rejected if it costs the primary wall more than `colour_primary_tol`.

Identical seeds across arms, so the ONLY variable is the feature. Reports clarity (SSIM_A+SSIM_B)
and colour-agreeing double duty, and renders a visual comparison.

Run:  py out_thickness_test/blend_compare.py [n_seeds] [w1,w2,...]
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BLENDS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.0, 0.3, 0.6]
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


def run(targets, seed, blend):
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
        damage_weight=0.5, credit_weight=0.5, match_tol=MATCH_TOL,
        colour_blend=blend)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pT, ts.panels, targets,
                                              ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    return dict(seed=seed, blend=blend, pred=pred, acc=acc, duty=duty, bleed=bleed,
                clarity=acc["A"]["ssim"] + acc["B"]["ssim"],
                rmse=0.5 * (acc["A"]["rmse"] + acc["B"]["rmse"]))


summary, best_per_arm = {}, {}
for label, pa, pb in PAIRS:
    targets = {"A": C.load_color_target(pa, WR, white_thr=scene0.white_threshold),
               "B": C.load_color_target(pb, WR, white_thr=scene0.white_threshold)}
    summary[label] = {}; best_per_arm[label] = {"targets": targets}
    print(f"\n=== {label}  ({N} seeds x blends {BLENDS}) ===")
    print(f"{'blend':>6} {'clarity':>18} {'duty A/B (mean)':>20} {'bleed A/B (mean)':>20}")
    print("-" * 70)
    for w in BLENDS:
        runs = [run(targets, s, w) for s in range(1, N + 1)]
        cl = np.array([r["clarity"] for r in runs])
        dA = np.mean([r["duty"]["A"] for r in runs]); dB = np.mean([r["duty"]["B"] for r in runs])
        bA = np.mean([r["bleed"]["A"] for r in runs]); bB = np.mean([r["bleed"]["B"] for r in runs])
        best = min(runs, key=lambda r: r["rmse"])          # same selection rule as the deliverable
        best_per_arm[label][w] = best
        summary[label][w] = dict(clarity_mean=float(cl.mean()), clarity_std=float(cl.std()),
                                 clarity_best=float(best["clarity"]), best_seed=best["seed"],
                                 dutyA=float(dA), dutyB=float(dB), bleedA=float(bA), bleedB=float(bB))
        print(f"{w:6.2f} {cl.mean():8.3f}+-{cl.std():.3f} (best {best['clarity']:.3f}) "
              f"{dA:9.1f}/{dB:6.1f} {bA:12.1f}/{bB:6.1f}")

# ---------------- visual: OFF vs each ON arm, best-by-RMSE of each ----------------
on_arms = [w for w in BLENDS if w > 0]
cols = 2 * (1 + len(on_arms))
fig, ax = plt.subplots(len(PAIRS), cols, figsize=(3.6 * cols, 3.9 * len(PAIRS)))
if len(PAIRS) == 1:
    ax = ax[None, :]
for i, (label, _pa, _pb) in enumerate(PAIRS):
    ci = 0
    for w in [0.0] + on_arms:
        b = best_per_arm[label][w]
        for wall in ("A", "B"):
            ax[i, ci].imshow(np.clip(b["pred"][wall], 0, 1), origin="lower", aspect="auto")
            ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
            if i == 0:
                tag = "OFF (today)" if w == 0 else f"blend {w:g}"
                ax[i, ci].set_title(f"{tag} — Wall {wall}", fontsize=10, fontweight="bold",
                                    color=("#b03030" if w == 0 else "#207040"))
            ci += 1
    s0 = summary[label][0.0]
    ax[i, 0].set_ylabel(f"{label}\nOFF clarity {s0['clarity_best']:.3f}", fontsize=9)
plt.suptitle("Joint (depth+colour) shard design: OFF vs ON — same seeds, best-by-RMSE per arm",
             fontsize=13, y=0.999)
plt.tight_layout(); plt.savefig(f"{OUT}/blend_compare.png", dpi=105, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/blend_compare.png")
with open(f"{OUT}/blend_compare.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {OUT}/blend_compare.json\n")

print(f"{'pair':26s} {'blend':>6} {'clarity(best)':>14} {'duty A':>8} {'duty B':>8}")
print("-" * 68)
for label in summary:
    for w in BLENDS:
        s = summary[label][w]
        print(f"{label:26s} {w:6.2f} {s['clarity_best']:14.3f} {s['dutyA']:8.1f} {s['dutyB']:8.1f}")
