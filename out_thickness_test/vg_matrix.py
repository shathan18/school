"""
OBJECT x PORTRAIT matrix (user's pairing rule: an object image is paired with a Van Gogh
self-portrait). 3 objects x 4 portraits = 12 pairs, identical protocol throughout so the
numbers are directly comparable to every earlier pair this session.

Wall A = the object, Wall B = the portrait.

Prediction on record before running (from this session's central finding -- cross-talk is
geometric and unavoidable, so what matters is PALETTE agreement): the warm/golden portraits
should pair best with the yellow objects, and the BLUE-coat portrait should be clearly worst,
the same way Irises was the worst Sunflowers partner.

Run:  py out_thickness_test/vg_matrix.py [n_seeds]
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
BLEND = 0.6                      # joint depth+colour ON (free double-duty gain, see blend_summary)
MATCH_TOL = 0.30
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)

OBJECTS = [("cut sunflowers", "examples/sf_surface_nobg.png"),
           ("oranges",        "examples/oranges_nobg.png"),
           ("sunflower vase", "examples/sunflowers_clean_nobg.png")]
PORTRAITS = [("gold",   "examples/vg_p_gold_nobg.png"),
             ("yellow", "examples/vg_p_yellow_nobg.png"),
             ("dark",   "examples/vg_p_dark_nobg.png"),
             ("blue",   "examples/vg_p_blue_nobg.png")]

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)


def run(targets, seed):
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=targets,
                                    seed=seed, angle_deg_range=(5, 85),
                                    anchor_range=SP.search_anchor_range, standoff=SP.search_standoff,
                                    mag_cap=SP.search_mag_cap, u_size_range=SP.search_u_size_range,
                                    v_range=SP.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=NAMES, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed, damage_weight=0.5, credit_weight=0.5,
        match_tol=MATCH_TOL, colour_blend=BLEND)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pT, ts.panels, targets,
                                              ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    return dict(seed=seed, pred=pred, acc=acc, duty=duty, bleed=bleed,
                clarity=acc["A"]["ssim"] + acc["B"]["ssim"],
                rmse=0.5 * (acc["A"]["rmse"] + acc["B"]["rmse"]))


rows, grid = [], {}
print(f"{N} seeds/pair, colour_blend={BLEND}, Wall A = object, Wall B = portrait\n")
print(f"{'object':16s} {'portrait':9s} {'clarity':>16} {'ssimA/ssimB':>14} {'duty A/B':>13}")
print("-" * 76)
for oname, opath in OBJECTS:
    tA = C.load_color_target(opath, WR, white_thr=scene0.white_threshold)
    for pname, ppath in PORTRAITS:
        targets = {"A": tA, "B": C.load_color_target(ppath, WR, white_thr=scene0.white_threshold)}
        runs = [run(targets, s) for s in range(1, N + 1)]
        cl = np.array([r["clarity"] for r in runs])
        best = min(runs, key=lambda r: r["rmse"])
        rec = dict(obj=oname, por=pname, clarity_mean=float(cl.mean()), clarity_std=float(cl.std()),
                   clarity_best=float(best["clarity"]), best_seed=best["seed"],
                   ssimA=float(best["acc"]["A"]["ssim"]), ssimB=float(best["acc"]["B"]["ssim"]),
                   dutyA=float(np.mean([r["duty"]["A"] for r in runs])),
                   dutyB=float(np.mean([r["duty"]["B"] for r in runs])))
        rows.append(rec); grid[(oname, pname)] = (best, targets)
        print(f"{oname:16s} {pname:9s} {cl.mean():8.3f}+-{cl.std():.3f} "
              f"{rec['ssimA']:6.3f}/{rec['ssimB']:6.3f} {rec['dutyA']:6.1f}/{rec['dutyB']:6.1f}")

rows.sort(key=lambda r: r["clarity_mean"], reverse=True)
print("\n=== RANKED (by mean clarity) ===")
print(f"{'#':>2} {'pair':30s} {'clarity':>9} {'duty A/B':>13}")
print("-" * 60)
for i, r in enumerate(rows, 1):
    print(f"{i:>2} {r['obj'] + ' x ' + r['por']:30s} {r['clarity_mean']:9.3f} "
          f"{r['dutyA']:6.1f}/{r['dutyB']:6.1f}")

# ---- heat-map of clarity over the object x portrait matrix ----
M = np.zeros((len(OBJECTS), len(PORTRAITS)))
for i, (oname, _) in enumerate(OBJECTS):
    for j, (pname, _) in enumerate(PORTRAITS):
        M[i, j] = next(r["clarity_mean"] for r in rows if r["obj"] == oname and r["por"] == pname)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
im = ax.imshow(M, cmap="YlGn")
ax.set_xticks(range(len(PORTRAITS))); ax.set_xticklabels([p[0] for p in PORTRAITS])
ax.set_yticks(range(len(OBJECTS))); ax.set_yticklabels([o[0] for o in OBJECTS])
ax.set_xlabel("Van Gogh self-portrait (Wall B)"); ax.set_ylabel("object (Wall A)")
for i in range(len(OBJECTS)):
    for j in range(len(PORTRAITS)):
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=11)
plt.colorbar(im, label="clarity (SSIM_A + SSIM_B)")
plt.title("Object x Portrait — which pairing reconstructs best")
plt.tight_layout(); plt.savefig(f"{OUT}/vg_matrix_heat.png", dpi=110, bbox_inches="tight"); plt.close()

# ---- visual: top 4 pairings, both walls ----
top = rows[:4]
fig, ax = plt.subplots(len(top), 4, figsize=(15, 3.9 * len(top)))
for i, r in enumerate(top):
    best, targets = grid[(r["obj"], r["por"])]
    cells = [(targets["A"], "A source"), (best["pred"]["A"], "A shadow"),
             (targets["B"], "B source"), (best["pred"]["B"], "B shadow")]
    for ci, (img, tag) in enumerate(cells):
        ax[i, ci].imshow(np.clip(img, 0, 1), origin="lower", aspect="auto")
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
        if i == 0:
            ax[i, ci].set_title(tag, fontsize=11, fontweight="bold")
    ax[i, 0].set_ylabel(f"#{i+1} {r['obj']} x {r['por']}\nclarity {r['clarity_mean']:.3f}", fontsize=9)
plt.suptitle("Van Gogh object x self-portrait — top 4 pairings", fontsize=13, y=0.999)
plt.tight_layout(); plt.savefig(f"{OUT}/vg_matrix_top.png", dpi=105, bbox_inches="tight"); plt.close()

with open(f"{OUT}/vg_matrix.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nsaved {OUT}/vg_matrix_heat.png, {OUT}/vg_matrix_top.png, {OUT}/vg_matrix.json")
