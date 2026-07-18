"""
Run a pair with OBJECT-AWARE fragmentation: every perceptual object (each flower, the vase,
the beard, the coat, a star, the river band) is tiled on its OWN Voronoi, so shards follow the
object outlines instead of straddling them. Optionally gives the portrait's FACE a finer shard
spacing so it carries more detail than the coat.

  py out_thickness_test/objects_run.py OUT IMG_A IMG_B [n_seeds] [face_scale] [k]

  face_scale <1 packs smaller shards into the face regions of Wall B (0.5 = ~4x the shard
  density there). 1.0 = uniform.

Writes region previews, walls.png, per-wall PNGs, scene.html and metrics.json.
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.preview.interactive3d import build_interactive
import objectseg as OS

OUT = sys.argv[1]
A_IMG, B_IMG = sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 6
FACE_SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
K = int(sys.argv[6]) if len(sys.argv) > 6 else 10
# argv[7]: smallest object as a fraction of the subject. Lower it to keep SMALL bright features
# (stars, lamp reflections) as their own objects instead of letting them be absorbed into the
# surrounding colour field. argv[8]: extra shard density for objects below `small_frac`.
MIN_FRAC = float(sys.argv[7]) if len(sys.argv) > 7 else 0.004
SMALL_SCALE = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
BLEND, MATCH_TOL = 0.6, 0.30
os.makedirs(OUT, exist_ok=True)

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)
targets = {"A": C.load_color_target(A_IMG, WR, white_thr=scene0.white_threshold),
           "B": C.load_color_target(B_IMG, WR, white_thr=scene0.white_threshold)}

# ---- read each wall as objects -------------------------------------------------------
labels, scales = {}, {}
for w in ("A", "B"):
    subj = C.subject_mask(targets[w], scene0.white_threshold)
    labels[w] = OS.segment_objects(targets[w], subj, k=K, min_frac=MIN_FRAC, max_objects=140)
    scales[w] = {}
    print(f"Wall {w}: {int(labels[w].max())} objects")
    if SMALL_SCALE != 1.0:
        # small bright features (stars, lamp reflections) get their own finer shards, so they
        # survive as points of light instead of dissolving into the surrounding field.
        counts = np.bincount(labels[w].ravel()); counts[0] = 0
        thresh = 0.01 * max(int(subj.sum()), 1)
        small = [v for v in range(1, len(counts)) if 0 < counts[v] < thresh]
        scales[w].update({v: SMALL_SCALE for v in small})
        print(f"  {len(small)} small objects at spacing x{SMALL_SCALE}")
if FACE_SCALE != 1.0:
    faces = OS.face_like_regions(targets["B"], labels["B"])
    scales["B"] = {v: FACE_SCALE for v in faces}
    px = int(np.isin(labels["B"], faces).sum()) if faces else 0
    print(f"  face-like regions on Wall B: {faces}  ({px} px) at spacing x{FACE_SCALE}")

fig, ax = plt.subplots(2, 2, figsize=(9, 9))
for i, w in enumerate(("A", "B")):
    ax[i, 0].imshow(np.clip(targets[w], 0, 1), origin="lower"); ax[i, 0].set_title(f"Wall {w} source")
    ax[i, 1].imshow(OS.overlay(targets[w], labels[w]), origin="lower")
    ax[i, 1].set_title(f"Wall {w} objects ({int(labels[w].max())})")
    for c in (0, 1): ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/objects_preview.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/objects_preview.png")


def run(seed):
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
        match_tol=MATCH_TOL, colour_blend=BLEND,
        region_masks=labels, region_scales=scales)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pT, ts.panels, targets,
                                              ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    return dict(seed=seed, pred=pred, acc=acc, duty=duty, bleed=bleed, ts=ts, table=table,
                op=op, sc=sc, si=si, fr=fr,
                shards=bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0),
                clarity=acc["A"]["ssim"] + acc["B"]["ssim"],
                rmse=0.5 * (acc["A"]["rmse"] + acc["B"]["rmse"]))


print(f"\n{'seed':>4} {'shards':>7} {'clarity':>8} | {'A ssim/edge':>14} | {'B ssim/edge':>14}")
print("-" * 60)
runs = []
for s in range(1, N + 1):
    r = run(s); runs.append(r); a, b = r["acc"]["A"], r["acc"]["B"]
    print(f"{s:>4} {r['shards']:>7} {r['clarity']:8.3f} | {a['ssim']:.3f}/{a['edge_fidelity']:.3f}   "
          f"| {b['ssim']:.3f}/{b['edge_fidelity']:.3f}")
cl = np.array([r["clarity"] for r in runs])
print(f"\nclarity {cl.mean():.3f} +- {cl.std():.3f}")
best = min(runs, key=lambda r: r["rmse"])
print(f"best seed by mean RMSE = {best['seed']} (clarity {best['clarity']:.3f})")

r = best; pred = r["pred"]
fig, ax = plt.subplots(2, 2, figsize=(10, 12))
for ri, w in enumerate(("A", "B")):
    ax[ri, 0].imshow(np.clip(targets[w], 0, 1), origin="lower", aspect="auto")
    ax[ri, 1].imshow(np.clip(pred[w], 0, 1), origin="lower", aspect="auto")
    for ci in (0, 1): ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
    ax[ri, 0].set_ylabel(f"Wall {w}", fontsize=12)
ax[0, 0].set_title("SOURCE", fontweight="bold"); ax[0, 1].set_title("RECONSTRUCTED SHADOW", fontweight="bold")
plt.suptitle(f"Object-aware shards — seed {r['seed']}, {r['shards']} shards, "
             f"face x{FACE_SCALE:g}, clarity {r['clarity']:.3f}", fontsize=13, y=0.997)
plt.tight_layout(); plt.savefig(f"{OUT}/walls.png", dpi=110, bbox_inches="tight"); plt.close()

def save(arr, path, scale=3):
    a = np.clip(np.flipud(arr), 0, 1); im = Image.fromarray((a * 255).astype(np.uint8))
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)
for w in ("A", "B"):
    save(targets[w], f"{OUT}/src{w}.png"); save(pred[w], f"{OUT}/recon{w}.png")

ts, table = r["ts"], r["table"]
sp_pieces = decompose.panel_stack_pieces(ts, r["sc"], NAMES)
pc = {id(p): ch for items in sp_pieces.values() for p, ch, _s in items}
flat = {n: [p for p, _c, _s in items] for n, items in sp_pieces.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
html, _ = build_interactive(ts, table, r["op"], None, f"{OUT}/scene.html", rays=40,
                            auto_open=False, wall_rgb=pred, pieces=flat, color_of=col,
                            embed_plotly=True)
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_seed": r["seed"], "shards": r["shards"], "clarity_mean": float(cl.mean()),
               "clarity_best": float(r["clarity"]), "objectsA": int(labels["A"].max()),
               "objectsB": int(labels["B"].max()), "face_scale": FACE_SCALE,
               "A": r["acc"]["A"], "B": r["acc"]["B"], "duty": r["duty"], "bleed": r["bleed"]}, f, indent=2)
print(f"wrote {OUT}/walls.png, scene.html, metrics.json")
