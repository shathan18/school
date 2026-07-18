"""
Run a pair with MODEL-BASED semantic regions: every shard stays inside one real part
(hair / skin / cloth / a single flower / the vase), and parts that carry identity get finer
shards than flat ones.

  py out_thickness_test/semantic_run.py OUT IMG_A IMG_B [n_seeds] [segA] [segB]
      segA/segB: face | scene | classical      (default scene)

Uses semseg.to_regions (SegFormer face-parsing / ADE20K) -> decompose region_masks, and
semseg.part_scales -> region_scales, both already supported by the solver.
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
import objectseg as OS, semseg as SS

OUT, A_IMG, B_IMG = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SEG = {"A": sys.argv[5] if len(sys.argv) > 5 else "scene",
       "B": sys.argv[6] if len(sys.argv) > 6 else "scene"}
BLEND, MATCH_TOL = 0.6, 0.30
# argv[7]: shard ceiling (fabrication budget). argv[8]: multiplier on the per-part density table
# (<1 = finer shards on skin/eyes). Defaults leave prior runs unchanged.
BUDGET = int(sys.argv[7]) if len(sys.argv) > 7 else 0
DENS = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
os.makedirs(OUT, exist_ok=True)

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
if BUDGET:
    scene0 = dataclasses.replace(scene0, overlap_shard_budget=BUDGET)
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)
targets = {"A": C.load_color_target(A_IMG, WR, white_thr=scene0.white_threshold),
           "B": C.load_color_target(B_IMG, WR, white_thr=scene0.white_threshold)}

labels, scales, infos = {}, {}, {}
for w in ("A", "B"):
    subj = C.subject_mask(targets[w], scene0.white_threshold)
    if SEG[w] == "classical":
        labels[w] = OS.segment_objects_edges(targets[w], subj, k=22, target=18)
        infos[w] = {v: "classical" for v in range(1, int(labels[w].max()) + 1)}
    elif SEG[w] == "sam":
        # SAM: class-agnostic INSTANCES -- each flower, each orange, each star its own object.
        # Semantic parsers cannot do this (they return "flower"/"food" as ONE region), which is
        # why this is the right tool for object-y walls. Slow on CPU (~2-5 min/image).
        import sam_seg
        labels[w] = sam_seg.masks_to_labels(sam_seg.sam_masks(targets[w]), subj)
        infos[w] = {v: "sam" for v in range(1, int(labels[w].max()) + 1)}
    else:
        labels[w], infos[w] = SS.to_regions(targets[w], subj, kind=SEG[w])
    scales[w] = {k: v * DENS for k, v in SS.part_scales(infos[w]).items()}
    print(f"Wall {w} [{SEG[w]}]: {int(labels[w].max())} regions  {dict(SS.summarise(infos[w]))}")
    if scales[w]:
        print(f"   density overrides: { {infos[w][k]: v for k, v in scales[w].items()} }")

fig, ax = plt.subplots(2, 2, figsize=(9, 9))
for i, w in enumerate(("A", "B")):
    ax[i, 0].imshow(np.clip(targets[w], 0, 1), origin="lower"); ax[i, 0].set_title(f"Wall {w} source")
    ax[i, 1].imshow(OS.overlay(targets[w], labels[w]), origin="lower")
    ax[i, 1].set_title(f"{SEG[w]} regions ({int(labels[w].max())})")
    for c in (0, 1): ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/regions.png", dpi=100, bbox_inches="tight"); plt.close()


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
                op=op, sc=sc, si=si,
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
cl = np.array([r["clarity"] for r in runs]); sh = np.array([r["shards"] for r in runs])
print(f"\nclarity {cl.mean():.3f} +- {cl.std():.3f}   shards {sh.mean():.0f}")
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
plt.suptitle(f"Semantic regions ({SEG['A']}/{SEG['B']}) — seed {r['seed']}, {r['shards']} shards, "
             f"clarity {r['clarity']:.3f}", fontsize=13, y=0.997)
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
build_interactive(ts, table, r["op"], None, f"{OUT}/scene.html", rays=40, auto_open=False,
                  wall_rgb=pred, pieces=flat, color_of=col, embed_plotly=True)
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_seed": r["seed"], "shards": int(r["shards"]),
               "clarity_mean": float(cl.mean()), "clarity_best": float(r["clarity"]),
               "segA": SEG["A"], "segB": SEG["B"],
               "regionsA": int(labels["A"].max()), "regionsB": int(labels["B"].max()),
               "partsA": dict(SS.summarise(infos["A"])), "partsB": dict(SS.summarise(infos["B"])),
               "A": r["acc"]["A"], "B": r["acc"]["B"], "duty": r["duty"], "bleed": r["bleed"]}, f, indent=2)
print(f"wrote {OUT}/walls.png, regions.png, scene.html, metrics.json")
