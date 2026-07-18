"""
Region-constrained fragmentation for the Sunflowers / lemon-bowl pair: fragment the LEMONS and
the DISH separately (own Voronoi each) so no shard crosses the lemon<->dish boundary and their
outlines stay accurate. Uses the new `region_masks` path in fragment_shards_overlap. Wall A
(sunflowers) is fragmented normally; only Wall B (lemons) is region-split.

Writes a region-segmentation preview, walls.png, per-wall PNGs, scene.html, metrics.json to
out_thickness_test/sunflowers_lemons_regions/.
Run:  py out_thickness_test/lemons_regions.py
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.preview.interactive3d import build_interactive

A_IMG = sys.argv[2] if len(sys.argv) > 2 else "examples/sunflowers_clean_nobg.png"
B_IMG = sys.argv[3] if len(sys.argv) > 3 else "examples/lemons_clean_nobg.png"
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
# Segment the lemon/dish REGIONS from this image (default: the Wall-B image itself). Pass the
# ORIGINAL lemons here when Wall B is a palette-harmonised copy: harmonisation warms the pale
# dish until it reads as "lemon" and the split collapses, but the geometry is identical so the
# mask from the original transfers exactly.
SEG_IMG = sys.argv[5] if len(sys.argv) > 5 else B_IMG
# damage_weight (argv[1]): how hard cross-talk-aware placement steers each shard's stray shadow
# off the OTHER wall's subject. Higher = less orange-sunflower noise on the yellow lemons.
DW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
OUT = f"out_thickness_test/sunflowers_lemons{TAG}_dw{DW:g}"; os.makedirs(OUT, exist_ok=True)
MATCH_TOL = 0.30
scene = load_scene("scenes/tabletop60.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
           "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
SP = dataclasses.replace(scene.solve, diagonal_frac=0.0)


def segment_lemons_dish(rgb):
    """Split the isolated lemon-bowl into region 1 = LEMONS (warm, saturated yellow-green fruit)
    and region 2 = DISH (the pale plate + everything else in the subject). Colour-based:
    lemons are yellow (R,G high, B low) and saturated; the plate is pale/low-sat or bluish."""
    subj = C.subject_mask(rgb, scene.white_threshold)
    mx = rgb.max(-1); mn = rgb.min(-1); sat = mx - mn
    yellowness = (rgb[..., 0] + rgb[..., 1]) * 0.5 - rgb[..., 2]
    lemon = subj & (yellowness > 0.12) & (sat > 0.12)
    lemon = ndimage.binary_opening(lemon, iterations=2)
    lemon = ndimage.binary_closing(lemon, iterations=4)
    lemon = ndimage.binary_fill_holes(lemon)
    labels = np.zeros(rgb.shape[:2], int)
    labels[subj] = 2                      # dish / remainder of the subject
    labels[lemon] = 1                     # lemons
    return labels


seg_src = (targets["B"] if SEG_IMG == B_IMG
           else C.load_color_target(SEG_IMG, wr, white_thr=scene.white_threshold))
labelsB = segment_lemons_dish(seg_src)
nl = int((labelsB == 1).sum()); nd = int((labelsB == 2).sum())
print(f"lemon region: {nl} px | dish region: {nd} px | lemon frac of subject "
      f"{nl / max(nl + nd, 1) * 100:.0f}%")

# region preview: lemons red-tint, dish blue-tint over the source
prev = np.clip(targets["B"], 0, 1).copy()
ov = prev.copy()
ov[labelsB == 1] = 0.5 * ov[labelsB == 1] + 0.5 * np.array([1.0, 0.2, 0.2])
ov[labelsB == 2] = 0.5 * ov[labelsB == 2] + 0.5 * np.array([0.2, 0.4, 1.0])
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(prev, origin="lower"); ax[0].set_title("lemon bowl (source)")
ax[1].imshow(ov, origin="lower"); ax[1].set_title("regions: lemons (red) / dish (blue)")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/region_preview.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/region_preview.png")


def run(seed):
    # Searched panels (measured better than the scene's fixed axis-aligned layout here: the
    # axis-aligned version concentrated cross-talk into one coherent, very visible sunflower
    # ghost on Wall B, while the searched layout spreads it thinner). Cross-talk itself is
    # geometric and unavoidable -- every panel sits in BOTH lights' paths -- so it is fixed by
    # making the two palettes agree (see harmonise_palette.py), not by placement.
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16, targets=targets,
                                    seed=seed, angle_deg_range=(5, 85),
                                    anchor_range=SP.search_anchor_range, standoff=SP.search_standoff,
                                    mag_cap=SP.search_mag_cap, u_size_range=SP.search_u_size_range,
                                    v_range=SP.search_v_range)
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=names, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=seed, damage_weight=DW, credit_weight=0.5, match_tol=MATCH_TOL,
        region_masks={"B": labelsB})          # <-- lemons/dish fragmented separately
    pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    return dict(seed=seed, pred=pred, acc=acc, ts=ts, table=table, op=op, sc=sc, si=si,
                panels=panels, fr=fr, used=len({f["panel"] for f in fr}),
                shards=bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0))


SEEDS = [1, 2, 3]
print(f"\n{'seed':>4} {'shards':>7} | {'A ssim/edge':>14} | {'B ssim/edge':>14}")
print("-" * 48)
runs = []
for s in SEEDS:
    r = run(s); runs.append(r); a, b = r["acc"]["A"], r["acc"]["B"]
    print(f"{s:>4} {r['shards']:>7} | {a['ssim']:.3f}/{a['edge_fidelity']:.3f}   | "
          f"{b['ssim']:.3f}/{b['edge_fidelity']:.3f}")
best = min(runs, key=lambda r: 0.5 * (r["acc"]["A"]["rmse"] + r["acc"]["B"]["rmse"]))
print(f"best seed by mean RMSE = {best['seed']}")

r = best; pred = r["pred"]
fig, ax = plt.subplots(2, 2, figsize=(10, 12))
for ri, w in enumerate(("A", "B")):
    ax[ri, 0].imshow(np.clip(targets[w], 0, 1), origin="lower", aspect="auto")
    ax[ri, 1].imshow(np.clip(pred[w], 0, 1), origin="lower", aspect="auto")
    for ci in (0, 1): ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
    ax[ri, 0].set_ylabel(["Wall A  (sunflowers)", "Wall B  (lemons, region-split)"][ri], fontsize=12)
ax[0, 0].set_title("SOURCE", fontweight="bold"); ax[0, 1].set_title("RECONSTRUCTED SHADOW", fontweight="bold")
plt.suptitle(f"Sunflowers / lemons — region-split, damage_weight={DW:g}, seed {r['seed']} "
             f"({r['shards']} shards)", fontsize=13, y=0.997)
plt.tight_layout(); plt.savefig(f"{OUT}/walls.png", dpi=110, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/walls.png")

def save(arr, path, scale=3):
    a = np.clip(np.flipud(arr), 0, 1); im = Image.fromarray((a * 255).astype(np.uint8))
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)
for w in ("A", "B"):
    save(targets[w], f"{OUT}/src{w}.png"); save(pred[w], f"{OUT}/recon{w}.png")

ts, table = r["ts"], r["table"]
sp_pieces = decompose.panel_stack_pieces(ts, r["sc"], names)
pc = {id(p): ch for items in sp_pieces.values() for p, ch, _s in items}
flat = {n: [p for p, _c, _s in items] for n, items in sp_pieces.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
html, _ = build_interactive(ts, table, r["op"], None, f"{OUT}/scene.html",
                            rays=40, auto_open=False, wall_rgb=pred, pieces=flat, color_of=col, embed_plotly=True)
print(f"wrote {html}")
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_seed": r["seed"], "shards": r["shards"],
               "A": r["acc"]["A"], "B": r["acc"]["B"],
               "lemon_px": nl, "dish_px": nd}, f, indent=2)
print(f"wrote {OUT}/metrics.json")
