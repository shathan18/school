"""
Sunflowers pair, done properly:

1. BETTER CUT for "Four Cut Sunflowers" -- the generic border-connected remover left the blue
   ground in, because that blue is ENCLOSED inside the subject's outline (between/below the
   stems), so it is never reachable from the image border. This painting has a clean colour
   rule instead: flowers are yellow and stems are green (both WARM: (R+G)/2 > B), the ground is
   blue (cool). Cut on warmth, which removes enclosed blue too.

2. REGION-CONSTRAINED VORONOI so each sunflower's OUTLINE is clear: split the flower mass into
   INDIVIDUAL HEADS (distance-transform peaks -> nearest-seed watershed, which separates
   touching round blobs) plus a stems/leaves region, and fragment each on its own Voronoi via
   `region_masks` -- no shard crosses a flower's edge. Wall A (bouquet) is split into flower
   mass vs vase for the same reason.

Run:  py out_thickness_test/sf_regions.py [damage_weight]
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

DW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
A_IMG = "examples/sunflowers_clean_nobg.png"
B_SRC = "examples/sf_cut_src.jpg"
B_IMG = "examples/sf_cut_clean_nobg.png"
OUT = "out_thickness_test/sf_cut_regions"; os.makedirs(OUT, exist_ok=True)
MATCH_TOL = 0.30


# ---------------------------------------------------------------- 1. warmth-based cut
def warm_cut(src, dst, warm_thr=0.04, sat_thr=0.16, hole_px=4000):
    """Cut the flower mass out of "Four Cut Sunflowers" on colour, not on border-connectivity.

    The generic remover failed here because the blue ground is ENCLOSED inside the subject's
    outline (between/below the stems), so a border flood-fill can never reach it. Colour rules
    that do work on this painting:
      * blue ground   -> COOL: (R+G)/2 - B <= warm_thr, so it drops (enclosed blue included).
      * dark foliage  -> warm-ish but DESATURATED: sat <= sat_thr drops it.
      * flowers       -> warm AND saturated, including their dark brown disc centres (an earlier
        luma floor punched holes in exactly those centres, so brightness is NOT used).
    Then keep the largest connected mass and fill only SMALL holes, so real gaps between the
    stems stay open instead of being re-filled with background."""
    im = np.asarray(Image.open(src).convert("RGB"), float) / 255.0
    warmth = (im[..., 0] + im[..., 1]) * 0.5 - im[..., 2]
    sat = im.max(-1) - im.min(-1)
    subj = (warmth > warm_thr) & (sat > sat_thr)
    subj = ndimage.binary_opening(subj, iterations=2)
    subj = ndimage.binary_closing(subj, iterations=6)
    lbl, n = ndimage.label(subj)
    if n:
        counts = np.bincount(lbl.ravel()); counts[0] = 0
        subj = lbl == counts.argmax()                 # the flower mass only
    holes = ndimage.binary_fill_holes(subj) & ~subj
    hl, hn = ndimage.label(holes)
    if hn:
        hc = np.bincount(hl.ravel())
        ids = [i for i in range(1, hn + 1) if hc[i] < hole_px]   # NB: skip label 0 (background)
        if ids:
            subj = subj | np.isin(hl, ids)
    out = np.ones_like(im); out[subj] = im[subj]
    Image.fromarray((out * 255).astype(np.uint8)).save(dst)
    return float(subj.mean())


cov = warm_cut(B_SRC, B_IMG)
print(f"warmth cut: subject {cov*100:.0f}% -> {B_IMG}")

scene = load_scene("scenes/tabletop60.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
           "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
SP = dataclasses.replace(scene.solve, diagonal_frac=0.0)


# ---------------------------------------------------------------- 2. region segmentation
def split_touching(mask, peak_frac=0.10, min_dist_frac=0.03):
    """Separate touching round blobs (sunflower heads) into individual labels: peaks of the
    distance transform become seeds, then every mask pixel takes its NEAREST seed. Same
    nearest-seed idea decompose._voronoi_labels uses, applied to split heads rather than tile."""
    H, W = mask.shape
    dist = ndimage.distance_transform_edt(mask)
    win = max(3, int(peak_frac * min(H, W)) | 1)
    mx = ndimage.maximum_filter(dist, size=win)
    peaks = mask & (dist == mx) & (dist > min_dist_frac * min(H, W))
    pl, n = ndimage.label(peaks)
    if n == 0:
        return mask.astype(int)
    seed_id = np.zeros((H, W), int)
    for i, (cy, cx) in enumerate(ndimage.center_of_mass(peaks, pl, range(1, n + 1)), start=1):
        seed_id[int(round(cy)), int(round(cx))] = i
    inds = ndimage.distance_transform_edt(seed_id == 0, return_indices=True, return_distances=False)
    nearest = seed_id[tuple(inds)]
    return np.where(mask, nearest, 0)


def regions_cut_sunflowers(rgb):
    """Wall B: individual flower HEADS (own label each) + stems/leaves as one more region."""
    subj = C.subject_mask(rgb, scene.white_threshold)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    heads = subj & (R > 0.42) & (R > G * 0.88)          # yellow/gold petals+disc (not green stems)
    heads = ndimage.binary_opening(heads, iterations=2)
    heads = ndimage.binary_closing(heads, iterations=4)
    heads = ndimage.binary_fill_holes(heads)
    lab = split_touching(heads)
    n_heads = int(lab.max())
    rest = subj & ~heads                                 # stems / leaves
    lab = np.where(rest, n_heads + 1, lab)
    return lab, n_heads


def regions_bouquet(rgb):
    """Wall A: flower mass vs vase (the vase has the most visible outline in the bouquet)."""
    subj = C.subject_mask(rgb, scene.white_threshold)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    flowers = subj & (sat > 0.18) & (R > 0.40)
    flowers = ndimage.binary_opening(flowers, iterations=2)
    flowers = ndimage.binary_closing(flowers, iterations=3)
    lab = np.zeros(rgb.shape[:2], int)
    lab[subj] = 2                                        # vase / remainder
    lab[flowers] = 1                                     # flower mass
    return lab


labB, n_heads = regions_cut_sunflowers(targets["B"])
labA = regions_bouquet(targets["A"])
print(f"Wall B: {n_heads} sunflower heads + stems region | Wall A: flowers/vase split")


def overlay(rgb, lab):
    rng = np.random.default_rng(3)
    out = np.clip(rgb, 0, 1).copy()
    for v in range(1, int(lab.max()) + 1):
        m = lab == v
        if m.any():
            out[m] = 0.45 * out[m] + 0.55 * rng.uniform(0.2, 1.0, 3)
    return out


fig, ax = plt.subplots(2, 2, figsize=(9, 9))
ax[0, 0].imshow(np.clip(targets["B"], 0, 1), origin="lower"); ax[0, 0].set_title("Wall B — cut sunflowers (warmth cut)")
ax[0, 1].imshow(overlay(targets["B"], labB), origin="lower"); ax[0, 1].set_title(f"regions: {n_heads} heads + stems")
ax[1, 0].imshow(np.clip(targets["A"], 0, 1), origin="lower"); ax[1, 0].set_title("Wall A — bouquet")
ax[1, 1].imshow(overlay(targets["A"], labA), origin="lower"); ax[1, 1].set_title("regions: flowers / vase")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/region_preview.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/region_preview.png")


# ---------------------------------------------------------------- 3. solve
def run(seed):
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
        region_masks={"A": labA, "B": labB})       # per-flower Voronoi -> crisp outlines
    pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    return dict(seed=seed, pred=pred, acc=acc, ts=ts, table=table, op=op, sc=sc, si=si,
                panels=panels, fr=fr,
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
    ax[ri, 0].set_ylabel(["Wall A  (bouquet)", "Wall B  (cut sunflowers)"][ri], fontsize=12)
ax[0, 0].set_title("SOURCE", fontweight="bold"); ax[0, 1].set_title("RECONSTRUCTED SHADOW", fontweight="bold")
plt.suptitle(f"Sunflowers pair — warmth cut + per-flower Voronoi, seed {r['seed']} ({r['shards']} shards)",
             fontsize=13, y=0.997)
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
    json.dump({"best_seed": r["seed"], "shards": r["shards"], "heads": n_heads,
               "A": r["acc"]["A"], "B": r["acc"]["B"]}, f, indent=2)
print(f"wrote {OUT}/metrics.json")
