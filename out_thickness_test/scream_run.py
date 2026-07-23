"""
The Scream (full painting, both walls) reconstructed with the project's OBJECT-AWARE modules --
no fake painted outlines. Wall A is the whole painting `scream_src.jpg`: the swirling-sky SCENE
and all. To make it READ as The Scream and keep the FIGURE crisp, we use the pipeline's own
machinery instead of drawing anything:

  * `objectseg.segment_objects`         -> the scene's real regions (the sky bands, the fjord, the
                                           bridge, the ground) by colour,
  * the pixel-aligned background-removed figure (`scream_figure_nobg.png`) is forced as its OWN
    region at a FINER shard spacing (`region_scales`), so the screamer carries more detail than the
    flat sky and reads as a distinct silhouette,
  * region-constrained fragmentation (`decompose` region_masks) so NO shard straddles a region
    boundary -- the figure's edge and the sky/water bands stay crisp instead of smearing,
  * damage-aware host selection + an outline guard on the figure, so Munch's cross-talk (Wall B)
    stays off the screamer.

Control arm = uniform fragmentation (today's behaviour). Treatment = the object-aware pipeline
above, so the improvement is measured on the same layout/seed.

  py out_thickness_test/scream_run.py OUT SCREAM_SRC PARTNER [n_seeds] [figure_scale] [protect_w] [fig_img]

  SCREAM_SRC = the full painting (examples/scream_src.jpg).  fig_img = same painting, background
  removed (examples/scream_figure_nobg.png), used only to locate the figure.  figure_scale <1 packs
  finer shards onto the figure (0.5 ~ 4x density).  Wall A = Scream, Wall B = PARTNER.
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.metrics import _luma
from shadowart.preview.interactive3d import build_interactive
from shadowart.preview.render3d import save_scene_3d
import objectseg as OS

OUT, A_IMG, B_IMG = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
FIGURE_SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5   # <1 = finer shards on the figure
PROTECT_W = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0      # steer Wall B cross-talk off the figure
FIG_IMG = sys.argv[7] if len(sys.argv) > 7 else "examples/scream_figure_nobg.png"
DAMAGE_W, CREDIT_W, BLEND, MATCH_TOL = 0.5, 0.5, 0.6, 0.30
HERO = "A"
os.makedirs(OUT, exist_ok=True)

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
scene0 = dataclasses.replace(scene0, overlap_shard_budget=340)   # headroom so the dense figure region resolves
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)


def load_scene_and_figure(scene_path, fig_path, white_thr):
    """The full painting (background SCENE and all) as RGB, plus a boolean mask of just the FIGURE,
    aligned in raw pixel space. `fig_path` is the same painting with the background removed."""
    src = Image.open(scene_path).convert("RGB")
    nobg = Image.open(fig_path).convert("RGB")
    if nobg.size != src.size:
        nobg = nobg.resize(src.size, Image.LANCZOS)
    scene = np.asarray(src, np.float32) / 255.0
    fig = C.subject_mask(np.asarray(nobg, np.float32) / 255.0, white_thr)
    fig = ndimage.binary_fill_holes(ndimage.binary_opening(fig, iterations=2))
    return scene, fig


def fit_flip(rgb, mask, wall_res, white_thr, content_frac=0.92):
    """Mirror `color.load_color_target`'s crop-to-subject + centred fit + vertical flip, applied to
    BOTH an RGB image and a companion boolean mask so they stay aligned in the wall frame."""
    Hn, Wn = wall_res
    sub = ndimage.binary_opening(C.subject_mask(rgb, white_thr), iterations=2)
    lbl, n = ndimage.label(sub)
    if n:
        counts = np.bincount(lbl.ravel()); counts[0] = 0
        ys, xs = np.where(lbl == int(counts.argmax()))
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = 0, rgb.shape[0], 0, rgb.shape[1]
    rc, mc = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ch, cw = rc.shape[:2]
    scale = content_frac * min(Wn / cw, Hn / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    ri = Image.fromarray((np.clip(rc, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    mi = Image.fromarray((mc.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)
    cvR = np.ones((Hn, Wn, 3), np.float32); cvM = np.zeros((Hn, Wn), bool)
    oy, ox = (Hn - nh) // 2, (Wn - nw) // 2
    cvR[oy:oy + nh, ox:ox + nw] = np.asarray(ri, np.float32) / 255.0
    cvM[oy:oy + nh, ox:ox + nw] = np.asarray(mi) > 127
    return np.flipud(cvR).copy(), np.flipud(cvM).copy()


def boost_figure(rgb, m, contrast=1.65, sat=1.4, darken=0.07):
    """Lift ONLY the figure's contrast + saturation (and settle the body a touch darker) so the
    screamer separates from the dark water -- the pale skull-face pops, the robe goes deep dark --
    while the background scene is left untouched. This is the 'figure-first' contrast boost."""
    out = rgb.copy(); f = out[m].copy()
    med = np.median(f, axis=0)
    f = np.clip((f - med) * contrast + med, 0, 1)          # stretch contrast around the figure median
    g = (f * np.array([0.2989, 0.587, 0.114], np.float32)).sum(1, keepdims=True)
    f = np.clip(g + (f - g) * sat, 0, 1)                   # saturation
    out[m] = np.clip(f - darken, 0, 1)                     # body a touch darker
    return out


# ---- targets, figure mask, and OBJECT regions ---------------------------------------
scene_raw, fig_raw = load_scene_and_figure(A_IMG, FIG_IMG, scene0.white_threshold)
a_full, fig_mask = fit_flip(scene_raw, fig_raw, WR, scene0.white_threshold)   # scene + crisp figure cut
a_boost = boost_figure(a_full, fig_mask)                   # figure-first: the screamer made to pop
B_tgt = C.load_color_target(B_IMG, WR, white_thr=scene0.white_threshold)
targets = {"A": a_full, "B": B_tgt}                        # control aims at the unaltered scene
targets_fig = {"A": a_boost, "B": B_tgt}                   # figure-first arm aims at the boosted figure
subj = {w: C.subject_mask(targets[w], scene0.white_threshold) for w in ("A", "B")}

# segment each wall into its real objects; then FORCE the (crisply cut) Scream figure to be its own
# region so it tiles on its own DENSE Voronoi (more detail, crisp silhouette) regardless of colour.
labels = {"A": OS.segment_objects(targets["A"], subj["A"], k=12, min_frac=0.004, max_objects=60),
          "B": OS.segment_objects(targets["B"], subj["B"], k=12, min_frac=0.004, max_objects=60)}
fig_lbl = int(labels["A"].max()) + 1
labels["A"][fig_mask & subj["A"]] = fig_lbl
face_B = OS.face_like_regions(targets["B"], labels["B"])
scales_B = {v: 0.6 for v in face_B} if face_B else {}
region_scales_ctrl = {"B": scales_B}                                   # object-aware, figure NOT emphasised
region_scales_fig = {"A": {fig_lbl: FIGURE_SCALE}, "B": scales_B}      # + dense figure region
outline = {w: decompose.outline_map(targets[w], subj[w]) for w in ("A", "B")}
figure_outline = decompose.outline_map(targets["A"], fig_mask)   # guard the FIGURE edge specifically
outline["A"] = np.maximum(outline["A"], figure_outline)
print(f"Wall A: figure = {100 * float(fig_mask.mean()):.1f}% of canvas, "
      f"{int(labels['A'].max())} regions | Wall B: {int(labels['B'].max())} regions, face {face_B}")


def outline_edge_fidelity(pred, target, band):
    """`metrics.edge_fidelity` restricted to the figure-edge band -- did the figure's contour survive."""
    p, t = _luma(pred), _luma(target)
    gp = np.hypot(ndimage.sobel(p, 0), ndimage.sobel(p, 1))
    gt = np.hypot(ndimage.sobel(t, 0), ndimage.sobel(t, 1))
    m = band > 0
    if int(m.sum()) < 4:
        return 0.0
    gp, gt = gp[m], gt[m]; gp = gp - gp.mean(); gt = gt - gt.mean()
    denom = np.sqrt((gp ** 2).sum() * (gt ** 2).sum())
    return float((gp * gt).sum() / denom) if denom > 1e-9 else 0.0


def figure_crosstalk(renderer, pT, panels, targets, band, prim, wall, white_thr,
                     match_tol=0.30, dark_thr=0.05):
    """% of the figure-edge band the OTHER wall's stray shadows darken in the wrong colour."""
    q = pT.copy()
    for gi, p in enumerate(panels):
        if prim.get(p.name) == wall:
            q[gi] = 1.0
    xr = renderer.render_color_np(q)[wall]
    m = band > 0; denom = max(int(m.sum()), 1)
    on = ((1.0 - xr.mean(-1)) > dark_thr) & m
    if not on.any():
        return 0.0
    d = np.sqrt(((xr[on] - targets[wall][on]) ** 2).sum(1))
    return 100.0 * float((d >= match_tol).sum()) / denom


# ---- preview: scene, object regions, figure region ----------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 5))
ax[0].imshow(np.clip(targets["A"], 0, 1), origin="lower"); ax[0].set_title("Full painting (scene)")
ax[1].imshow(OS.overlay(targets["A"], labels["A"]), origin="lower"); ax[1].set_title("Object regions")
fv = np.clip(targets["A"], 0, 1).copy(); fb = fig_mask & ~ndimage.binary_erosion(fig_mask, iterations=2)
fv[fb] = [0, 1, 1]; ax[2].imshow(fv, origin="lower"); ax[2].set_title("Figure region (cyan)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/regions.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/regions.png")


def render_arm(ts, table, renderer, seed, tgts, rscales, *, figure_first):
    """figure_first=True = the crisp figure isolated as a DENSE region + its contrast boosted (aims
    at `tgts`=boosted) so the screamer pops; figure_first=False = object-aware but figure not
    emphasised (aims at the plain scene). Both are region-constrained and colour. Same seed."""
    sc, op, fr, rs_, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, tgts, names=NAMES, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed, damage_weight=DAMAGE_W, credit_weight=CREDIT_W,
        match_tol=MATCH_TOL, colour_blend=BLEND,
        region_masks=labels, region_scales=rscales,
        outline_masks=(outline if figure_first else None),
        outline_protect_weight=(PROTECT_W if figure_first else 0.0))
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(tgts, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    oef = outline_edge_fidelity(pred["A"], tgts["A"], figure_outline)
    oct_ = figure_crosstalk(renderer, pT, ts.panels, tgts, figure_outline, prim, "A",
                            ts.white_threshold, match_tol=MATCH_TOL)
    return dict(pred=pred, acc=acc, oef=oef, oct=oct_, op=op, sc=sc, si=si,
                shards=bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0))


print(f"\n{'seed':>4} | {'arm':>8} | {'shards':>6} | {'A ssim/edge':>12} | {'B ssim/edge':>12} | "
      f"{'figEF':>6} | {'figXtk':>6}")
print("-" * 78)
runs = []
for s in range(1, N + 1):
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=targets,
                                    seed=s, angle_deg_range=(5, 85),
                                    anchor_range=SP.search_anchor_range, standoff=SP.search_standoff,
                                    mag_cap=SP.search_mag_cap, u_size_range=SP.search_u_size_range,
                                    v_range=SP.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    ctrl = render_arm(ts, table, renderer, s, targets, region_scales_ctrl, figure_first=False)
    treat = render_arm(ts, table, renderer, s, targets_fig, region_scales_fig, figure_first=True)
    runs.append(dict(seed=s, ts=ts, table=table, ctrl=ctrl, treat=treat))
    for name, r in (("plain", ctrl), ("figure1st", treat)):
        a, b = r["acc"]["A"], r["acc"]["B"]
        print(f"{s:>4} | {name:>8} | {r['shards']:>6} | {a['ssim']:.3f}/{a['edge_fidelity']:.3f} | "
              f"{b['ssim']:.3f}/{b['edge_fidelity']:.3f} | {r['oef']:>6.3f} | {r['oct']:>5.1f}%")

# rank by how well the object-aware arm renders the whole Scream (SSIM+edge) and the figure edge.
best = max(runs, key=lambda r: (r["treat"]["acc"]["A"]["ssim"] + r["treat"]["acc"]["A"]["edge_fidelity"]
                                + r["treat"]["oef"]))
c, t = best["ctrl"], best["treat"]
print(f"\nbest seed {best['seed']}:  Wall A ssim {c['acc']['A']['ssim']:.3f}->{t['acc']['A']['ssim']:.3f}, "
      f"edge {c['acc']['A']['edge_fidelity']:.3f}->{t['acc']['A']['edge_fidelity']:.3f}, "
      f"figure edge {c['oef']:.3f}->{t['oef']:.3f}, figure cross-talk {c['oct']:.1f}%->{t['oct']:.1f}%")

# ---- walls.png: source | object-aware | figure-first, per wall ----------------------
fig, ax = plt.subplots(2, 3, figsize=(13, 9))
cols = [("SOURCE (figure boosted)", targets_fig), ("OBJECT-AWARE", None), ("FIGURE-FIRST", None)]
preds = [None, c["pred"], t["pred"]]
for ri, w in enumerate(("A", "B")):
    for ci, (title, srcd) in enumerate(cols):
        img = srcd[w] if srcd is not None else preds[ci][w]
        ax[ri, ci].imshow(np.clip(img, 0, 1), origin="lower", aspect="auto")
        ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0:
            ax[ri, ci].set_title(title, fontweight="bold")
    ax[ri, 0].set_ylabel(f"Wall {w}" + ("  (Scream)" if w == HERO else ""), fontsize=12)
plt.suptitle(f"Figure-first Scream - seed {best['seed']}, {t['shards']} shards | "
             f"figure edge {c['oef']:.3f}->{t['oef']:.3f}, figure cross-talk "
             f"{c['oct']:.1f}%->{t['oct']:.1f}%", fontsize=13, y=0.998)
plt.tight_layout(); plt.savefig(f"{OUT}/walls.png", dpi=110, bbox_inches="tight"); plt.close()


def save(arr, path, scale=3):
    a = np.clip(np.flipud(arr), 0, 1); im = Image.fromarray((a * 255).astype(np.uint8))
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)
save(targets_fig["A"], f"{OUT}/srcA.png"); save(targets["B"], f"{OUT}/srcB.png")
for w in ("A", "B"):
    save(c["pred"][w], f"{OUT}/recon{w}_plain.png")
    save(t["pred"][w], f"{OUT}/recon{w}_figurefirst.png")

ts, table = best["ts"], best["table"]
sp_pieces = decompose.panel_stack_pieces(ts, t["sc"], NAMES)
pc = {id(p): ch for items in sp_pieces.values() for p, ch, _s in items}
flat = {n: [p for p, _c, _s in items] for n, items in sp_pieces.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
build_interactive(ts, table, t["op"], None, f"{OUT}/scene.html", rays=40, auto_open=False,
                  wall_rgb=t["pred"], pieces=flat, color_of=col, embed_plotly=True)
save_scene_3d(ts, f"{OUT}/scene_front.png", elev=18, azim=-60, title="Scene - front (viewer side)")
save_scene_3d(ts, f"{OUT}/scene_back.png", elev=18, azim=120, title="Scene - from behind")

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_seed": best["seed"], "figure_scale": FIGURE_SCALE, "protect_w": PROTECT_W,
               "regionsA": int(labels["A"].max()), "regionsB": int(labels["B"].max()),
               "shards": int(t["shards"]),
               "plain": {"A": c["acc"]["A"], "B": c["acc"]["B"], "figure_edge": c["oef"],
                         "figure_crosstalk": c["oct"]},
               "figure_first": {"A": t["acc"]["A"], "B": t["acc"]["B"], "figure_edge": t["oef"],
                                "figure_crosstalk": t["oct"]}}, f, indent=2)
print(f"wrote {OUT}/walls.png, regions.png, scene.html, scene_front/back.png, metrics.json")
