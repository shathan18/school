"""FINAL Scream deliverable: the full painting (scene + recognizable figure) via the grayscale
gradient OPTIMIZER -- the rig's most faithful mode -- on Wall A, with Munch's self-portrait on
Wall B. Grayscale is inherent to a real cast shadow; this is what makes the figure recognizable
where the coloured-shard mode cannot.

Searches a few panel layouts, keeps the most faithful, then writes:
  walls.png (target | optimizer | fabricable-cut, both walls), reconA*.png, opacity.npy,
  scene.html (interactive 3D), scene_front/back.png, and the DXF/SVG cut files under cut/.

  py out_thickness_test/scream_final.py [OUT] [SCENE_IMG] [PARTNER_IMG] [FIG_IMG] [n_seeds]
"""
import sys, os, dataclasses, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve.optimizer import solve
from shadowart.solve.initializer import back_project
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart.metrics import _luma, ssim, edge_fidelity
from shadowart.preview.render3d import save_scene_3d
from shadowart.preview.interactive3d import build_interactive
from shadowart.cli import raster_to_pieces
from shadowart.fabricate.joints import build_panel_drawings
from shadowart.fabricate import nesting, layers, export_dxf, export_svg

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out_scream_final"); OUT.mkdir(exist_ok=True)
A_IMG = sys.argv[2] if len(sys.argv) > 2 else "examples/scream_src.jpg"
B_IMG = sys.argv[3] if len(sys.argv) > 3 else "examples/munch_self_nobg.png"
FIG_IMG = sys.argv[4] if len(sys.argv) > 4 else "examples/scream_figure_nobg.png"
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res


def load_scene_and_figure(scene_path, fig_path, white_thr):
    s = Image.open(scene_path).convert("RGB"); n = Image.open(fig_path).convert("RGB")
    if n.size != s.size:
        n = n.resize(s.size, Image.LANCZOS)
    return np.asarray(s, np.float32) / 255.0, C.subject_mask(np.asarray(n, np.float32) / 255.0, white_thr)


def fit_flip(rgb, mask, wall_res, white_thr, content_frac=0.92):
    Hn, Wn = wall_res
    sub = ndimage.binary_opening(C.subject_mask(rgb, white_thr), iterations=2)
    lbl, n = ndimage.label(sub)
    if n:
        cnt = np.bincount(lbl.ravel()); cnt[0] = 0
        ys, xs = np.where(lbl == int(cnt.argmax()))
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = 0, rgb.shape[0], 0, rgb.shape[1]
    rc, mc = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ch, cw = rc.shape[:2]
    sc = content_frac * min(Wn / cw, Hn / ch)
    nw, nh = max(1, int(round(cw * sc))), max(1, int(round(ch * sc)))
    ri = Image.fromarray((np.clip(rc, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    mi = Image.fromarray((mc.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)
    R = np.ones((Hn, Wn, 3), np.float32); M = np.zeros((Hn, Wn), bool)
    oy, ox = (Hn - nh) // 2, (Wn - nw) // 2
    R[oy:oy + nh, ox:ox + nw] = np.asarray(ri, np.float32) / 255.0
    M[oy:oy + nh, ox:ox + nw] = np.asarray(mi) > 127
    return np.flipud(R).copy(), np.flipud(M).copy()


scene_raw, fig_raw = load_scene_and_figure(A_IMG, FIG_IMG, scene0.white_threshold)
a_scene, _ = fit_flip(scene_raw, fig_raw, WR, scene0.white_threshold)
B = C.load_color_target(B_IMG, WR, white_thr=scene0.white_threshold)
tg = {"A": (1.0 - _luma(a_scene)).astype(np.float32), "B": (1.0 - _luma(B)).astype(np.float32)}
sp = scene0.solve

best = None
for seed in range(1, N + 1):
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16,
                                    targets={"A": a_scene, "B": B}, seed=seed, angle_deg_range=(5, 85),
                                    anchor_range=sp.search_anchor_range, standoff=sp.search_standoff,
                                    mag_cap=sp.search_mag_cap, u_size_range=sp.search_u_size_range,
                                    v_range=sp.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=dataclasses.replace(sp, iters=500, lr=0.08, diagonal_frac=0.0))
    table = build_projection_table(ts); R = Renderer(ts, table)
    op, _ = solve(ts, R, tg, init_opacity=back_project(ts, table, tg), verbose=False)
    pred = R.render_np(op)
    mse = float(((pred["A"] - tg["A"]) ** 2).mean() + ((pred["B"] - tg["B"]) ** 2).mean())
    print(f"seed {seed}: MSE {mse:.5f}")
    if best is None or mse < best["mse"]:
        best = dict(mse=mse, seed=seed, ts=ts, table=table, R=R, op=op, pred=pred)

ts, table, R, op, pred = best["ts"], best["table"], best["R"], best["op"], best["pred"]
op_bin = (op > 0.5).astype(np.float32); pred_bin = R.render_np(op_bin)   # what actually gets cut
for w in ("A", "B"):
    p3 = np.repeat(np.clip(pred[w], 0, 1)[..., None], 3, 2); t3 = np.repeat(np.clip(tg[w], 0, 1)[..., None], 3, 2)
    print(f"Wall {w}: optimizer ssim {ssim(p3, t3):.3f} edge {edge_fidelity(p3, t3):.3f}")
print(f"best seed {best['seed']}")

# walls comparison (grayscale; cmap gray_r so high darkness = black)
fig, ax = plt.subplots(2, 3, figsize=(13, 10))
for ri, w in enumerate(("A", "B")):
    for ci, (title, d) in enumerate([("TARGET", tg[w]), ("OPTIMIZER", pred[w]), ("FABRICABLE (cut)", pred_bin[w])]):
        ax[ri, ci].imshow(np.clip(d, 0, 1), origin="lower", cmap="gray_r")
        ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0:
            ax[ri, ci].set_title(title, fontweight="bold")
    ax[ri, 0].set_ylabel(f"Wall {w}" + ("  (Scream)" if w == "A" else "  (Munch)"), fontsize=12)
plt.suptitle(f"The Scream - grayscale optimizer, full scene (seed {best['seed']})", y=0.995, fontsize=13)
plt.tight_layout(); plt.savefig(OUT / "walls.png", dpi=110, bbox_inches="tight"); plt.close()


def save(arr, path, scale=3):
    a = np.clip(1.0 - np.flipud(np.clip(arr, 0, 1)), 0, 1)     # darkness -> lightness for a natural look
    im = Image.fromarray((a * 255).astype(np.uint8)); im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)
for w in ("A", "B"):
    save(pred[w], OUT / f"recon{w}_optimizer.png"); save(pred_bin[w], OUT / f"recon{w}_fabricable.png")

np.save(OUT / "opacity.npy", op)
save_scene_3d(ts, str(OUT / "scene_front.png"), elev=18, azim=-60, title="Scene - front (viewer side)")
save_scene_3d(ts, str(OUT / "scene_back.png"), elev=18, azim=120, title="Scene - from behind")
build_interactive(ts, table, op, pred, str(OUT / "scene.html"), rays=40, auto_open=False)

# fabrication: threshold -> min-feature -> contours -> kerf -> nest -> DXF/SVG
pieces, total = raster_to_pieces(ts, op)
print(f"{total} cut pieces across {len(ts.panels)} panels")
drawings = build_panel_drawings(ts, pieces)
placements = nesting.nest(drawings, ts.fab.sheet_size)
oversize = [p.name for p in placements if p.oversize]
if oversize:
    print(f"WARNING: panels exceed stock sheet {ts.fab.sheet_size} m: {oversize}")
written = []
for ln, dl in layers.group_mono(drawings).items():
    cd = OUT / "cut" / ln
    written += export_dxf.export_all_dxf(dl, cd) + export_svg.export_all_svg(dl, cd)
print(f"wrote {len(written)} cut files under {OUT/'cut'}")
print(f"done -> {OUT}/  (walls.png, recon*, scene.html, scene_front/back.png, cut/)")
