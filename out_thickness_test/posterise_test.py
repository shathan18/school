"""
TEST 2: does a POSTERISED face (flattened to a handful of solid tones) become recognizable
at 300 shards where a photographic one does not? Control config (signed dw0.5 cw0.5
match_tol0.30). Runs the SAME pipeline/seeds on the photographic girl and on a posterised
girl (front/back), renders both reconstructions side by side, reports face + global SSIM.
"""
import sys, dataclasses, os, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
from semantic_lib import face_roi_from_target

MT = 0.30; K = 7; OUT = "out_thickness_test/posterise"; os.makedirs(OUT, exist_ok=True)
SCRATCH = r"C:\Users\User1\AppData\Local\Temp\claude\c--Users-User1-Downloads-matterOfPerspective-school\4aeb4c2a-1c5d-477e-85fe-c11abe7b1561\scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK

def posterise(src, dst, k=K):
    """Structure-preserving posterise: quantise SUBJECT luminance into k equal-population
    bands (so dark eyes/lips/brows keep their own tone) and paint each band its mean colour.
    White background left white. Preserves face structure, unlike a global colour median-cut."""
    im = np.asarray(Image.open(src).convert("RGB"), float) / 255.0
    luma = im @ np.array([0.299, 0.587, 0.114])
    subj = luma < 0.86                                   # non-white = the figure
    out = im.copy()
    edges = np.quantile(luma[subj], np.linspace(0, 1, k + 1))
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        band = subj & (luma >= lo) & ((luma <= hi) if i == k - 1 else (luma < hi))
        if band.any():
            out[band] = im[band].mean(0)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst); return dst

photo = {"A": "examples/girl_front_nobg.png", "B": "examples/girl_back_nobg.png"}
post = {w: posterise(photo[w], os.path.join(SCRATCH, f"post_{w}.png")) for w in ("A", "B")}
print(f"posterised to {K} tones -> {post['A']}, {post['B']}")

def load(paths):
    return {w: C.load_color_target(paths[w], wr, white_thr=scene.white_threshold) for w in ("A", "B")}
T = {"photo": load(photo), "post": load(post)}
bbox = {k: face_roi_from_target(T[k]["A"])[1] for k in T}     # face bbox per (its own) target

def run(targets, seed):
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                    targets=targets, seed=seed, angle_deg_range=(5, 85))
    FS = scene.solve.fragment_size * 0.60
    SP = dataclasses.replace(scene.solve, fragment_size=FS,
                             fragment_min_area=scene.solve.fragment_min_area * 0.25,
                             fragment_max_area=scene.solve.fragment_max_area * 0.40)
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=names, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=seed, damage_weight=0.5, credit_weight=0.5, match_tol=MT)
    pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))
    return pred, bs.get("A", {}).get("achieved", 0)

SEEDS = [1, 2, 3]
res = {"photo": {"face": [], "glob": [], "n": []}, "post": {"face": [], "glob": [], "n": []}}
show = {}
for s in SEEDS:
    for k in ("photo", "post"):
        pred, n = run(T[k], s)
        y0, y1, x0, x1 = bbox[k]
        res[k]["face"].append(_metrics.ssim(pred["A"][y0:y1, x0:x1], T[k]["A"][y0:y1, x0:x1]))
        res[k]["glob"].append(_metrics.ssim(pred["A"], T[k]["A"]))
        res[k]["n"].append(n)
        if s == SEEDS[0]: show[k] = pred["A"]

ms = lambda a: (st.mean(a), st.pstdev(a) if len(a) > 1 else 0.0)
print(f"\n{'target':8s} {'shards/A':>9} {'face-SSIM':>18} {'global-SSIM':>18}")
print("-" * 58)
for k in ("photo", "post"):
    fa, fas = ms(res[k]["face"]); gl, gls = ms(res[k]["glob"])
    print(f"{k:8s} {st.mean(res[k]['n']):8.0f} {fa:7.3f}±{fas:.3f}      {gl:7.3f}±{gls:.3f}")
print(f"\nface-SSIM: photo {ms(res['photo']['face'])[0]:.3f}  ->  posterised {ms(res['post']['face'])[0]:.3f}"
      f"   (delta {ms(res['post']['face'])[0]-ms(res['photo']['face'])[0]:+.3f})")

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(np.clip(T["photo"]["A"], 0, 1), origin="lower"); ax[0, 0].set_title("photographic — source", fontweight="bold")
ax[0, 1].imshow(np.clip(show["photo"], 0, 1), origin="lower"); ax[0, 1].set_title("photographic — reconstruction")
ax[1, 0].imshow(np.clip(T["post"]["A"], 0, 1), origin="lower"); ax[1, 0].set_title(f"posterised ({K} tones) — source", fontweight="bold")
ax[1, 1].imshow(np.clip(show["post"], 0, 1), origin="lower"); ax[1, 1].set_title(f"posterised ({K} tones) — reconstruction")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.suptitle(f"Photographic vs posterised face @ ~{st.mean(res['post']['n']):.0f} shards/wall, seed {SEEDS[0]}", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/compare.png")
