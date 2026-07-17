"""
TEST 1: narrow feature-only importance mask (eyes/nose/lips box, NOT the whole face) at 300
shards, control config (signed dw0.5 cw0.5 match_tol0.30), 3 seeds. Same redistribution
mechanism as the (null) semantic face mask -- does NOT add shards, only concentrates them.
Only the mask differs between control and treatment (same seed -> same panels -> clean).

Reports feature-region SSIM, whole-face SSIM (starving check), global SSIM. Wall A (front,
has the face); wall B unaffected (no mask). Honest verdict: null, or actively harmful.
"""
import sys, dataclasses, os, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
from semantic_lib import face_roi_from_target

MT = 0.30; BUDGET = 300; SEM_W = 0.7; OUT = "out_thickness_test/feature_mask"; os.makedirs(OUT, exist_ok=True)
scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK
t = {"A": C.load_color_target("examples/girl_front_nobg.png", wr, white_thr=scene.white_threshold),
     "B": C.load_color_target("examples/girl_back_nobg.png", wr, white_thr=scene.white_threshold)}

# --- narrow feature box from the face bbox (central eyes/nose/lips zone) ---
soft_face, bbox = face_roi_from_target(t["A"])
y0, y1, x0, x1 = bbox; fh, fw = y1 - y0, x1 - x0
fy0, fy1 = y0 + int(0.22 * fh), y0 + int(0.80 * fh)
fx0, fx1 = x0 + int(0.18 * fw), x0 + int(0.82 * fw)
feat = np.zeros(t["A"].shape[:2], np.float32); feat[fy0:fy1, fx0:fx1] = 1.0
feat = ndimage.gaussian_filter(feat, sigma=max(t["A"].shape[:2]) * 0.02)
if feat.max() > 0: feat /= feat.max()
print(f"face bbox rows {y0}:{y1} cols {x0}:{x1}   feature box rows {fy0}:{fy1} cols {fx0}:{fx1}"
      f"  ({(fy1-fy0)*(fx1-fx0)/(t['A'].shape[0]*t['A'].shape[1])*100:.1f}% of wall)\n")

def run(seed, use_mask):
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                    targets=t, seed=seed, angle_deg_range=(5, 85))
    FS = scene.solve.fragment_size * 0.62
    SP = dataclasses.replace(scene.solve, fragment_size=FS,
                             fragment_min_area=scene.solve.fragment_min_area * 0.25,
                             fragment_max_area=scene.solve.fragment_max_area * 0.40)
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sem = {"A": feat} if use_mask else None
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, t, names=names, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=seed, damage_weight=0.5, credit_weight=0.5, match_tol=MT,
        shard_budget=None, semantic_masks=sem, semantic_weight=(SEM_W if use_mask else 0.0))
    pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))["A"]
    nsh = bs.get("A", {}).get("achieved", 0)
    return pred, nsh

def metrics(pred):
    return dict(feat=_metrics.ssim(pred[fy0:fy1, fx0:fx1], t["A"][fy0:fy1, fx0:fx1]),
                face=_metrics.ssim(pred[y0:y1, x0:x1], t["A"][y0:y1, x0:x1]),
                glob=_metrics.ssim(pred, t["A"]))

SEEDS = [1, 2, 3]
acc = {"control": {"feat": [], "face": [], "glob": [], "n": []},
       "feature-mask": {"feat": [], "face": [], "glob": [], "n": []}}
imgs = {}
for s in SEEDS:
    for lbl, um in (("control", False), ("feature-mask", True)):
        pred, nsh = run(s, um); m = metrics(pred)
        for k in ("feat", "face", "glob"): acc[lbl][k].append(m[k])
        acc[lbl]["n"].append(nsh)
        if s == SEEDS[0]: imgs[lbl] = pred

def ms(a): return st.mean(a), (st.pstdev(a) if len(a) > 1 else 0.0)
print(f"{'arm':14s} {'shards/A':>9} {'feat-SSIM':>18} {'face-SSIM':>18} {'global-SSIM':>18}")
print("-" * 82)
for lbl in ("control", "feature-mask"):
    fe, fes = ms(acc[lbl]["feat"]); fa, fas = ms(acc[lbl]["face"]); gl, gls = ms(acc[lbl]["glob"])
    print(f"{lbl:14s} {st.mean(acc[lbl]['n']):8.0f} {fe:7.3f}±{fes:.3f}      {fa:7.3f}±{fas:.3f}      {gl:7.3f}±{gls:.3f}")

dfe = st.mean(acc["feature-mask"]["feat"]) - st.mean(acc["control"]["feat"])
dgl = st.mean(acc["feature-mask"]["glob"]) - st.mean(acc["control"]["glob"])
noise = max(ms(acc["control"]["feat"])[1], ms(acc["feature-mask"]["feat"])[1], 0.003)
print(f"\nfeature-SSIM delta = {dfe:+.4f}   (per-seed noise ~{noise:.3f})")
print(f"global-SSIM  delta = {dgl:+.4f}")
verdict = ("NULL (|delta| < noise)" if abs(dfe) < noise else
           ("HELPS" if dfe > 0 else "HURTS")) + (" ; global " + ("HURT" if dgl < -noise else "unchanged"))
print(f"VERDICT: {verdict}")

# render control vs treatment with feature box
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].imshow(np.clip(t["A"], 0, 1), origin="lower"); ax[0].set_title("SOURCE (front)", fontweight="bold")
ax[1].imshow(np.clip(imgs["control"], 0, 1), origin="lower"); ax[1].set_title(f"control (no mask)")
ax[2].imshow(np.clip(imgs["feature-mask"], 0, 1), origin="lower"); ax[2].set_title(f"feature-mask (eyes/nose/lips)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
    # feature box in display coords (origin lower -> row maps directly)
    a.add_patch(plt.Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False, ec="#e04b3a", lw=2))
plt.suptitle(f"Narrow feature-only mask @ {BUDGET} shards, seed {SEEDS[0]} (red box = feature region)", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/compare.png")
