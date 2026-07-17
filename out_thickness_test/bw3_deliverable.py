"""
Fixed pipeline on the BLACK-AND-WHITE Girl front/back pair (grayscale, bg removed).
Grayscale => reproduced purely by the K (black) channel at varying intensity; and the two
targets are trivially colour-compatible (both gray), so double-duty should be near-maximal.
Same config as the colour deliverable: signed dw0.5 cw0.5 match_tol0.30, default density, 5 seeds.
"""
import sys, dataclasses, os, json, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.preview.interactive3d import build_interactive

A_IMG = "examples/WhatsApp Image 2026-07-16 at 15.33.42 (1).jpeg"       # 1456 tall = front
B_IMG = "examples/WhatsApp Image 2026-07-16 at 15.33.42.jpeg"   # 1503 tall = back
MT = 0.30; OUT = "out_thickness_test/bw3_final"; os.makedirs(OUT, exist_ok=True)
scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK
t = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
     "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
subject = {w: C.subject_mask(t[w], scene.white_threshold) for w in ("A", "B")}
SP = scene.solve

def bgood(renderer, pT, panels, prim):
    out = {}
    for w in ("A", "B"):
        s = subject[w]; den = max(s.sum(), 1)
        nonprim = {p.name for p in panels if prim[p.name] != w}
        q = pT.copy()
        for gi, p in enumerate(panels):
            if p.name not in nonprim: q[gi] = 1.0
        xr = renderer.render_color_np(q)[w]
        on = ((1.0 - xr.mean(-1)) > 0.05) & s
        out[w] = 100.0 * (np.sqrt(((xr[on] - t[w][on]) ** 2).sum(1)) < MT).sum() / den if on.any() else 0.0
    return out

def run(seed):
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16, targets=t, seed=seed, angle_deg_range=(5, 85))
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, t, names=names, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=seed, damage_weight=0.5, credit_weight=0.5, match_tol=MT)
    pT = C.stack_transmit_lut(names, sc, si); pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(t, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    return dict(seed=seed, pred=pred, acc=acc, bg=bgood(renderer, pT, panels, prim),
                shards=bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0),
                used=len({f["panel"] for f in fr}), ts=ts, table=table, op=op, sc=sc, si=si, panels=panels)

SEEDS = [1, 2, 3, 4, 5]
print(f"pair: Girl LOW-CONTRAST B&W  |  signed dw0.5 cw0.5 match_tol {MT}\n")
print(f"{'seed':>4} {'shards':>7} {'used':>5} | {'A rmse/ssim/edge':>22} | {'B rmse/ssim/edge':>22} | {'B_good A/B':>16}")
print("-" * 92)
runs = []
for s in SEEDS:
    r = run(s); runs.append(r); a, b = r["acc"]["A"], r["acc"]["B"]
    print(f"{s:>4} {r['shards']:>7} {r['used']:>4}/14 | {a['rmse']:.3f}/{a['ssim']:.3f}/{a['edge_fidelity']:.3f}      | "
          f"{b['rmse']:.3f}/{b['ssim']:.3f}/{b['edge_fidelity']:.3f}      | {r['bg']['A']:5.1f}% /{r['bg']['B']:5.1f}%")
ssA=[r["acc"]["A"]["ssim"] for r in runs]; ssB=[r["acc"]["B"]["ssim"] for r in runs]
bgA=[r["bg"]["A"] for r in runs]; bgB=[r["bg"]["B"] for r in runs]
print(f"\nacross {len(SEEDS)} seeds: SSIM_A {np.mean(ssA):.3f}±{np.std(ssA):.3f} SSIM_B {np.mean(ssB):.3f}±{np.std(ssB):.3f}"
      f"  |  B_good_A {np.mean(bgA):.1f}±{np.std(bgA):.1f}% B_good_B {np.mean(bgB):.1f}±{np.std(bgB):.1f}%")
best = min(runs, key=lambda r: 0.5*(r["acc"]["A"]["rmse"]+r["acc"]["B"]["rmse"]))
print(f"best seed by mean RMSE = {best['seed']}")

r = best; pred = r["pred"]
fig, ax = plt.subplots(2, 2, figsize=(10, 12))
for ri, w in enumerate(("A", "B")):
    ax[ri,0].imshow(np.clip(t[w],0,1), origin="lower"); ax[ri,1].imshow(np.clip(pred[w],0,1), origin="lower")
    for ci in (0,1): ax[ri,ci].set_xticks([]); ax[ri,ci].set_yticks([])
    ax[ri,0].set_ylabel(["Wall A (front, B&W)","Wall B (back, B&W)"][ri], fontsize=13)
ax[0,0].set_title("SOURCE", fontweight="bold", fontsize=14); ax[0,1].set_title("RECONSTRUCTED SHADOW", fontweight="bold", fontsize=14)
plt.suptitle(f"Girl low-contrast B&W — seed {r['seed']} ({r['shards']} shards, {r['used']}/14 panels)", fontsize=14, y=0.997)
plt.tight_layout(); plt.savefig(f"{OUT}/walls.png", dpi=110, bbox_inches="tight"); plt.close()
print(f"saved {OUT}/walls.png")

def save(arr, path, scale=3):
    im = Image.fromarray((np.clip(np.flipud(arr),0,1)*255).astype(np.uint8))
    im.resize((im.width*scale, im.height*scale), Image.NEAREST).save(path)
for w in ("A","B"): save(t[w], f"{OUT}/src{w}.png"); save(pred[w], f"{OUT}/recon{w}.png")

ts, table = r["ts"], r["table"]
spp = decompose.panel_stack_pieces(ts, r["sc"], names)
pc = {id(p): ch for items in spp.values() for p, ch, _s in items}
flat = {n: [p for p,_c,_s in items] for n,items in spp.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
html,_ = build_interactive(ts, table, r["op"], None, f"{OUT}/scene.html", rays=40, auto_open=False,
                           wall_rgb=pred, pieces=flat, color_of=col, embed_plotly=True)
print(f"wrote {html}")
json.dump({"best_seed": r["seed"], "shards": r["shards"], "panels_used": r["used"],
           "per_seed":[{"seed":x["seed"],"A":x["acc"]["A"],"B":x["acc"]["B"],"bg":x["bg"]} for x in runs],
           "mean":{"ssimA":float(np.mean(ssA)),"ssimB":float(np.mean(ssB)),"bgA":float(np.mean(bgA)),
                   "bgB":float(np.mean(bgB)),"bgA_sd":float(np.std(bgA)),"bgB_sd":float(np.std(bgB))}},
          open(f"{OUT}/metrics.json","w"), indent=2)
print(f"wrote {OUT}/metrics.json")
