"""Run the full current pipeline (wide-angle search + signed-damage assignment) on the
two Monet San Giorgio Maggiore paintings. These are a deliberately hard test:
  - smooth gradients / impressionist brushwork (report limitation 6.2: untested)
  - SAME SCENE, opposite palettes -> the colour-agreement question (limitation 6.6) made concrete
"""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.preview.interactive3d import build_interactive

A_IMG = "examples/monet_day.jpg"
B_IMG = "examples/monet_dusk.jpeg"

scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res
names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
           "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
subject = {w: C.subject_mask(targets[w], scene.white_threshold) for w in ("A", "B")}

# CRITICAL DIAGNOSTIC: these paintings are full-bleed (no white background).
# Our bad/good cross-talk metric assumes a white background exists to splatter onto.
for w in ("A", "B"):
    frac = subject[w].mean() * 100
    print(f"wall {w}: subject mask covers {frac:.1f}% of the canvas")
print("  (if these are ~100%, there IS no background -- the bad-cross-talk metric\n"
      "   degenerates to ~0 by construction, because every stray shadow lands 'on content'.)\n")

FS = 0.135 / np.sqrt(0.5)     # 0.5x density = measured structural optimum
SP = dataclasses.replace(scene.solve, fragment_size=FS,
     fragment_min_area=scene.solve.fragment_min_area * (FS / 0.135) ** 2,
     fragment_max_area=scene.solve.fragment_max_area * (FS / 0.135) ** 2)

def run(seed, dw, cw, label):
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                    targets=targets, seed=seed, angle_deg_range=(5, 85))
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, res, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=names, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed, damage_weight=dw, credit_weight=cw)
    pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    def rk(keep):
        pt = C.stack_transmit_lut(names, sc, si).copy()
        for gi, p in enumerate(panels):
            if p.name not in keep: pt[gi] = 1.0
        return renderer.render_color_np(pt)
    dark = lambda im: (1.0 - im.mean(axis=-1)) > 0.05
    ct = {}
    for w in ("A", "B"):
        d = dark(pred[w]); tot = d.sum()
        nonprim = {p.name for p in panels if prim[p.name] != w}
        xt = dark(rk(nonprim)[w]) if nonprim else np.zeros_like(d)
        ct[w] = {"bad": 100*(xt & ~subject[w]).sum()/max(tot,1),
                 "good": 100*(xt & subject[w]).sum()/max(tot,1)}
    used = len({f["panel"] for f in fr})
    print(f"{label:26s} shards A={bs.get('A',{}).get('achieved',0)} B={bs.get('B',{}).get('achieved',0)} "
          f"panels={used}/14 | B_bad={ct['B']['bad']:5.1f}% B_good={ct['B']['good']:5.1f}% | "
          f"A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f} B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}")
    return pred, acc, ct, ts, table, op, sc, si, panels

SEED = 3
pred_rand, acc_r, ct_r, *_ = run(SEED, 0.0, None, "random (old rng.choice)")
pred_sign, acc_s, ct_s, ts, table, op, sc, si, panels = run(SEED, 0.5, 0.5, "signed damage (c=0.5)")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
cols = [("SOURCE (Monet)", {"A": np.clip(targets["A"],0,1), "B": np.clip(targets["B"],0,1)}),
        ("random assignment", {"A": np.clip(pred_rand["A"],0,1), "B": np.clip(pred_rand["B"],0,1)}),
        ("signed damage", {"A": np.clip(pred_sign["A"],0,1), "B": np.clip(pred_sign["B"],0,1)})]
for ci, (lbl, im) in enumerate(cols):
    for ri, w in enumerate(("A", "B")):
        ax[ri, ci].imshow(im[w], origin="lower", aspect="auto")
        ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0: ax[ri, ci].set_title(lbl, fontsize=13, fontweight="bold")
        if ci == 0: ax[ri, ci].set_ylabel(["Wall A (day)", "Wall B (dusk)"][ri], fontsize=12)
plt.suptitle("Monet San Giorgio Maggiore — same scene, opposite palettes", fontsize=15)
plt.tight_layout()
import os; os.makedirs("out_thickness_test/monet", exist_ok=True)
plt.savefig("out_thickness_test/monet/compare.png", dpi=100, bbox_inches="tight")
print("\nsaved out_thickness_test/monet/compare.png")

sp = decompose.panel_stack_pieces(ts, sc, names)
pc = {id(p): ch for items in sp.values() for p, ch, _s in items}
flat = {n: [p for p, _c, _s in items] for n, items in sp.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
html, _ = build_interactive(ts, table, op, None, "out_thickness_test/monet/scene.html",
                            rays=40, auto_open=False, wall_rgb=pred_sign,
                            pieces=flat, color_of=col, embed_plotly=True)
print(f"wrote {html}")
