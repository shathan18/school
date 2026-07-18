"""
Three-way mix & match of the Van Gogh yellow still lifes: sunflowers in a VASE, cut sunflowers
on a SURFACE, and the basket of ORANGES -- every pairing, 10 greedy restarts each.

One grid (row per pairing: Wall-A recon | Wall-B recon | metrics) plus a ranked table.
Ranked by CLARITY (SSIM_A + SSIM_B); double duty shown secondary (it trades against visible
cross-talk staining, so it is not the ranking criterion).

Run AFTER the three runs finish:  py out_thickness_test/mix_gallery.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

RUNS = [("vase  x  surface", "out_thickness_test/mix_vase_surface"),
        ("vase  x  oranges", "out_thickness_test/mix_vase_oranges"),
        ("surface x oranges", "out_thickness_test/mix_surface_oranges")]
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)
COMPARE = f"{OUT}/compare_mix.png"

rows = []
for label, d in RUNS:
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        print(f"  MISSING {d} -- skipped"); continue
    with open(p) as f:
        m = json.load(f)
    mean = m["mean"]
    rows.append(dict(label=label, dir=d, ssimA=mean["ssimA"], ssimB=mean["ssimB"],
                     bgA=mean["bgA"], bgB=mean["bgB"],
                     clarity=mean["ssimA"] + mean["ssimB"],
                     seed=m["best_seed"], shards=m["shards"]))

rows.sort(key=lambda r: r["clarity"], reverse=True)
print(f"\n{'pairing':22s} {'SSIM A/B':>14} {'clarity':>8} {'B_good A/B':>16}")
print("-" * 66)
for r in rows:
    print(f"{r['label']:22s} {r['ssimA']:.3f}/{r['ssimB']:.3f}   {r['clarity']:.3f}   "
          f"{r['bgA']:5.1f}%/{r['bgB']:5.1f}%")
print("\nreference (5 seeds): Sunflowers/cut 1.379 | Sunflowers/lemons 1.358 | "
      "Mona/Pearl 1.439 | Girl front/back 1.481")

n = len(rows)
fig, ax = plt.subplots(n, 3, figsize=(11, 3.6 * n), gridspec_kw={"width_ratios": [1, 1, 0.9]})
if n == 1:
    ax = ax[None, :]
for i, r in enumerate(rows):
    for ci, w in enumerate(("A", "B")):
        p = os.path.join(r["dir"], f"recon{w}.png")
        if os.path.exists(p):
            ax[i, ci].imshow(np.asarray(Image.open(p)))
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
    ax[i, 0].set_title("Wall A (recon)" if i == 0 else "", fontsize=11)
    ax[i, 1].set_title("Wall B (recon)" if i == 0 else "", fontsize=11)
    ax[i, 0].set_ylabel(f"#{i+1}  {r['label']}", fontsize=11)
    ax[i, 2].axis("off")
    txt = (f"SSIM  A {r['ssimA']:.3f}\n      B {r['ssimB']:.3f}\n  clarity {r['clarity']:.3f}\n\n"
           f"double duty\n  A {r['bgA']:.1f}%\n  B {r['bgB']:.1f}%\n\n"
           f"best seed {r['seed']}/10\n{r['shards']} shards")
    ax[i, 2].text(0.0, 0.5, txt, fontsize=11, va="center", family="monospace")
plt.suptitle("Van Gogh yellow still lifes — every pairing, tabletop-60, 10 greedy restarts each",
             fontsize=13, y=0.997)
plt.tight_layout(); plt.savefig(COMPARE, dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved {COMPARE}")
if rows:
    print(f"WINNER: {rows[0]['label']}  ->  {rows[0]['dir']}/scene.html")
