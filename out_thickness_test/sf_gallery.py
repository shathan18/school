"""
Compare the candidate partners for Van Gogh's Sunflowers (Wall A fixed = sunflowers_clean_nobg).
One grid (row per candidate: Wall-A recon | Wall-B recon | metrics) plus a ranked table.

Ranked by CLARITY (SSIM_A + SSIM_B) rather than raw double duty: double duty trades against
visible cross-talk staining, which is the very thing we're trying to avoid here.

Run AFTER the pair runs finish:  py out_thickness_test/sf_gallery.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

RUNS = [("Cut sunflowers (same palette, objects)", "out_thickness_test/sf_cut"),
        ("2nd Sunflowers vase version",            "out_thickness_test/sf_v2"),
        ("Lemon bowl",                             "out_thickness_test/sunflowers_lemons_both"),
        ("Irises (palette-clash control)",         "out_thickness_test/sf_irises")]
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)
COMPARE = f"{OUT}/compare_sf.png"

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
print(f"\n{'candidate':40s} {'SSIM A/B':>14} {'clarity':>8} {'B_good A/B':>16}")
print("-" * 84)
for r in rows:
    print(f"{r['label']:40s} {r['ssimA']:.3f}/{r['ssimB']:.3f}   {r['clarity']:.3f}   "
          f"{r['bgA']:5.1f}%/{r['bgB']:5.1f}%")
print("\nreference  Girl front/back (identical palette): clarity 1.481 | Mona/Pearl: 1.439")

n = len(rows)
fig, ax = plt.subplots(n, 3, figsize=(11, 3.5 * n), gridspec_kw={"width_ratios": [1, 1, 0.9]})
if n == 1:
    ax = ax[None, :]
for i, r in enumerate(rows):
    for ci, w in enumerate(("A", "B")):
        p = os.path.join(r["dir"], f"recon{w}.png")
        if os.path.exists(p):
            ax[i, ci].imshow(np.asarray(Image.open(p)))
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
    ax[i, 0].set_title("Wall A — Sunflowers (recon)" if i == 0 else "", fontsize=11)
    ax[i, 1].set_title("Wall B — partner (recon)" if i == 0 else "", fontsize=11)
    ax[i, 0].set_ylabel(f"#{i+1}  {r['label']}", fontsize=10)
    ax[i, 2].axis("off")
    txt = (f"SSIM  A {r['ssimA']:.3f}\n      B {r['ssimB']:.3f}\n  clarity {r['clarity']:.3f}\n\n"
           f"double duty\n  A {r['bgA']:.1f}%\n  B {r['bgB']:.1f}%\n\n"
           f"best seed {r['seed']}\n{r['shards']} shards")
    ax[i, 2].text(0.0, 0.5, txt, fontsize=11, va="center", family="monospace")
plt.suptitle("Sunflowers partner comparison — tabletop-60, 5 seeds each (ranked by clarity)",
             fontsize=13, y=0.997)
plt.tight_layout(); plt.savefig(COMPARE, dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved {COMPARE}")
if rows:
    print(f"WINNER: {rows[0]['label']}  ->  {rows[0]['dir']}/scene.html")
