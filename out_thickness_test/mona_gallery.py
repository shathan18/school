"""
Assemble the Mona-pair comparison from the three deliverable runs (mona_prado / mona_stjohn
/ mona_pearl): one grid image (row per candidate: Wall-A recon | Wall-B recon | metrics) plus
a ranked table by combined double-duty and SSIM. Reads each run's metrics.json + reconA/B.png.

Run AFTER the three girl_tabletop.py runs finish.
Run:  py out_thickness_test/mona_gallery.py
"""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

# optional argv: a suffix on the run dirs, e.g. "_nobg" -> reads mona_prado_nobg etc.
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
RUNS = [("Prado twin (the Mona itself)",   f"out_thickness_test/mona_prado{SUF}"),
        ("Pearl girl (Mona of the North)", f"out_thickness_test/mona_pearl{SUF}"),
        ("Van Gogh self-portrait",         f"out_thickness_test/mona_vangogh{SUF}"),
        ("Klimt Woman in Gold",            f"out_thickness_test/mona_klimt{SUF}")]
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)
COMPARE = f"{OUT}/compare{SUF}.png"


def load_run(d):
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)
    return m


rows = []
for label, d in RUNS:
    if not os.path.exists(os.path.join(d, "metrics.json")):
        print(f"  MISSING {d} -- skipped"); continue
    m = load_run(d)
    mean = m["mean"]
    rows.append(dict(label=label, dir=d, ssimA=mean["ssimA"], ssimB=mean["ssimB"],
                     bgA=mean["bgA"], bgB=mean["bgB"], combined=mean["bgA"] + mean["bgB"],
                     clarity=mean["ssimA"] + mean["ssimB"],
                     seed=m["best_seed"], shards=m["shards"]))

# Fame is a given for all four; rank on CLARITY (both walls readable, low noise) since raw
# double duty trades against noise. Double duty still shown as a secondary column.
rows.sort(key=lambda r: r["clarity"], reverse=True)

print(f"\n{'candidate':32s} {'SSIM A/B':>14} {'clarity':>8} {'B_good A/B':>16}")
print("-" * 76)
for r in rows:
    print(f"{r['label']:32s} {r['ssimA']:.3f}/{r['ssimB']:.3f}   {r['clarity']:.3f}   "
          f"{r['bgA']:5.1f}%/{r['bgB']:5.1f}%")
print("\nBASELINE girl_tabletop60:              0.721/0.760   1.481   20.4%/ 7.3%")

# grid: row per candidate, cols = Wall A recon, Wall B recon
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
    ax[i, 0].set_title("Wall A — Mona (recon)" if i == 0 else "", fontsize=11)
    ax[i, 1].set_title("Wall B — partner (recon)" if i == 0 else "", fontsize=11)
    ax[i, 0].set_ylabel(f"#{i+1}  {r['label']}", fontsize=11)
    ax[i, 2].axis("off")
    txt = (f"SSIM  A {r['ssimA']:.3f}\n      B {r['ssimB']:.3f}\n  clarity {r['clarity']:.3f}\n\n"
           f"double duty\n  A {r['bgA']:.1f}%\n  B {r['bgB']:.1f}%\n\n"
           f"best seed {r['seed']}\n{r['shards']} shards")
    ax[i, 2].text(0.0, 0.5, txt, fontsize=11, va="center", family="monospace")
plt.suptitle("Mona Lisa partner comparison — tabletop-60, 5 seeds each (ranked by clarity = SSIM_A+SSIM_B)",
             fontsize=13, y=0.997)
plt.tight_layout(); plt.savefig(COMPARE, dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved {COMPARE}")
if rows:
    print(f"WINNER: {rows[0]['label']}  ->  {rows[0]['dir']}/scene.html")
