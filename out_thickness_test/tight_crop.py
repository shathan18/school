"""
Reframe the sources so the shard budget lands on what identifies them.

`load_color_target` already scales the subject to 92% of the wall -- so the PAINTING fills the
wall, but inside a head-and-shoulders composition the FACE is only ~25-30% of it, and so gets
~25-30% of the shards. That, not the solver, is why features never resolve. Cropping to the head
puts the face at ~70% of the wall: ~3x the shards on the eyes/nose/mouth with no change to the
physics.

Asymmetric on purpose:
  portrait -> crop HARD (its icon is the face, which is what's being lost)
  vase     -> tighten GENTLY (its icon is the arrangement, which already reads; cropping to a few
              blooms would resolve each flower but make the painting less identifiable)

Run:  py out_thickness_test/tight_crop.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from shadowart.targets import color as C
import semseg as SS

WHITE_THR = 0.90
PORTRAITS = [("examples/vg_p_yellow_nobg.png", "examples/vg_p_yellow_head.png"),
             ("examples/vg_p_gemini_nobg.png", "examples/vg_p_gemini_head.png"),
             ("examples/vg_p_gold_nobg.png",   "examples/vg_p_gold_head.png")]
VASE = ("examples/sunflowers_clean_nobg.png", "examples/sunflowers_tight.png")
HEAD_PARTS = {"skin", "hair", "l_ear", "r_ear", "nose", "l_eye", "r_eye", "l_brow", "r_brow",
              "mouth", "u_lip", "l_lip", "eye_g", "hat"}


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), float) / 255.0


def crop_to(im, mask, margin=0.12):
    ys, xs = np.where(mask)
    if not len(ys):
        return im
    H, W = im.shape[:2]
    my, mx = int(margin * (ys.max() - ys.min())), int(margin * (xs.max() - xs.min()))
    y0, y1 = max(0, ys.min() - my), min(H, ys.max() + my)
    x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + mx)
    return im[y0:y1, x0:x1]


rows = []
for src, dst in PORTRAITS:
    if not os.path.exists(src):
        print(f"  missing {src}"); continue
    im = load(src)
    subj = C.subject_mask(im, WHITE_THR)
    labels, info = SS.to_regions(im, subj, kind="face")
    head_ids = [rid for rid, n in info.items() if n in HEAD_PARTS]
    head = np.isin(labels, head_ids) if head_ids else subj
    head = ndimage.binary_closing(head, iterations=4)
    if head.sum() < 0.02 * subj.sum():                    # face parse failed -> keep full frame
        print(f"  {os.path.basename(src)}: head parse too small, skipped")
        continue
    out = crop_to(im, head)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst)
    before = float(head.sum()) / max(float(subj.sum()), 1)
    after_subj = C.subject_mask(out, WHITE_THR)
    print(f"  {os.path.basename(dst):26s} head was {before*100:.0f}% of subject -> "
          f"crop {out.shape[1]}x{out.shape[0]}")
    rows.append((os.path.basename(src), im, out))

# vase: tighten by dropping the smallest outlying object, then a small margin
src, dst = VASE
im = load(src)
subj = C.subject_mask(im, WHITE_THR)
lbl, n = ndimage.label(subj)
if n > 1:
    counts = np.bincount(lbl.ravel()); counts[0] = 0
    keep = counts >= 0.06 * counts.max()                  # drop stray specks / outlier blooms
    core = keep[lbl]
else:
    core = subj
out = crop_to(im, core, margin=0.04)
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst)
print(f"  {os.path.basename(dst):26s} {im.shape[1]}x{im.shape[0]} -> {out.shape[1]}x{out.shape[0]}")
rows.append((os.path.basename(src), im, out))

fig, ax = plt.subplots(len(rows), 2, figsize=(8.5, 4.2 * len(rows)))
if len(rows) == 1:
    ax = ax[None, :]
for i, (name, a, b) in enumerate(rows):
    ax[i, 0].imshow(np.clip(a, 0, 1)); ax[i, 1].imshow(np.clip(b, 0, 1))
    for j in (0, 1):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[i, 0].set_ylabel(name, fontsize=9)
    if i == 0:
        ax[i, 0].set_title("BEFORE (as run)", fontweight="bold")
        ax[i, 1].set_title("AFTER (reframed)", fontweight="bold")
plt.suptitle("Reframing: put the shard budget on what identifies the painting", fontsize=13)
plt.tight_layout()
plt.savefig("out_thickness_test/mona_pairs/tight_crop.png", dpi=100, bbox_inches="tight")
print("\nsaved out_thickness_test/mona_pairs/tight_crop.png")
