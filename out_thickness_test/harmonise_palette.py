"""
Palette harmonisation for a ShadowArt pair.

Cross-talk between the two walls is GEOMETRIC and unavoidable (every panel sits in both
lights' paths), so material serving Wall A always casts something onto Wall B. It only reads
as NOISE when the colours disagree -- e.g. the sunflowers' orange landing where the lemons
want yellow. Per corrections_note.md, genuine double duty is bounded by how colour-compatible
the two sources are, not by the solver. So instead of fighting the cross-talk with placement,
pull the two subjects' palettes toward a shared warm target: the same stray shadow then lands
as a colour the other image also wants.

Applies a per-channel gain to each subject so both subject means move a fraction `strength`
of the way to their midpoint. Structure/shading is untouched; only the overall cast shifts.
White background stays pure white.

Run:  py out_thickness_test/harmonise_palette.py [strength]      (default 0.6)
"""
import sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shadowart.targets import color as C

STRENGTH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
PAIRS = [("examples/sunflowers_clean_nobg.png", "examples/sunflowers_harm.png"),
         ("examples/lemons_clean_nobg.png",     "examples/lemons_harm.png")]
WHITE_THR = 0.90


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), float) / 255.0


imgs = [load(src) for src, _ in PAIRS]
masks = [C.subject_mask(im, WHITE_THR) for im in imgs]
means = [im[m].mean(0) for im, m in zip(imgs, masks)]
target = 0.5 * (means[0] + means[1])                 # shared warm midpoint
print(f"subject mean  sunflowers {np.round(means[0], 3)}   lemons {np.round(means[1], 3)}")
print(f"shared target {np.round(target, 3)}  (strength {STRENGTH})")

for (src, dst), im, m, mean in zip(PAIRS, imgs, masks, means):
    gain = np.where(mean > 1e-6, target / np.maximum(mean, 1e-6), 1.0)
    gain = 1.0 + STRENGTH * (gain - 1.0)             # partial move, keep artwork character
    out = im.copy()
    out[m] = np.clip(im[m] * gain[None, :], 0.0, 1.0)
    out[~m] = 1.0                                    # background stays pure white
    Image.fromarray((out * 255).astype(np.uint8)).save(dst)
    new_mean = out[m].mean(0)
    print(f"  {os.path.basename(dst):26s} gain {np.round(gain, 3)}  -> mean {np.round(new_mean, 3)}")

# side-by-side preview
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(9, 9))
for r, (src, dst) in enumerate(PAIRS):
    ax[r, 0].imshow(np.asarray(Image.open(src))); ax[r, 0].set_title(f"before — {os.path.basename(src)}", fontsize=9)
    ax[r, 1].imshow(np.asarray(Image.open(dst))); ax[r, 1].set_title(f"after — {os.path.basename(dst)}", fontsize=9)
    for c in (0, 1): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
plt.suptitle(f"Palette harmonisation (strength {STRENGTH}) — orange/yellow brought together", fontsize=12)
plt.tight_layout()
out_dir = "out_thickness_test/mona_pairs"; os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/harmonise_preview.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"saved {out_dir}/harmonise_preview.png")
