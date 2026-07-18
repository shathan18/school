"""
Cut a specific FIGURE out of a painting with point-prompted SAM.

SAM's automatic mode proposes masks for everything; here we want one named thing (the screaming
figure, Munch himself), so we prompt it with a couple of points on the subject and take its
best-scoring mask. Far more reliable than automatic masks + heuristics, and it is the same model
already downloaded.

  py out_thickness_test/extract_figure.py SRC DST x1,y1 [x2,y2 ...]
      points are FRACTIONS of width/height, measured from the TOP-LEFT of the source image.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from scipy import ndimage
import torch

SRC, DST = sys.argv[1], sys.argv[2]
PTS = [tuple(float(v) for v in a.split(",")) for a in sys.argv[3:]]
MODEL = "facebook/sam-vit-huge"

img = Image.open(SRC).convert("RGB")
W, H = img.size
points = [[[float(x * W), float(y * H)] for (x, y) in PTS]]

from transformers import SamModel, SamProcessor
proc = SamProcessor.from_pretrained(MODEL)
model = SamModel.from_pretrained(MODEL).eval()
inputs = proc(img, input_points=points, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
masks = proc.image_processor.post_process_masks(
    out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
scores = out.iou_scores[0][0].cpu().numpy()
best = int(np.argmax(scores))
m = masks[best].numpy().astype(bool)
print(f"masks {tuple(masks.shape)}  iou scores {np.round(scores, 3)}  -> using #{best}")

m = ndimage.binary_closing(m, iterations=3)
m = ndimage.binary_fill_holes(m)
lbl, n = ndimage.label(m)
if n > 1:                                   # keep the blob the prompts actually sit on
    want = {lbl[int(y * H), int(x * W)] for (x, y) in PTS}
    want.discard(0)
    m = np.isin(lbl, list(want)) if want else (lbl == np.bincount(lbl.ravel())[1:].argmax() + 1)

a = np.asarray(img, float) / 255.0
outimg = np.ones_like(a)
outimg[m] = a[m]
Image.fromarray((outimg * 255).astype(np.uint8)).save(DST)
print(f"subject {100 * m.mean():.0f}%  ->  {DST}")
