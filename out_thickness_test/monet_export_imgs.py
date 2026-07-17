"""Export clean per-wall source/reconstruction PNGs (seed 3) for the artifact comparison sliders."""
import sys, dataclasses, os
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from PIL import Image
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C

A_IMG = "examples/monet_day_edited.jpeg"; B_IMG = "examples/monet_dusk_edited.jpeg"
OUT = "out_thickness_test/monet_final"; os.makedirs(OUT, exist_ok=True)
SEED = 3
scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res; names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
           "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                targets=targets, seed=SEED, angle_deg_range=(5, 85))
ts = dataclasses.replace(scene, panels=panels, solve=scene.solve)
table = build_projection_table(ts); renderer = Renderer(ts, table)
sc, op, fr, res, sd, bs, si = decompose.fragment_shards_overlap(
    ts, table, targets, names=names, white_thr=ts.white_threshold,
    max_stack=ts.color_max_stack, seed=SEED, damage_weight=0.5, credit_weight=0.5, match_tol=0.30)
pred = renderer.render_color_np(C.stack_transmit_lut(names, sc, si))

def save(arr, path, scale=3):
    a = np.clip(np.flipud(arr), 0, 1)            # origin='lower' -> upright PNG
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    print("saved", path, im.size)

for w in ("A", "B"):
    save(targets[w], f"{OUT}/src{w}.png")
    save(pred[w], f"{OUT}/recon{w}.png")
