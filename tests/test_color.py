"""CMYK colour: channel shards (C/M/Y/K only), subtractive mixing, tone by stacking, PLY."""
import math

import numpy as np

from shadowart.config.scene import FabParams, Light, Panel, Scene, SolveParams, Wall
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.targets import color as C
from shadowart.solve.decompose import fragment_shards_cmyk


def _scene():
    walls = {"A": Wall("A", "x", 0, "y", 1.2, 1.2, 0.3),
             "B": Wall("B", "y", 0, "x", 1.2, 1.2, 0.3)}
    panels = [Panel(f"A{i}", math.pi / 2, (x, 0.0), (0, 1.2), (0.1, 1.55), 0.003)
              for i, x in enumerate((0.35, 0.65, 0.95, 1.15))]
    panels += [Panel(f"B{i}", 0.0, (0.0, y), (0, 1.2), (0.1, 1.55), 0.003)
               for i, y in enumerate((0.35, 0.65, 0.95, 1.15))]
    return Scene(walls=walls,
                 lights={"A": Light("A", (2.4, 0.6, 0.06)), "B": Light("B", (0.6, 2.4, 0.06))},
                 panels=panels, source_radius=0.0, material_thickness=0.003,
                 solve=SolveParams(wall_res=(120, 120), panel_res=(100, 100), mode="partition",
                                   fragment_size=0.10, fragment_max_area=0.02,
                                   fragment_min_area=0.0006, shard_gap=0.0))


NAMES = C.palette_names(["C", "M", "Y", "K"])              # ['clear','C','M','Y','K']


def _block(res, rgb_val):
    Hn, Wn = res
    img = np.ones((Hn, Wn, 3), np.float32)
    img[Hn // 4:3 * Hn // 4, Wn // 4:3 * Wn // 4] = rgb_val
    return img


def _wall_A_from_family_A(scene, table, colorid, names):
    """Render wall A using only family-A panels (isolate from cross-talk)."""
    panel_T = C.transmit_lut(names)[colorid].copy()
    for i, p in enumerate(scene.panels):
        if primary_wall_of(scene, table, p) != "A":
            panel_T[i] = 1.0                               # clear -> no contribution
    return Renderer(scene, table).render_color_np(panel_T)["A"]


def _center(img):
    Hn, Wn = img.shape[:2]
    return img[Hn // 4 + 4:3 * Hn // 4 - 4, Wn // 4 + 4:3 * Wn // 4 - 4]


def test_channels_are_cmyk_only():
    scene = _scene(); table = build_projection_table(scene)
    tgt = {"A": _block(scene.solve.wall_res, (0, 0, 1)),   # blue = C+M
           "B": _block(scene.solve.wall_res, (0, 1, 1))}   # cyan
    _, frags, colorid, _, stacks = fragment_shards_cmyk(scene, table, tgt, names=NAMES, max_layers=2)
    assert frags and all(f["channel"] in ("C", "M", "Y", "K") for f in frags)
    assert set(np.unique(colorid)).issubset({0, 1, 2, 3, 4})
    assert max(stacks) >= 2                                # blue needs C+M -> multi-layer regions


def test_mixing_blue_from_c_plus_m():
    scene = _scene(); table = build_projection_table(scene)
    tgt = {"A": _block(scene.solve.wall_res, (0, 0, 1)),   # blue
           "B": _block(scene.solve.wall_res, (1, 1, 1))}
    _, _, colorid, _, _ = fragment_shards_cmyk(scene, table, tgt, names=NAMES, max_layers=1)
    c = _center(_wall_A_from_family_A(scene, table, colorid, NAMES))
    r, g, b = c[..., 0].mean(), c[..., 1].mean(), c[..., 2].mean()
    assert b > r and b > g and r < 0.4 and g < 0.4         # cyan+magenta -> blue-dominant


def test_stacking_darkens():
    scene = _scene(); table = build_projection_table(scene)
    tgt = {"A": _block(scene.solve.wall_res, (0, 1, 1)),   # pure cyan
           "B": _block(scene.solve.wall_res, (1, 1, 1))}
    _, _, cid1, _, _ = fragment_shards_cmyk(scene, table, tgt, names=NAMES, max_layers=1)
    _, _, cid2, _, _ = fragment_shards_cmyk(scene, table, tgt, names=NAMES, max_layers=2)
    r1 = _center(_wall_A_from_family_A(scene, table, cid1, NAMES))[..., 0].mean()
    r2 = _center(_wall_A_from_family_A(scene, table, cid2, NAMES))[..., 0].mean()
    assert r2 < r1                                         # 2 stacked cyans block red more


def test_ply_export(tmp_path):
    from shadowart.fabricate.export_ply import export_ply
    from shadowart.preview.interactive3d import pieces_from_opacity, _palette_color_of
    scene = _scene(); table = build_projection_table(scene)
    tgt = {"A": _block(scene.solve.wall_res, (0, 0, 1)),
           "B": _block(scene.solve.wall_res, (0, 1, 1))}
    op, _, colorid, _, _ = fragment_shards_cmyk(scene, table, tgt, names=NAMES, max_layers=2)
    pieces = pieces_from_opacity(scene, op)
    path = export_ply(scene, pieces, scene.material_thickness,
                      _palette_color_of(scene, colorid, NAMES), tmp_path / "s.ply")
    head = open(path).read(400)
    assert head.startswith("ply") and "property uchar red" in head
