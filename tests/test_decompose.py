"""Fragmenter: reconstructs the target, shatters into many capped-size shards, no collisions."""
import math

import numpy as np

from shadowart.config.scene import FabParams, Light, Panel, Scene, SolveParams, Wall
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve.decompose import fragment_shards, count_collisions


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
                                   fragment_min_area=0.0006, shard_gap=0.0),
                 fab=FabParams())


def _block_targets(res):
    Hn, Wn = res
    t = {w: np.zeros((Hn, Wn), np.float32) for w in ("A", "B")}
    for w in ("A", "B"):
        t[w][Hn // 4:3 * Hn // 4, Wn // 4:3 * Wn // 4] = 1.0
    return t


def _primary_iou(scene, renderer, table, op, tgt, family):
    opf = op.copy()
    for i, p in enumerate(scene.panels):
        if primary_wall_of(scene, table, p) != family:
            opf[i] = 0.0
    pr = renderer.render_np(opf)[family] >= 0.5
    b = tgt[family] >= 0.5
    return (pr & b).sum() / max((pr | b).sum(), 1)


def test_reconstructs_target():
    scene = _scene(); table = build_projection_table(scene)
    tgt = _block_targets(scene.solve.wall_res)
    op, frags, _ = fragment_shards(scene, table, tgt)
    renderer = Renderer(scene, table)
    for w in ("A", "B"):                                  # shards-only reconstruction
        assert _primary_iou(scene, renderer, table, op, tgt, w) > 0.92


def test_shatters_into_many_capped_shards():
    scene = _scene(); table = build_projection_table(scene)
    op, frags, _ = fragment_shards(scene, table, _block_targets(scene.solve.wall_res))
    for fam in ("A", "B"):
        ff = [f for f in frags if f["wall"] == fam]
        panels_used = {f["panel"] for f in ff}
        n_panels = sum(1 for p in scene.panels if primary_wall_of(scene, table, p) == fam)
        assert len(ff) > n_panels                        # more shards than panels -> fragmented
        assert len(panels_used) >= 2                    # spread across depths
        cap_mm2 = scene.solve.fragment_max_area * 1e6
        assert max(f["wall_mm2"] for f in ff) <= cap_mm2 * 1.3   # size cap honoured


def test_no_collisions_after_resolve():
    scene = _scene(); table = build_projection_table(scene)
    op, _, resolved = fragment_shards(scene, table, _block_targets(scene.solve.wall_res))
    assert count_collisions(scene, op) == 0
    assert resolved >= 0
