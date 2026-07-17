"""Multi-run search: composite score monotonicity, palette presets, baseline equivalence."""
import math

import numpy as np

from shadowart.config.scene import FabParams, Light, Panel, Scene, SolveParams, Wall
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.targets import color as C
from shadowart.solve import decompose, search


# ---------------------------------------------------------------------------
# score_layout: higher SSIM/edge -> higher; higher cross-talk -> lower
# ---------------------------------------------------------------------------
def _acc(ssim, edge):
    d = {"mse": 0.0, "rmse": 0.0, "psnr_db": 0.0, "ssim": ssim, "edge_fidelity": edge}
    return {"A": dict(d), "B": dict(d)}


def test_score_rewards_ssim_and_edge():
    lo = search.score_layout(_acc(0.5, 0.5), None)
    hi_ssim = search.score_layout(_acc(0.7, 0.5), None)
    hi_edge = search.score_layout(_acc(0.5, 0.7), None)
    assert hi_ssim > lo
    assert hi_edge > lo


def test_score_penalises_crosstalk():
    clean = search.score_layout(_acc(0.6, 0.6), {0.3: 0.0})
    dirty = search.score_layout(_acc(0.6, 0.6), {0.3: 40.0})
    assert dirty < clean


def test_score_weights_respected():
    base = search.score_layout(_acc(0.6, 0.6), None, weights={"edge": 0.0})
    with_edge = search.score_layout(_acc(0.6, 0.6), None, weights={"edge": 1.0})
    assert with_edge > base                        # edge term now contributes


# ---------------------------------------------------------------------------
# _restart_iter: restart count and single-run defaults
# ---------------------------------------------------------------------------
def test_restart_iter_counts():
    got = list(search._restart_iter(restarts=5, time_budget=None, base_seed=10))
    assert got == [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]


def test_restart_iter_single_when_no_condition():
    assert list(search._restart_iter(restarts=1, time_budget=None, base_seed=0)) == [(0, 0)]
    assert list(search._restart_iter(restarts=None, time_budget=None, base_seed=3)) == [(0, 3)]


# ---------------------------------------------------------------------------
# palette presets
# ---------------------------------------------------------------------------
def test_palette_presets_registered():
    assert set(C.PALETTES) == {"cmyk", "muted", "noir"}
    for name, cols in C.PALETTES.items():
        for c in cols:
            assert c in C.PERSPEX, f"{name} references unknown perspex {c}"


def test_has_cmyk_channels():
    assert C.has_cmyk_channels(C.PALETTES["cmyk"])           # literal C/M/Y present
    # muted (C_MUTE/M_MUTE/Y_MUTE) and noir (greys) have no literal C/M/Y members, so the
    # overlap loop uses the direct nearest-colour quantise path for both.
    assert not C.has_cmyk_channels(C.PALETTES["muted"])
    assert not C.has_cmyk_channels(C.PALETTES["noir"])


# ---------------------------------------------------------------------------
# baseline equivalence: multi_run_shards with restarts=1 == single fragment_shards_overlap
# ---------------------------------------------------------------------------
def _mini_color_scene(diagonals=False, diagonal_frac=0.0):
    walls = {"A": Wall("A", "x", 0, "y", 1.2, 1.2, 0.3),
             "B": Wall("B", "y", 0, "x", 1.2, 1.2, 0.3)}
    panels = [Panel(f"A{i}", math.pi / 2, (x, 0.0), (0, 1.2), (0.1, 1.55), 0.003)
              for i, x in enumerate((0.35, 0.65, 0.95, 1.15))]
    panels += [Panel(f"B{i}", 0.0, (0.0, y), (0, 1.2), (0.1, 1.55), 0.003)
               for i, y in enumerate((0.35, 0.65, 0.95, 1.15))]
    if diagonals:                                          # two diagonal planes fanning the corner
        panels += [Panel("D0", math.radians(60), (0.7, 0.05), (0.0, 0.9), (0.1, 1.55), 0.003),
                   Panel("D1", math.radians(30), (0.05, 0.7), (0.0, 0.9), (0.1, 1.55), 0.003)]
    return Scene(walls=walls,
                 lights={"A": Light("A", (2.4, 0.6, 0.06)), "B": Light("B", (0.6, 2.4, 0.06))},
                 panels=panels, source_radius=0.0, material_thickness=0.003,
                 solve=SolveParams(wall_res=(96, 96), panel_res=(80, 80), seed=0,
                                   diagonal_frac=diagonal_frac),
                 fab=FabParams())


def _block(res, rgb):
    Hn, Wn = res
    img = np.ones((Hn, Wn, 3), np.float32)
    img[Hn // 4:3 * Hn // 4, Wn // 4:3 * Wn // 4] = rgb
    return img


def test_baseline_equivalence_restarts_one():
    scene = _mini_color_scene()
    table = build_projection_table(scene)
    renderer = Renderer(scene, table)
    names = C.palette_names(scene.color_palette)
    targets = {"A": _block(scene.solve.wall_res, (0, 1, 1)),
               "B": _block(scene.solve.wall_res, (1, 0, 1))}

    # single direct call (today's behaviour: damage_weight=0)
    direct = decompose.fragment_shards_overlap(
        scene, table, targets, names=names, white_thr=scene.white_threshold,
        max_stack=scene.color_max_stack, seed=0)[0]

    # multi_run_shards restarts=1, damage_weight=0 -> identical shard colour-id array
    best = search.multi_run_shards(scene, table, targets, names, renderer,
                                   restarts=1, damage_weight=0.0)
    assert best["n_runs"] == 1
    assert best["seed"] == 0
    np.testing.assert_array_equal(direct, best["stack_colorid"])


def test_remove_background_clean_case(tmp_path):
    """A dark square on a white ground -> subject isolated, coverage sane, trustworthy."""
    from PIL import Image
    from shadowart.targets.image_ops import remove_background
    a = np.ones((200, 200, 3), np.float32)
    a[60:140, 60:140] = (0.1, 0.1, 0.1)                  # centered dark subject
    src = tmp_path / "sq.png"
    Image.fromarray((a * 255).astype(np.uint8)).save(src)
    out, cov, ok = remove_background(str(src), str(tmp_path / "sq_nobg.png"))
    assert ok
    assert 0.10 <= cov <= 0.35                            # ~subject area fraction
    res = np.asarray(Image.open(out).convert("RGB"), np.float32) / 255.0
    assert res[10, 10].mean() > 0.95                      # corner cleared to white
    assert res[100, 100].mean() < 0.3                     # subject centre preserved


def test_remove_background_untrustworthy_keeps_original(tmp_path):
    """A near-uniform frame has no separable subject -> flagged, original kept unchanged."""
    from PIL import Image
    from shadowart.targets.image_ops import remove_background
    a = np.full((120, 120, 3), 0.5, np.float32)          # flat grey: no subject/background split
    src = tmp_path / "flat.png"
    Image.fromarray((a * 255).astype(np.uint8)).save(src)
    out, cov, ok = remove_background(str(src), str(tmp_path / "flat_nobg.png"))
    assert not ok                                         # coverage ~1.0 -> untrustworthy
    res = np.asarray(Image.open(out).convert("RGB"), np.float32) / 255.0
    assert abs(res.mean() - 0.5) < 0.05                  # original kept, not blanked to white


def test_multi_run_never_worse_than_first():
    scene = _mini_color_scene()
    table = build_projection_table(scene)
    renderer = Renderer(scene, table)
    names = C.palette_names(scene.color_palette)
    targets = {"A": _block(scene.solve.wall_res, (0, 1, 1)),
               "B": _block(scene.solve.wall_res, (1, 0, 1))}
    first = search.multi_run_shards(scene, table, targets, names, renderer,
                                    restarts=1, damage_weight=0.5)
    many = search.multi_run_shards(scene, table, targets, names, renderer,
                                   restarts=4, damage_weight=0.5)
    assert many["n_runs"] == 4
    assert many["score"] >= first["score"] - 1e-9        # best-of-N never below the first run


def _diag_material_share(scene, stack_colorid):
    """Fraction of active shard pixels that sit on diagonal panels."""
    import math as _m
    tot = diag = 0
    for i, p in enumerate(scene.panels):
        px = int((stack_colorid[:, i] > 0).sum())
        tot += px
        if p.is_diagonal():
            diag += px
    return diag / max(tot, 1)


def test_diagonal_frac_puts_material_on_diagonals():
    """diagonal_frac=0 leaves diagonals empty (the greedy avoids them); >0 forces material
    onto them so the piece is non-trivial."""
    from shadowart.solve import decompose
    names = C.palette_names(["C", "M", "Y", "K"])
    targets = {"A": _block((96, 96), (0, 1, 1)), "B": _block((96, 96), (1, 0, 1))}

    s0 = _mini_color_scene(diagonals=True, diagonal_frac=0.0)
    t0 = build_projection_table(s0)
    sc0 = decompose.fragment_shards_overlap(s0, t0, targets, names=names,
                                            max_stack=s0.color_max_stack, seed=0,
                                            damage_weight=0.5)[0]
    assert _diag_material_share(s0, sc0) < 0.02           # greedy leaves diagonals ~empty

    s1 = _mini_color_scene(diagonals=True, diagonal_frac=0.3)
    t1 = build_projection_table(s1)
    sc1 = decompose.fragment_shards_overlap(s1, t1, targets, names=names,
                                            max_stack=s1.color_max_stack, seed=0,
                                            damage_weight=0.5, diagonal_frac=0.3)[0]
    assert _diag_material_share(s1, sc1) > 0.15           # quota honoured: real material on diagonals


def test_is_diagonal_classification():
    from shadowart.config.scene import Panel
    ax_a = Panel("A", math.pi / 2, (1.0, 0.0), (0, 1), (0.1, 1.5), 0.003)
    ax_b = Panel("B", 0.0, (0.0, 1.0), (0, 1), (0.1, 1.5), 0.003)
    diag = Panel("D", math.radians(45), (0.5, 0.5), (0, 1), (0.1, 1.5), 0.003)
    assert not ax_a.is_diagonal() and not ax_b.is_diagonal()
    assert diag.is_diagonal()
