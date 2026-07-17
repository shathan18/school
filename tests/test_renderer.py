"""Forward-renderer sanity + differentiability + a short end-to-end solve."""
import numpy as np
import torch

from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve.optimizer import solve


def test_opaque_panel_casts_shadow(mini_scene):
    table = build_projection_table(mini_scene)
    r = Renderer(mini_scene, table)
    P, Hp, Wp = len(mini_scene.panels), *mini_scene.solve.panel_res
    op = np.zeros((P, Hp, Wp), np.float32)
    op[0] = 1.0                                   # family-A panel fully opaque
    out = r.render_np(op)
    assert out["A"].shape == tuple(mini_scene.solve.wall_res)
    assert 0.0 <= out["A"].min() and out["A"].max() <= 1.0 + 1e-5
    assert out["A"].max() > 0.9                   # a real shadow appears on Wall A
    assert out["A"].mean() > 0.05


def test_empty_scene_is_bright(mini_scene):
    table = build_projection_table(mini_scene)
    r = Renderer(mini_scene, table)
    P, Hp, Wp = len(mini_scene.panels), *mini_scene.solve.panel_res
    out = r.render_np(np.zeros((P, Hp, Wp), np.float32))
    assert out["A"].max() < 1e-5 and out["B"].max() < 1e-5


def test_render_is_differentiable(mini_scene):
    table = build_projection_table(mini_scene)
    r = Renderer(mini_scene, table)
    P, Hp, Wp = len(mini_scene.panels), *mini_scene.solve.panel_res
    op = torch.full((P, Hp, Wp), 0.3, requires_grad=True)
    loss = sum(v.mean() for v in r.render(op).values())
    loss.backward()
    assert op.grad is not None and torch.isfinite(op.grad).all()
    assert op.grad.abs().sum() > 0


def test_short_solve_reduces_error(mini_scene):
    table = build_projection_table(mini_scene)
    r = Renderer(mini_scene, table)
    Hn, Wn = mini_scene.solve.wall_res
    targets = {"A": np.zeros((Hn, Wn), np.float32), "B": np.zeros((Hn, Wn), np.float32)}
    targets["A"][Hn // 4:3 * Hn // 4, Wn // 4:3 * Wn // 4] = 1.0   # a block on Wall A
    _, hist = solve(mini_scene, r, targets, verbose=False)
    assert hist[-1] < hist[0]                     # optimisation makes progress
