"""Joint two-wall optimisation of the per-panel opacity fields (PyTorch/Adam).

We optimise logits (sigmoid -> opacity in [0,1]) so both wall reconstructions improve
together; cross-talk is fought implicitly by the opposite-wall term and, optionally,
explicitly. This is inverse rendering: the same renderer used for preview is the
objective. Always re-score the *binarised* result afterwards (see raster2vec) — the
continuous optimum flatters you.
"""
from __future__ import annotations

import numpy as np
import torch

from ..forward.backend import DTYPE, DEVICE, to_t, to_np
from ..forward import crosstalk as ct
from . import losses


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def solve(scene, renderer, targets, init_opacity=None, verbose=True, log_every=50):
    sp = scene.solve
    torch.manual_seed(sp.seed)
    if init_opacity is None:
        P, (Hp, Wp) = len(scene.panels), sp.panel_res
        init_opacity = np.full((P, Hp, Wp), 0.05, dtype=np.float32)

    logits = torch.tensor(_logit(init_opacity), dtype=DTYPE, device=DEVICE,
                          requires_grad=True)
    tgt = {k: to_t(v) for k, v in targets.items()}
    opt = torch.optim.Adam([logits], lr=sp.lr)

    history = []
    for it in range(sp.iters):
        opt.zero_grad()
        op = torch.sigmoid(logits)
        pred = renderer.render(op)
        loss = losses.reconstruction(pred, tgt)
        loss = loss + sp.lambda_sparsity * losses.sparsity(op)
        loss = loss + sp.lambda_tv * losses.total_variation(op)
        if sp.lambda_crosstalk > 0:
            loss = loss + sp.lambda_crosstalk * ct.crosstalk_energy(renderer, op)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if verbose and (it % log_every == 0 or it == sp.iters - 1):
            with torch.no_grad():
                errs = "  ".join(f"mse{k} {float(((pred[k] - tgt[k]) ** 2).mean()):.5f}"
                                 for k in sorted(pred))
            print(f"  iter {it:4d}  loss {float(loss):.5f}  {errs}")

    with torch.no_grad():
        op = torch.sigmoid(logits)
    return to_np(op), history
