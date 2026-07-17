"""Cross-talk metrics.

Cross-talk = the ghost a panel casts on a wall it isn't currently the primary
contributor to (`geometry.projection.primary_wall_of` -- no family label to read this
from). The renderer already folds it into each wall's prediction, so the
reconstruction loss fights it implicitly. These helpers isolate it for diagnostics/
heatmaps and for an optional explicit penalty in the loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .backend import DTYPE, DEVICE


def crosstalk_only(renderer, opacities, wall_name):
    """Darkness on `wall_name` from *non-primary* panels only (the ghost image)."""
    from ..geometry.projection import primary_wall_of
    if not torch.is_tensor(opacities):
        from .backend import to_t
        opacities = to_t(opacities)
    grids = renderer._grids[wall_name]; kerns = renderer._kernels[wall_name]
    trans = torch.ones((renderer.Hn, renderer.Wn), dtype=DTYPE, device=DEVICE)
    for pi, panel in enumerate(renderer.panels):
        if primary_wall_of(renderer.scene, renderer.table, panel) == wall_name:
            continue                          # skip primary panels for this wall
        op = opacities[pi].clamp(0, 1).view(1, 1, *opacities.shape[1:])
        c = F.grid_sample(op, grids[pi], mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        c = renderer._blur(c, kerns[pi]).clamp(0, 1)[0, 0]
        trans = trans * (1.0 - c)
    return 1.0 - trans


def crosstalk_energy(renderer, opacities):
    """Scalar cross-talk penalty: ghost darkness where the target is meant to be blank
    is handled by reconstruction; here we simply sum ghost darkness on both walls."""
    total = 0.0
    for wall_name in renderer.scene.walls:
        total = total + crosstalk_only(renderer, opacities, wall_name).mean()
    return total
