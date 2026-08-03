"""The differentiable forward renderer — the single source of truth.

It is simultaneously (a) the physics simulator, (b) the optimisation objective, and
(c) the preview. Given an opacity field per panel it predicts the darkness image on
Wall A and Wall B:

    for each wall (lit by its own light):
        warp each panel's opacity to the wall via the cached homography (grid_sample)
        blur it by that panel's penumbra PSF (finite source)
        composite by transmittance:  T = prod_p (1 - alpha_p) ,  darkness = 1 - T

Because every panel is warped onto *both* walls, cross-talk (a family-A panel casting
a skewed ghost onto Wall B) appears automatically in the prediction — so a plain
two-wall reconstruction loss already fights it.

Opacities are a tensor [P, Hp, Wp] in [0,1] (P panels, in scene.panels order).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..geometry import homography as H
from ..geometry import psf
from .backend import DTYPE, DEVICE, to_t, to_np


class Renderer:
    def __init__(self, scene, table):
        self.scene = scene
        self.table = table
        self.Hn, self.Wn = scene.solve.wall_res          # rows(height/z), cols(width)
        self.panels = scene.panels
        self.P = len(self.panels)
        self.panel_res = tuple(scene.solve.panel_res)
        self._grids = {}          # wall -> [P] tensors [1,Hn,Wn,2]
        self._kernels = {}        # wall -> [P] (kx, ky, rx, ry) or None
        for wall_name, wall in scene.walls.items():
            grids, kerns = [], []
            aa, bb = self._wall_grid_m(wall)             # [Hn,Wn] each, metres
            wall_pts = np.stack([aa, bb], axis=-1)       # [Hn,Wn,2]
            for panel in self.panels:
                m = table[(panel.name, wall_name)]
                uv = H.apply_homography(m.H_wp, wall_pts)  # wall metres -> panel metres
                gx = 2.0 * (uv[..., 0] - panel.u_range[0]) / max(panel.u_size, 1e-9) - 1.0
                gy = 2.0 * (uv[..., 1] - panel.v_range[0]) / max(panel.v_size, 1e-9) - 1.0
                grid = np.stack([gx, gy], axis=-1)[None].astype(np.float32)  # [1,Hn,Wn,2]
                grids.append(torch.as_tensor(grid, device=DEVICE))
                kerns.append(self._make_kernel(m.penumbra_sigma_m, wall))
            self._grids[wall_name] = grids
            self._kernels[wall_name] = kerns

    # -- geometry setup ----------------------------------------------------
    def _wall_grid_m(self, wall):
        a = (np.arange(self.Wn) + 0.5) / self.Wn * wall.width     # width coord (u/a)
        b = (np.arange(self.Hn) + 0.5) / self.Hn * wall.height    # height coord (v/b)
        aa = np.broadcast_to(a[None, :], (self.Hn, self.Wn)).copy()
        bb = np.broadcast_to(b[:, None], (self.Hn, self.Wn)).copy()
        return aa, bb

    def _make_kernel(self, sigma_m, wall):
        sx = psf.sigma_m_to_px(sigma_m, wall.width, self.Wn)
        sy = psf.sigma_m_to_px(sigma_m, wall.height, self.Hn)
        if max(sx, sy) < 0.3:
            return None
        kx = psf.gaussian_kernel1d(sx); ky = psf.gaussian_kernel1d(sy)
        kx_t = torch.as_tensor(kx, device=DEVICE).view(1, 1, 1, -1)
        ky_t = torch.as_tensor(ky, device=DEVICE).view(1, 1, -1, 1)
        return (kx_t, ky_t, (kx.size - 1) // 2, (ky.size - 1) // 2)

    @staticmethod
    def _blur(x, kernel):
        if kernel is None:
            return x
        kx, ky, rx, ry = kernel
        x = F.conv2d(F.pad(x, (rx, rx, 0, 0), mode="reflect"), kx)
        x = F.conv2d(F.pad(x, (0, 0, ry, ry), mode="reflect"), ky)
        return x

    # -- forward -----------------------------------------------------------
    def render(self, opacities):
        """opacities: tensor [P,Hp,Wp] in [0,1].  Returns {wall_name: darkness}."""
        opacities = to_t(opacities)
        out = {}
        for wall_name in self.scene.walls:
            grids = self._grids[wall_name]; kerns = self._kernels[wall_name]
            trans = torch.ones((self.Hn, self.Wn), dtype=DTYPE, device=DEVICE)
            for pi in range(self.P):
                op = opacities[pi].clamp(0, 1).view(1, 1, *opacities.shape[1:])
                c = F.grid_sample(op, grids[pi], mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
                c = self._blur(c, kerns[pi]).clamp(0, 1)[0, 0]
                trans = trans * (1.0 - c)
            out[wall_name] = 1.0 - trans
        return out

    def render_np(self, opacities):
        with torch.no_grad():
            d = self.render(opacities)
        return {k: v.detach().cpu().numpy() for k, v in d.items()}

    # -- colour forward (subtractive) --------------------------------------
    @staticmethod
    def _blur3(x, kernel):
        """Blur a [1,3,H,W] tensor per channel with a 1-channel separable kernel."""
        if kernel is None:
            return x
        b = Renderer._blur(x[0].unsqueeze(1), kernel)     # [3,1,H,W]
        return b.squeeze(1).unsqueeze(0)

    def render_color_np(self, panel_T):
        """panel_T: [P,Hp,Wp,3] transmittance in [0,1] (clear=1). Returns {A,B} RGB [Hn,Wn,3].

        White light through the stack of coloured shards: wall = product of transmittances
        along each ray (subtractive). Outside a panel -> transmittance 1 (handled via the
        1-T zeros-padding trick)."""
        panel_T = to_t(panel_T)
        out = {}
        with torch.no_grad():
            for wall_name in self.scene.walls:
                grids, kerns = self._grids[wall_name], self._kernels[wall_name]
                wallT = torch.ones((3, self.Hn, self.Wn), dtype=DTYPE, device=DEVICE)
                for pi in range(self.P):
                    A = (1.0 - panel_T[pi].clamp(0, 1)).permute(2, 0, 1).unsqueeze(0)
                    Aw = F.grid_sample(A, grids[pi], mode="bilinear",
                                       padding_mode="zeros", align_corners=False)
                    Aw = self._blur3(Aw, kerns[pi]).clamp(0, 1)[0]
                    wallT = wallT * (1.0 - Aw)
                out[wall_name] = wallT.permute(1, 2, 0)
        return {k: to_np(v) for k, v in out.items()}
