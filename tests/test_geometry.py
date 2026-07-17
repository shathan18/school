"""Geometry unit tests: homography round-trips, analytic magnification, projection."""
import math

import numpy as np

from shadowart.config.scene import Light, Panel, Wall
from shadowart.geometry import homography as H


def _wall_A():
    return Wall("A", "x", 0.0, "y", 1.2, 1.2, 0.3)


def _panel_A(x=0.6):
    # angle=pi/2 + anchor=(x,0) = the old "family A" orientation (parallel to wall A)
    return Panel("A0", math.pi / 2, (x, 0.0), (0.0, 1.2), (0.3, 1.5), 0.003)


def test_projection_matches_manual_ray():
    wall, panel = _wall_A(), _panel_A(0.6)
    L = np.array([2.4, 0.6, 0.06])
    ab = H.project_uv_to_wall(panel, L, wall, np.array([[0.6, 0.9]]))[0]
    # panel angle=pi/2, anchor=(0.6,0): local (u=0.6,v=0.9) -> world P=(0.6, 0.6, 0.9).
    # ray from L=(2.4,0.6,0.06) to wall A (x=0): t=Lx/(Lx-Px)=2.4/1.8; a=Qy=0.6;
    # b=Qz-z0 where Qz=0.06+t*(0.9-0.06).
    t = 2.4 / (2.4 - 0.6)
    assert np.isclose(ab[0], 0.6, atol=1e-6)
    assert np.isclose(ab[1], (0.06 + t * (0.9 - 0.06)) - 0.3, atol=1e-6)


def test_homography_round_trip():
    wall, panel = _wall_A(), _panel_A(0.7)
    L = np.array([2.4, 0.6, 0.06])
    Hpw = H.homography_panel_to_wall(panel, L, wall)
    Hwp = np.linalg.inv(Hpw)
    uv = np.array([[0.1, 0.4], [1.0, 1.3], [0.6, 0.9], [0.3, 1.1]])
    back = H.apply_homography(Hwp, H.apply_homography(Hpw, uv))
    assert np.allclose(back, uv, atol=1e-6)


def test_magnification_parallel_planes():
    # For a family-A panel parallel to Wall A, m == Lx/(Lx - xp) exactly.
    wall = _wall_A()
    L = np.array([2.4, 0.6, 0.06])
    for xp in (0.35, 0.7, 1.05, 1.4):
        m = H.magnification(_panel_A(xp), L, wall)
        assert np.isclose(m, 2.4 / (2.4 - xp), rtol=1e-6)


def test_near_light_magnifies_more():
    wall = _wall_A(); L = np.array([2.4, 0.6, 0.06])
    m_far = H.magnification(_panel_A(1.4), L, wall)    # near light
    m_near = H.magnification(_panel_A(0.35), L, wall)  # near wall
    assert m_far > m_near > 1.0
