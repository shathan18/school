"""Turntable rig <-> N-wall equivalence.

The turntable's whole premise is a change of frame: rotating the ASSEMBLY by +theta
under one fixed lamp and wall must produce exactly the image that the UNROTATED assembly
produces against the rig rotated by -theta. `geometry.turntable` relies on that identity
to turn one physical rig into N simultaneous (Wall, Light) pairs for the solver, so it is
worth asserting numerically rather than trusting the derivation.
"""
import math

import numpy as np
import pytest

from shadowart.config.scene import (
    FabParams, Light, Panel, Scene, SolveParams, TurntableSpec, Wall,
)
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table
from shadowart.geometry import turntable as TT


SPEC = TurntableSpec(
    stops_deg=(15.0, 135.0, 255.0),
    center=(1.0, 1.0),                 # deliberately NOT the origin: catches a rig that
    view_azimuth_deg=25.0,             # only works for a turntable centred at (0,0)
    wall_distance=0.9,
    lamp_distance=0.8,
    lamp_height=0.35,
    wall_width=0.6, wall_height=0.6, wall_z0=0.05,
)


def _assembly():
    """A small egg-crate-ish assembly straddling the turntable centre."""
    cx, cy = SPEC.center
    panels = []
    for i, off in enumerate((-0.06, 0.0, 0.06)):
        panels.append(Panel(f"P{i}", math.pi / 2, (cx + off, cy - 0.09),
                            (0.0, 0.18), (0.05, 0.35), 0.003))
        panels.append(Panel(f"Q{i}", 0.0, (cx - 0.09, cy + off),
                            (0.0, 0.18), (0.05, 0.35), 0.003))
    return panels


def _scene(walls, lights, panels, source_radius=0.004):
    return Scene(walls=walls, lights=lights, panels=panels,
                 source_radius=source_radius, material_thickness=0.003,
                 solve=SolveParams(wall_res=(96, 96), panel_res=(64, 64)),
                 fab=FabParams())


def _render(scene, opacity):
    return Renderer(scene, build_projection_table(scene)).render_np(opacity)


@pytest.fixture
def opacity():
    rng = np.random.default_rng(0)
    return rng.random((6, 64, 64)).astype(np.float32)


def test_rotating_the_assembly_equals_rotating_the_rig(opacity):
    """The identity the whole turntable model rests on, checked per stop."""
    lamp, wall_c, wall_n_deg = TT.rig_geometry(SPEC)
    walls, lights = TT.build_walls_and_lights(SPEC)
    names = TT.stop_names(SPEC)
    panels = _assembly()

    for name, theta in zip(names, SPEC.stops_deg):
        # (a) solver frame: assembly at rest, rig rotated by -theta.
        rig = _scene({name: walls[name]}, {name: lights[name]}, panels)
        got = _render(rig, opacity)[name]

        # (b) physical frame: rig at rest, assembly physically turned by +theta.
        t = math.radians(wall_n_deg)
        n_hat = np.array([math.cos(t), math.sin(t)])
        u_hat = np.array([-math.sin(t), math.cos(t)])
        fixed_wall = Wall(name=name, plane="n", offset=float(n_hat @ wall_c[:2]),
                          width_axis="", width=SPEC.wall_width, height=SPEC.wall_height,
                          z0=SPEC.wall_z0, normal_deg=wall_n_deg,
                          lateral_offset=float(u_hat @ wall_c[:2]))
        fixed_light = Light(name=name, pos=tuple(map(float, lamp)))
        turned = _render(_scene({name: fixed_wall}, {name: fixed_light},
                                TT.rotate_panels(panels, SPEC.center, theta)), opacity)[name]

        assert np.abs(got - turned).max() < 1e-5, f"stop {name} ({theta} deg) disagrees"


def test_stops_actually_differ(opacity):
    """Guard against a rig that is 'equivalent' only because it is degenerate: three
    stops 120 deg apart must render three genuinely different images."""
    walls, lights = TT.build_walls_and_lights(SPEC)
    pred = _render(_scene(walls, lights, _assembly()), opacity)
    names = TT.stop_names(SPEC)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert np.abs(pred[names[i]] - pred[names[j]]).max() > 1e-3


def test_zero_stop_is_the_physical_rig():
    """A stop at 0 deg must reproduce the untouched physical lamp and wall."""
    spec = TurntableSpec(stops_deg=(0.0,), center=(1.0, 1.0), view_azimuth_deg=25.0,
                         wall_distance=0.9, lamp_distance=0.8, lamp_height=0.35,
                         wall_width=0.6, wall_height=0.6, wall_z0=0.05)
    lamp, wall_c, wall_n_deg = TT.rig_geometry(spec)
    walls, lights = TT.build_walls_and_lights(spec)
    w, l = walls["V0"], lights["V0"]
    assert w.normal_deg == pytest.approx(wall_n_deg)
    assert np.allclose(l.xyz, lamp)
    # The imaged rectangle must be centred on the physical wall point.
    centre = w.origin + w.axis_u * (w.width / 2.0)
    assert np.allclose(centre[:2], wall_c[:2], atol=1e-9)


def test_footprint_is_measured_on_extents_not_centres():
    """The 30x30 cm limit is on occupied floor area; a centre-based box understates it."""
    panels = _assembly()
    wx, wy = TT.footprint_size(panels)
    assert wx == pytest.approx(0.18, abs=1e-9)     # 0.12 span of anchors + 0.18 length
    assert wy == pytest.approx(0.18, abs=1e-9)
    # The piece TURNS, so the real constraint is the swept circle, which is larger.
    assert 2 * TT.swept_radius(panels, SPEC.center) > max(wx, wy)


def test_names_must_match_stops():
    bad = TurntableSpec(stops_deg=(0.0, 120.0), names=("only_one",),
                        wall_distance=0.9, lamp_distance=0.8,
                        wall_width=0.6, wall_height=0.6)
    with pytest.raises(ValueError, match="one-to-one"):
        TT.stop_names(bad)
