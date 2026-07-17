"""Shared test fixtures: build small scenes without touching YAML."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowart.config.scene import (          # noqa: E402
    FabParams, Light, Panel, Scene, SolveParams, Wall,
)


def _walls():
    return {
        "A": Wall("A", "x", 0.0, "y", 1.2, 1.2, 0.3),
        "B": Wall("B", "y", 0.0, "x", 1.2, 1.2, 0.3),
    }


@pytest.fixture
def mini_scene():
    """One panel per family, tiny rasters -> fast render tests."""
    # Panel is (name, angle[rad from +x], anchor[(x,y) at u=0], u_range, v_range, thickness).
    # angle=pi/2 + anchor=(0.6, 0) reproduces the old "family A" orientation (parallel to
    # wall A); angle=0 + anchor=(0, 0.6) reproduces "family B".
    panels = [
        Panel("A0", math.pi / 2, (0.6, 0.0), (0.0, 1.2), (0.3, 1.5), 0.003),
        Panel("B0", 0.0, (0.0, 0.6), (0.0, 1.2), (0.3, 1.5), 0.003),
    ]
    return Scene(
        walls=_walls(),
        lights={"A": Light("A", (2.4, 0.6, 0.06)), "B": Light("B", (0.6, 2.4, 0.06))},
        panels=panels, source_radius=0.0, material_thickness=0.003,
        solve=SolveParams(wall_res=(64, 64), panel_res=(48, 48), iters=5),
        fab=FabParams(),
    )
