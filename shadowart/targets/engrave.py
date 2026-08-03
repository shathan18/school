"""Four-level laser-engraving tone model for clear acrylic.

The rest of the pipeline was written for COLOURED perspex stock, where a shard's tone is
whatever the supplier sells. Laser engraving inverts that: the sheets are all clear, and the
tone is produced by how hard the laser frosts the surface. That changes two things.

  * The tone values are FREE PARAMETERS, not fixed by stock. Where the grey ramp in
    `color.PERSPEX` had to take whatever acrylic existed, the engraver can be driven to any
    density in between, so *where the levels sit* becomes something to optimise rather than
    accept. `LEVEL_SWEEPS` below exists for exactly that.
  * There are FOUR tones, not five. The brief is clear / light grey / dark grey / black, so
    the alphabet is 3 engraved levels plus the unengraved sheet -- one fewer than the `noir`
    palette's four greys plus clear.

Transmittances here are per ENGRAVED FACE. Stacking still multiplies (`stack_transmit_lut`),
which is the physical truth: two frosted layers in series pass the product of their
transmittances, so overlapping shards go darker on their own.

CALIBRATION. The defaults are plausible, not measured. To pin them to a real machine, cut a
step wedge at the candidate power/speed settings, photograph it backlit against an
unengraved sheet of the same stock, and read the mean transmittance of each step; the ratio
to the clear patch is the number to put in `levels`. Everything downstream -- the tone
assignment, the cut files, the predicted render -- follows from these three numbers, so
they are the one place where a measurement is worth more than any amount of optimisation.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from . import color as C

# Transmittance of one engraved face, lightest first. Frosting scatters rather than absorbs,
# so even the heaviest practical engrave still passes some light -- a true 0.0 is not
# reachable on clear acrylic and should not be assumed.
DEFAULT_LEVELS: Tuple[float, float, float] = (0.72, 0.45, 0.18)

LEVEL_NAMES = ("ENG_L", "ENG_D", "ENG_K")     # light grey, dark grey, black

# Candidate allocations of the three engraved levels. The choice is a quantiser design
# problem: three cut-points on a tone axis, so the interesting question is whether they
# should sit uniformly, be pushed toward the dark end (where the eye reads the subject),
# or be spaced evenly in transmittance vs in optical density.
LEVEL_SWEEPS: Dict[str, Tuple[float, float, float]] = {
    "default":     DEFAULT_LEVELS,
    "uniform_T":   (0.75, 0.50, 0.25),        # even steps in transmittance
    "uniform_D":   (0.56, 0.32, 0.18),        # even steps in optical density (log T)
    "dark_biased": (0.60, 0.30, 0.10),        # more of the alphabet spent on the darks
    "light_biased": (0.85, 0.62, 0.35),       # gentler -- relies on stacking for the darks
    "wide":        (0.80, 0.40, 0.06),        # maximum contrast between adjacent levels
    "shallow":     (0.78, 0.60, 0.42),        # deliberately weak, as a control arm
}


def register(levels: Sequence[float] = DEFAULT_LEVELS) -> list:
    """Install the three engraved tones into the shared perspex table; return their names.

    Mutating the module-level table is what lets the engraved tones flow through the cut-file
    writers, the preview colouring and the renderer without any of them learning about
    engraving -- to all of them an engraved level is just another stock with a known
    transmittance. `display` == `transmit` because a frosted neutral looks like what it does.
    """
    if len(levels) != 3:
        raise ValueError(f"expected 3 engraved levels (clear is the 4th tone), got {len(levels)}")
    if not all(0.0 < t < 1.0 for t in levels):
        raise ValueError(f"transmittances must be in (0,1), got {tuple(levels)}")
    if not all(a > b for a, b in zip(levels, levels[1:])):
        raise ValueError(f"levels must be strictly darkening, got {tuple(levels)}")
    for name, t in zip(LEVEL_NAMES, levels):
        rgb = (float(t), float(t), float(t))
        C.PERSPEX[name] = (rgb, rgb)
    C.PALETTES["engrave"] = list(LEVEL_NAMES)
    return list(LEVEL_NAMES)


def tone_ladder(levels: Sequence[float], max_stack: int) -> list:
    """Every transmittance actually reachable, given that stacked shards multiply.

    Useful as a sanity check on a level choice: with stacking the 4-tone alphabet reaches
    many more than 4 tones, so levels that look coarse in isolation may still cover the
    range well. Returns sorted unique transmittances, brightest first.
    """
    reach = {1.0}
    for _ in range(max_stack):
        reach |= {r * t for r in reach for t in levels}
    return sorted(reach, reverse=True)
