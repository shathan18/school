"""Scene description dataclasses.

Coordinate system (all lengths in metres):
    - The corner where the two walls meet is the vertical Z axis.
    - Wall A is the plane x = 0 (its image spans y (width) and z (height)).
    - Wall B is the plane y = 0 (its image spans x (width) and z (height)).
    - Floor is z = 0.  Room interior is x>0, y>0, z>0.
    - Light A sits at x>0 and casts panel shadows onto Wall A (rays travel toward -x).
    - Light B sits at y>0 and casts panel shadows onto Wall B (rays travel toward -y).

Panels (the woven lattice):
    - Every panel is fully independent -- its own position (`anchor`) and orientation
      (`angle`), never grouped or sharing an orientation with any other panel. There is
      no "family" concept. A panel at angle=pi/2 happens to be parallel to Wall A (what
      used to be called "family A"); one at angle=0 happens to be parallel to Wall B
      ("family B"); anything in between is just as valid.
    - Any two panels whose floor-plan footprints cross get a cross-lap joint there
      (`geometry.panels.weave_crossings`, general 2D line intersection -- perpendicular
      crossings are simply the theta=90 special case).
    - Which wall(s) a panel actually contributes light-blocking area to is computed
      dynamically from its geometry, not declared -- see
      `geometry.projection.primary_wall_of`.

Every panel casts *some* shadow on both walls; how much of each is a matter of degree
(a panel's projected coverage area on a wall), not a binary in/out family membership.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Wall:
    """A rectangular image region on a wall. `plane` is 'x' (Wall A), 'y' (Wall B), or
    'n' -- a THIRD vertical wall at an arbitrary horizontal normal angle (`normal_deg`,
    degrees from +x), for 3-wall setups. 'n' walls are centred on their own in-plane axis
    (rather than starting at 0) since there's no natural origin for an arbitrary angle."""
    name: str
    plane: str          # 'x' -> plane x=offset (Wall A); 'y' -> plane y=offset (Wall B);
                        # 'n' -> plane (normal_deg) at signed distance `offset` from origin
    offset: float       # usually 0.0
    width_axis: str     # 'y' for Wall A, 'x' for Wall B; unused for plane='n'
    width: float        # extent along width_axis (metres)
    height: float       # extent along z (metres)
    z0: float           # bottom of the image region (metres above floor)
    normal_deg: float = 0.0   # only used when plane == 'n': horizontal normal angle (degrees)
    lateral_offset: float = 0.0   # only used when plane == 'n': shifts the centred width span
                                  # along axis_u, so the rectangle can be centred on wherever the
                                  # body actually projects instead of always on the origin normal

    @property
    def normal(self) -> np.ndarray:
        if self.plane == "x":
            return np.array([1.0, 0.0, 0.0])
        if self.plane == "y":
            return np.array([0.0, 1.0, 0.0])
        t = math.radians(self.normal_deg)
        return np.array([math.cos(t), math.sin(t), 0.0])

    @property
    def origin(self) -> np.ndarray:
        """3D point at (width=0, height=0) corner of the image region."""
        if self.plane == "x":              # Wall A: vary y (width) and z (height)
            return np.array([self.offset, 0.0, self.z0])
        if self.plane == "y":
            return np.array([0.0, self.offset, self.z0])   # Wall B: vary x (width) and z
        n = self.normal
        centre = n * self.offset
        return centre + self.axis_u * (self.lateral_offset - self.width / 2.0) + np.array([0.0, 0.0, self.z0])

    @property
    def axis_u(self) -> np.ndarray:
        """In-plane unit axis for image width."""
        if self.plane == "n":
            t = math.radians(self.normal_deg)
            return np.array([-math.sin(t), math.cos(t), 0.0])   # perpendicular to normal, horizontal
        return np.array([0.0, 1.0, 0.0]) if self.width_axis == "y" else np.array([1.0, 0.0, 0.0])

    @property
    def axis_v(self) -> np.ndarray:
        """In-plane unit axis for image height (always +z)."""
        return np.array([0.0, 0.0, 1.0])


@dataclass
class Light:
    name: str
    pos: Tuple[float, float, float]

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=float)


@dataclass
class Panel:
    """One flat rectangular panel plane holding an opacity field, independently
    positioned/oriented -- there is no grouping/family concept. `angle` (radians
    from the +x axis, in the floor (x,y) plane) and `anchor` (the (x,y) floor-plan
    point at local u=0) fully determine a panel's position; `v` always stays the
    vertical/height axis.

    This used to be two hardcoded special cases ("family A"/"family B") plus an
    optional general fallback for diagonals. It's now just the general
    parameterisation, always: `angle=pi/2` reproduces what used to be called
    family A's orientation, `angle=0` reproduces family B's -- those are simply
    two particular angles now, not special-cased anywhere in this class or in
    any code that consumes it (weave crossings, sanity checks, placement).

    Which wall(s) a panel actually contributes to is no longer a static label --
    it's determined dynamically from its geometry (how much of its area actually
    projects onto each wall), via `geometry.projection.primary_wall_of`.
    """
    name: str
    angle: float                          # radians from +x axis, floor (x,y) plane
    anchor: Tuple[float, float]           # (x,y) floor-plan point at local u=0
    u_range: Tuple[float, float]
    v_range: Tuple[float, float]
    thickness: float

    @property
    def origin(self) -> np.ndarray:
        """The 3D point at local (u,v)=(0,0)."""
        ax, ay = self.anchor
        return np.array([ax, ay, 0.0])

    def is_diagonal(self, tol_deg: float = 5.0) -> bool:
        """True if this panel is a DIAGONAL plane -- rotated about z away from both
        wall-parallel orientations (angle not ~0 deg = parallel to Wall B, nor ~90 deg =
        parallel to Wall A). Diagonals cast on both walls and are what make the piece
        non-trivial (no single flat plane is readable); axis-aligned panels alone let each
        wall's image live on one readable plane. `tol_deg` is the wall-parallel tolerance."""
        deg = math.degrees(self.angle) % 180.0
        return not (deg <= tol_deg or abs(deg - 90.0) <= tol_deg or deg >= 180.0 - tol_deg)

    @property
    def axis_u(self) -> np.ndarray:
        return np.array([np.cos(self.angle), np.sin(self.angle), 0.0])

    @property
    def axis_v(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    @property
    def normal(self) -> np.ndarray:
        return np.array([-np.sin(self.angle), np.cos(self.angle), 0.0])

    @property
    def u_size(self) -> float:
        return self.u_range[1] - self.u_range[0]

    @property
    def v_size(self) -> float:
        return self.v_range[1] - self.v_range[0]

    def corners_uv(self) -> np.ndarray:
        """4 corner points in local (u,v) metres."""
        (u0, u1), (v0, v1) = self.u_range, self.v_range
        return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])

    def uv_to_xyz(self, uv: np.ndarray) -> np.ndarray:
        """Map local (u,v) (metres) -> 3D. uv shape (...,2)."""
        uv = np.asarray(uv, dtype=float)
        return self.origin + uv[..., 0:1] * self.axis_u + uv[..., 1:2] * self.axis_v

    def floor_segment_xy(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """This panel's footprint as a floor-plan (x,y) line segment, ((x0,y0),(x1,y1))
        at u=u_range[0]/u_range[1] -- what weave-crossing and sanity-check geometry
        use for in-plane (x,y) position."""
        ax, ay = self.anchor
        dx, dy = np.cos(self.angle), np.sin(self.angle)
        u0, u1 = self.u_range
        return (ax + u0 * dx, ay + u0 * dy), (ax + u1 * dx, ay + u1 * dy)


@dataclass
class SolveParams:
    mode: str = "partition"                    # "partition" (scattered shards) | "optimize"
    fragment_size: float = 0.09                # target shard size in WALL/shadow space (m)
    fragment_max_area: float = 0.014           # hard cap on a shard's shadow area (m^2)
    fragment_min_area: float = 0.0012          # shards smaller than this (shadow) are dropped
    shard_gap: float = 0.0                     # optional erosion (m); 0 = keep full coverage
                                               # (shards already separate: neighbours sit on
                                               # other depth planes). Raise only for extra gap.
    depth_bias: float = 0.0                    # 0..~0.3: bias shard count toward near panels
    wall_res: Tuple[int, int] = (300, 300)     # (rows over z/height, cols over width)
    panel_res: Tuple[int, int] = (280, 280)
    iters: int = 300
    lr: float = 0.05
    lambda_sparsity: float = 0.002
    lambda_tv: float = 0.001
    lambda_crosstalk: float = 0.0              # extra explicit penalty (recon already covers it)
    seed: int = 0
    # --- multi-run search (overlap colour mode; 1/False/0 = today's single-run behaviour) ---
    restarts: int = 1                          # shard-placement restarts; keep best by composite score
    panel_restarts: int = 1                    # candidate panel layouts (needs search_panels)
    time_budget: float = 0.0                   # wallclock sec cap (perf_counter); 0 = off
    search_panels: bool = False                # greedy panel-layout search instead of the panels: block
    damage_weight: float = 0.0                 # >0 = cross-talk-damage-minimising host selection
                                               # (auto-set to 0.5 when multi-running, see cli)
    credit_weight: float = 0.0                 # companion to damage_weight; 0 -> library default
    score_ssim_weight: float = 1.0             # composite-score weights (solve/search.py::score_layout)
    score_edge_weight: float = 0.5
    score_crosstalk_weight: float = 0.25
    diagonal_frac: float = 0.0                  # overlap mode: force this SHARE of shards onto
                                               # diagonal planes (the greedy otherwise leaves
                                               # them empty and the piece is trivial). 0 = off.
    # --- panel-search geometry (panel_search.build_panels_greedy candidate box) ---
    # Defaults reproduce the values that used to be hardcoded in that function's signature,
    # so existing scenes behave identically. A scene whose sculpture lives elsewhere (e.g.
    # the tabletop scene: a small body near the lights at ~5-8x magnification) MUST override
    # these -- in particular search_mag_cap, which otherwise rejects every high-mag candidate.
    search_anchor_range: Tuple[float, float] = (0.5, 2.4)   # (x,y) box for candidate panel CENTRES
    search_standoff: float = 0.5               # min floor-plan distance from BOTH walls
    search_mag_cap: float = 3.0                # reject candidates with magnification above this
    search_u_size_range: Tuple[float, float] = (0.25, 0.75) # candidate lateral size (metres)
    search_v_range: Tuple[float, float] = (0.03, 1.18)      # candidate vertical extent (metres)


@dataclass
class FabParams:
    kerf: float = 0.0002                       # laser/CNC kerf width (metres)
    min_feature: float = 0.004                 # smallest cuttable feature (metres)
    threshold: float = 0.5                     # opacity -> solid threshold
    sheet_size: Tuple[float, float] = (0.6, 0.4)
    formats: Tuple[str, ...] = ("dxf", "svg")
    joint_clearance: float = 0.0001            # extra slot width for slide fit


@dataclass
class TableSpec:
    """A table/pedestal the shard body stands on. VISUALISATION ONLY -- drawn by the 3D
    previews (interactive3d, render3d) so the piece reads as a small object on furniture;
    it has zero effect on projection, solving, or fabrication. The loader only checks that
    no panel standing over the table dips below its top (the body would intersect it)."""
    top_z: float                                # height of the table top surface (metres)
    center: Tuple[float, float]                 # (x,y) centre of the top
    size: Tuple[float, float]                   # (x extent, y extent) of the top
    thickness: float = 0.04                     # top slab thickness
    legs: bool = True                           # draw 4 legs down to the floor


@dataclass
class Scene:
    walls: dict                                 # {'A': Wall, 'B': Wall}
    lights: dict                                # {'A': Light, 'B': Light}
    panels: List[Panel]
    source_radius: float                        # effective light source radius (penumbra)
    material_thickness: float
    solve: SolveParams = field(default_factory=SolveParams)
    fab: FabParams = field(default_factory=FabParams)
    table: Optional[TableSpec] = None           # optional display table under the shard body
    color_palette: List[str] = field(default_factory=lambda: ["C", "M", "Y", "K"])
    white_threshold: float = 0.90               # pixels brighter than this -> clear (no shard)
    color_max_layers: int = 2                   # max stacked shards per channel (tone levels)
    color_max_stack: int = 3                    # overlap mode: max channels laminated per shard
    overlap_jitter: float = 0.0                 # overlap mode: per-channel scatter, frac of shard size
                                                # (0 = fidelity-first: no position scatter, only
                                                # same-plane channel lamination; raise to trade
                                                # wall-reconstruction accuracy for a more scrambled,
                                                # harder-to-read physical structure)
    overlap_shard_budget: int = 220             # overlap mode: target shard count PER WALL --
                                                # base spacing is auto-tuned to hit this regardless
                                                # of image content, so fabrication effort stays
                                                # predictable (0/None = no budget, use fragment_size as-is)
    overlap_detail_bias: float = 2.0            # overlap mode: how strongly shards concentrate at
                                                # edges/colour transitions vs spreading evenly
                                                # (0 = uniform, like the legacy behaviour)
    overlap_penumbra_min_feature_sigmas: float = 2.0   # overlap mode: raise the shard fragmentation's
                                                # minimum-feature floor to this many multiples of the
                                                # worst-case (farthest) panel's physical penumbra sigma
                                                # in that family, so shard detail finer than a real,
                                                # non-ideal (non-point) light could ever resolve gets
                                                # merged into its neighbour instead of fabricated as
                                                # illusory precision. 0 disables (old behaviour).

    def light_for_wall(self, wall_name: str) -> Light:
        """Wall A is lit by Light A, Wall B by Light B."""
        return self.lights[wall_name]
