"""
HOW MUCH ROOM IS LEFT? Honest ceiling analysis for cross-wall double duty.

Today a shard's colour is fixed from its OWN wall (decompose.py:780 `_shard_channels(rgb[m],...)`)
BEFORE its host depth-plane is chosen (decompose.py:813-814). `_shard_damage` only SCORES that
fixed colour against the other wall; it never changes it. So a shard helps the second wall only
by luck. The question: would letting a shard pick a colour that serves BOTH walls actually buy
anything?

Reported per wall (as % of that wall's subject pixels):

  ACHIEVED   what the real pipeline gets today (cross-talk-only re-render, colour-agreeing).
  CEIL(a)    colour still fixed from its own wall, but every shard at its individually BEST
             host. Upper bound of TODAY's design.
  CEIL(c)~   the OLD loose estimate from ceiling_straddle.py: `d < 2*match_tol` -- a tolerance
             relaxation that never builds the colour. Kept only to show how much it overstates.
  CEIL(c)    HONEST: for each (shard, host) we CONSTRUCT a compromise colour by blending the
             shard's own dominant colour toward what the second wall wants at its landing,
             push it through the SAME realisable path the pipeline uses
             (rgb_to_cmyk -> _active_channels(max_stack) -> _shard_transmit), require it within
             match_tol of the second wall, and CHARGE the extra error it costs the first wall.
  cost_A     mean extra primary-wall colour error paid by those compromise colours (0 = free).
  STRADDLE   colour-uniformity of each shard's landing blob. LOW means one Voronoi cell lands
             across regions wanting different colours -- then no single colour can serve both
             and the SHAPE (the wall-space Voronoi) is the limiter, not the colour.

Run:  py out_thickness_test/headroom.py
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.geometry import homography as H
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C

MATCH_TOL = 0.30
PRIMARY_TOL = 0.10          # how much extra primary-wall error a compromise may cost
W_GRID = np.linspace(0.0, 1.0, 11)
SEED = 1
SCENE = "scenes/tabletop60.yaml"

PAIRS = [
    ("cut sunflowers x oranges", "examples/sf_surface_nobg.png", "examples/oranges_nobg.png"),
    ("vase x oranges",           "examples/sunflowers_clean_nobg.png", "examples/oranges_nobg.png"),
    ("vase x cut sunflowers",    "examples/sunflowers_clean_nobg.png", "examples/sf_surface_nobg.png"),
]

scene0 = load_scene(SCENE)
WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)


def build(pa, pb):
    t = {"A": C.load_color_target(pa, WR, white_thr=scene0.white_threshold),
         "B": C.load_color_target(pb, WR, white_thr=scene0.white_threshold)}
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=t, seed=SEED,
                                    angle_deg_range=(5, 85), anchor_range=SP.search_anchor_range,
                                    standoff=SP.search_standoff, mag_cap=SP.search_mag_cap,
                                    u_size_range=SP.search_u_size_range, v_range=SP.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=SP)
    return ts, build_projection_table(ts), t, panels


def achieved(ts, table, t, panels):
    """What the real pipeline gets today: render ONLY the panels whose primary wall is the other
    one (i.e. pure cross-talk), and count subject pixels it darkens in the RIGHT colour."""
    renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, t, names=NAMES, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=SEED, damage_weight=0.5, credit_weight=0.5, match_tol=MATCH_TOL)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    subject = {w: C.subject_mask(t[w], ts.white_threshold) for w in ("A", "B")}
    out = {}
    for w in ("A", "B"):
        q = pT.copy()
        for gi, p in enumerate(panels):
            if prim[p.name] == w:          # drop the panels that BUILD this wall
                q[gi] = 1.0
        xr = renderer.render_color_np(q)[w]
        subj = subject[w]
        onsub = ((1.0 - xr.mean(-1)) > 0.05) & subj
        if onsub.any():
            d = np.sqrt(((xr[onsub] - t[w][onsub]) ** 2).sum(1))
            out[w] = 100.0 * (d < MATCH_TOL).sum() / max(subj.sum(), 1)
        else:
            out[w] = 0.0
    return out


def realisable_colour(rgb_target, max_stack):
    """Push an arbitrary RGB wish through the SAME path the pipeline uses, so the result is a
    colour the fabricated shard can actually be: CMYK split -> strongest <=max_stack channels ->
    laminated transmittance."""
    cv = C.rgb_to_cmyk(rgb_target)
    act = decompose._active_channels(cv, max_stack)
    return decompose._shard_transmit(cv, act)


def ceilings(ts, table, t):
    subject = {w: C.subject_mask(t[w], ts.white_threshold) for w in ("A", "B")}
    sp = ts.solve
    res = {}
    for fi, family in enumerate(("A", "B")):
        wallP = ts.walls[family]; rgb = t[family]; HnP, WnP = rgb.shape[:2]
        secondary = "B" if family == "A" else "A"
        wallS = ts.walls[secondary]; tgtS = t[secondary]; HnS, WnS = tgtS.shape[:2]
        subjS = subject[secondary]
        fam = [(gi, p) for gi, p in enumerate(ts.panels)
               if primary_wall_of(ts, table, p) == family]
        if not fam:
            continue
        rng = np.random.default_rng(1 + 17 * fi)
        px_m = 0.5 * (wallP.width / WnP + wallP.height / HnP)
        frags, _mult, _nf = decompose._autotune_spacing(
            C.subject_mask(rgb, ts.white_threshold), sp.fragment_size / max(px_m, 1e-9),
            sp.fragment_min_area / px_m ** 2, sp.fragment_max_area / px_m ** 2, rng,
            decompose._importance_map(rgb), ts.overlap_detail_bias, ts.overlap_shard_budget)
        reach = {gi: decompose._panel_reachable_mask(p, table[(p.name, family)].H_pw, wallP, HnP, WnP)
                 for gi, p in fam}
        Gs = {gi: decompose._wall_to_wall_H(table, p, family, secondary) for gi, p in fam}

        a_acc = np.zeros((HnS, WnS), bool)      # colour from own wall
        cl_acc = np.zeros((HnS, WnS), bool)     # old loose 2*tol relaxation
        ch_acc = np.zeros((HnS, WnS), bool)     # honest constructed compromise
        costs = []
        strad_num = strad_den = 0.0
        for m in frags:
            dom_P = C.dominant_rgb(rgb[m])
            T_a = realisable_colour(dom_P, ts.color_max_stack)
            err_a_primary = float(np.sqrt(((T_a - dom_P) ** 2).sum()))
            ys, xs = np.where(m)
            cover = np.array([reach[gi][ys, xs].mean() for gi, _ in fam])
            viable = np.flatnonzero(cover > 0)
            if viable.size == 0:
                continue
            a_best = cl_best = ch_best = None
            a_n = cl_n = ch_n = -1
            ch_cost = None
            a_ = (xs + 0.5) / WnP * wallP.width
            b_ = (ys + 0.5) / HnP * wallP.height
            for k in viable:
                q = H.apply_homography(Gs[fam[k][0]], np.stack([a_, b_], axis=-1))
                ci = np.rint(q[:, 0] / max(wallS.width, 1e-9) * WnS - 0.5).astype(int)
                ri = np.rint(q[:, 1] / max(wallS.height, 1e-9) * HnS - 0.5).astype(int)
                on = (ci >= 0) & (ci < WnS) & (ri >= 0) & (ri < HnS)
                if not on.any():
                    continue
                rr, cc = ri[on], ci[on]
                onsub = subjS[rr, cc]
                if not onsub.any():
                    continue
                rr, cc = rr[onsub], cc[onsub]
                tb = tgtS[rr, cc]                       # what wall S wants over the landing
                d_a = np.sqrt(((T_a[None, :] - tb) ** 2).sum(1))
                am = d_a < MATCH_TOL
                cm = d_a < 2 * MATCH_TOL                # the old, loose estimate
                if int(am.sum()) > a_n:
                    a_n = int(am.sum()); a_best = (rr[am], cc[am])
                if int(cm.sum()) > cl_n:
                    cl_n = int(cm.sum()); cl_best = (rr[cm], cc[cm])

                # ---- HONEST compromise: construct it, keep it realisable, charge wall A ----
                dom_S = np.median(tb, axis=0)
                best_hit, best_T, best_cost = -1, None, None
                for w in W_GRID:
                    T_w = realisable_colour((1.0 - w) * dom_P + w * dom_S, ts.color_max_stack)
                    cost = float(np.sqrt(((T_w - dom_P) ** 2).sum())) - err_a_primary
                    if cost > PRIMARY_TOL:              # too expensive for the primary wall
                        continue
                    hit = int((np.sqrt(((T_w[None, :] - tb) ** 2).sum(1)) < MATCH_TOL).sum())
                    if hit > best_hit:
                        best_hit, best_T, best_cost = hit, T_w, max(0.0, cost)
                if best_hit > ch_n:
                    ch_n = best_hit
                    keep = np.sqrt(((best_T[None, :] - tb) ** 2).sum(1)) < MATCH_TOL
                    ch_best = (rr[keep], cc[keep]); ch_cost = best_cost
                    strad_blob = tb

            if a_best is not None and a_best[0].size:
                a_acc[a_best] = True
            if cl_best is not None and cl_best[0].size:
                cl_acc[cl_best] = True
            if ch_best is not None and ch_best[0].size:
                ch_acc[ch_best] = True
                if ch_cost is not None:
                    costs.append(ch_cost)
                dom = np.median(strad_blob, axis=0)
                uni = (np.sqrt(((strad_blob - dom) ** 2).sum(1)) < MATCH_TOL).mean()
                strad_num += uni * len(strad_blob); strad_den += len(strad_blob)
        denom = max(subjS.sum(), 1)
        res[secondary] = dict(ceil_a=100.0 * a_acc.sum() / denom,
                              ceil_c_loose=100.0 * cl_acc.sum() / denom,
                              ceil_c=100.0 * ch_acc.sum() / denom,
                              cost_primary=float(np.mean(costs)) if costs else 0.0,
                              straddle=(strad_num / strad_den) if strad_den else float("nan"))
    return res


print(f"match_tol={MATCH_TOL}  primary_tol={PRIMARY_TOL}  scene={SCENE}  seed={SEED}\n")
print(f"{'pair':26s} {'wall':4s} {'ACHIEVED':>9s} {'CEIL(a)':>8s} {'CEIL(c)~':>9s} "
      f"{'CEIL(c)':>8s} {'cost_A':>7s} {'STRADDLE':>9s}")
print("-" * 92)
out = {}
for label, pa, pb in PAIRS:
    ts, table, t, panels = build(pa, pb)
    ach = achieved(ts, table, t, panels)
    cei = ceilings(ts, table, t)
    out[label] = {"achieved": ach, "ceilings": cei}
    for w in ("A", "B"):
        c = cei.get(w)
        if c is None:
            continue
        print(f"{label:26s} {w:4s} {ach[w]:8.1f}% {c['ceil_a']:7.1f}% {c['ceil_c_loose']:8.1f}% "
              f"{c['ceil_c']:7.1f}% {c['cost_primary']:7.3f} {c['straddle']:9.2f}")
    print("-" * 92)

with open("out_thickness_test/mona_pairs/headroom.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote out_thickness_test/mona_pairs/headroom.json")
print("\nDECISION RULE (committed before seeing numbers):")
print("  CEIL(c) - CEIL(a) > ~5 points at acceptable cost_A -> build joint depth+colour.")
print("  STRADDLE < ~0.6                                    -> shard SHAPE is the limiter.")
print("  neither                                            -> near the physical ceiling; stop.")
