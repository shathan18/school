"""
Steps 2 & 3 of the credit-fix work.

Retire the naive "lands on subject" B_good entirely. For each image pair report:

  ACHIEVED  colour-agreeing B_good  -- the honest current number: fraction of the
            secondary wall's SUBJECT that receives stray coverage whose RENDERED colour
            actually matches what that wall wants there (||pred-tgt||<match_tol). This is
            the ONLY B_good reported.

  CEILING(a) colour fixed from wall A, depth free -- best host per shard: the max B_good
            that option (a) (re-choose depth only) could ever reach. Upper bound; each
            shard placed at its individually best host (not a single realisable layout).

  CEILING(c) colour free (compromise), depth free -- best host per shard AND a single
            colour allowed to serve both walls, feasible where the two walls' wanted
            colours sit within 2*match_tol of each other at the linked points. The
            option-(c) upside.

  STRADDLE  shape cap: for each shard's landing blob on the secondary subject, the
            fraction of the blob within match_tol of the blob's own dominant wanted
            colour. High => blobs land on colour-uniform regions (shape is NOT the
            bottleneck); low => one Voronoi cell straddles regions wanting different
            colours, which caps (c) no matter how cleverly colour is chosen.

Step 3 control: the ACHIEVED number is the current pipeline unchanged; comparing it
across an arbitrary pair (apples/breakfast) vs deliberately compatible pairs
(Monet day/dusk, Mona/Pearl) isolates whether double-duty is bounded by the image
pair or the algorithm.
"""
import sys, dataclasses
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.geometry import homography as H
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C

MATCH_TOL = 0.30
scene0 = load_scene("scenes/example.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK


def build(pairA, pairB, seed):
    t = {"A": C.load_color_target(pairA, WR, white_thr=scene0.white_threshold),
         "B": C.load_color_target(pairB, WR, white_thr=scene0.white_threshold)}
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16,
                                    targets=t, seed=seed, angle_deg_range=(5, 85))
    ts = dataclasses.replace(scene0, panels=panels)
    FS = 0.135 / np.sqrt(0.5)
    SP = dataclasses.replace(scene0.solve, fragment_size=FS,
                             fragment_min_area=scene0.solve.fragment_min_area * (FS / 0.135) ** 2,
                             fragment_max_area=scene0.solve.fragment_max_area * (FS / 0.135) ** 2)
    ts = dataclasses.replace(ts, solve=SP)
    table = build_projection_table(ts)
    return ts, table, t, panels


def achieved_bgood(ts, table, t, panels):
    """Honest colour-agreeing B_good on the ACTUAL rendered layout (current pipeline)."""
    renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, t, names=NAMES, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=1, damage_weight=0.5, credit_weight=0.5, match_tol=MATCH_TOL)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    subject = {w: C.subject_mask(t[w], ts.white_threshold) for w in ("A", "B")}

    def render_keep(keep):
        q = pT.copy()
        for gi, p in enumerate(panels):
            if p.name not in keep:
                q[gi] = 1.0
        return renderer.render_color_np(q)

    out = {}
    for w in ("A", "B"):
        subj = subject[w]; denom = max(subj.sum(), 1)
        nonprim = {p.name for p in panels if prim[p.name] != w}   # cross-talk sources only
        xr = render_keep(nonprim)[w]
        dark = (1.0 - xr.mean(-1)) > 0.05
        onsub = dark & subj
        if onsub.any():
            cx = xr[onsub]; ct = t[w][onsub]
            match = np.sqrt(((cx - ct) ** 2).sum(1)) < MATCH_TOL
            out[w] = 100.0 * match.sum() / denom
        else:
            out[w] = 0.0
    return out


def ceilings_and_straddle(ts, table, t, panels):
    """Replicate the family-loop geometry to compute (a)/(c) ceilings and the straddle cap.
    Upper bounds: each shard evaluated at its individually best host."""
    sp = ts.solve
    subject = {w: C.subject_mask(t[w], ts.white_threshold) for w in ("A", "B")}
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
        spacing = sp.fragment_size / max(px_m, 1e-9)
        min_px = sp.fragment_min_area / (px_m ** 2)
        max_px = sp.fragment_max_area / (px_m ** 2)
        importance = decompose._importance_map(rgb)
        subj_mask = C.subject_mask(rgb, ts.white_threshold)
        frags, mult, nf = decompose._autotune_spacing(subj_mask, spacing, min_px, max_px, rng,
                                                       importance, ts.overlap_detail_bias, None)
        reach = {gi: decompose._panel_reachable_mask(p, table[(p.name, family)].H_pw, wallP, HnP, WnP)
                 for gi, p in fam}
        Gs = {gi: decompose._wall_to_wall_H(table, p, family, secondary) for gi, p in fam}

        a_accum = np.zeros((HnS, WnS), bool)     # option (a): colour fixed from wall A
        c_accum = np.zeros((HnS, WnS), bool)     # option (c): compromise colour serving both
        g_accum = np.zeros((HnS, WnS), bool)     # geometry: lands on subject at all
        strad_num = 0.0; strad_den = 0.0
        for m in frags:
            cvals = C.rgb_to_cmyk(C.dominant_rgb(rgb[m]))
            active = decompose._active_channels(cvals, ts.color_max_stack)
            T_shard = decompose._shard_transmit(cvals, active)   # colour that arrives on wall S
            ys, xs = np.where(m)
            cover = np.array([reach[gi][ys, xs].mean() for gi, _ in fam])
            viable = np.flatnonzero(cover > 0)
            if viable.size == 0:
                continue
            a_best = c_best = g_best = None
            a_n = c_n = g_n = -1
            for k in viable:
                G = Gs[fam[k][0]]
                a = (xs + 0.5) / WnP * wallP.width
                b = (ys + 0.5) / HnP * wallP.height
                q = H.apply_homography(G, np.stack([a, b], axis=-1))
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
                tb = tgtS[rr, cc]
                d = np.sqrt(((T_shard[None, :] - tb) ** 2).sum(1))
                a_match = d < MATCH_TOL                 # wall A's colour already fits wall B
                c_match = d < 2 * MATCH_TOL             # a compromise colour can fit both
                if onsub.sum() > g_n:
                    g_n = int(onsub.sum()); g_best = (rr, cc)
                if int(a_match.sum()) > a_n:
                    a_n = int(a_match.sum()); a_best = (rr[a_match], cc[a_match])
                if int(c_match.sum()) > c_n:
                    c_n = int(c_match.sum()); c_best = (rr[c_match], cc[c_match], rr, cc)
            if g_best is not None:
                g_accum[g_best] = True
            if a_best is not None and a_best[0].size:
                a_accum[a_best] = True
            if c_best is not None and c_best[0].size:
                c_accum[(c_best[0], c_best[1])] = True
                # straddle: colour-uniformity of the blob's wanted colour under this landing
                brr, bcc = c_best[2], c_best[3]
                blob = tgtS[brr, bcc]
                dom = np.median(blob, axis=0)
                uni = (np.sqrt(((blob - dom) ** 2).sum(1)) < MATCH_TOL).mean()
                strad_num += uni * len(brr); strad_den += len(brr)
        denom = max(subjS.sum(), 1)
        res[secondary] = dict(
            ceil_geom=100.0 * g_accum.sum() / denom,
            ceil_a=100.0 * a_accum.sum() / denom,
            ceil_c=100.0 * c_accum.sum() / denom,
            straddle=(strad_num / strad_den) if strad_den else float("nan"),
        )
    return res


PAIRS = [
    ("apples/breakfast  (ARBITRARY)", "examples/apples.jpg", "examples/breakfast.jpg"),
    ("Monet day/dusk    (COMPATIBLE)", "examples/monet_day_edited.jpeg", "examples/monet_dusk_edited.jpeg"),
    ("Mona/Pearl        (COMPATIBLE)", "examples/Mona_Lisa.jpg", "examples/The_Girl_With_The_Pearl_Earring.jpg"),
]

print(f"match_tol = {MATCH_TOL}   (credit fires only where ||arrived-wanted|| < this)\n")
print(f"{'pair':32s} {'wall':5s} {'B_good':>8s} {'ceil(a)':>8s} {'ceil(c)':>8s} {'straddle':>9s}")
print("-" * 76)
for label, pa, pb in PAIRS:
    ts, table, t, panels = build(pa, pb, seed=1)
    ach = achieved_bgood(ts, table, t, panels)
    cs = ceilings_and_straddle(ts, table, t, panels)
    for w in ("A", "B"):
        c = cs.get(w, {})
        print(f"{label if w=='A' else '':32s} {w:5s} {ach[w]:7.1f}% "
              f"{c.get('ceil_a',float('nan')):7.1f}% {c.get('ceil_c',float('nan')):7.1f}% "
              f"{c.get('straddle',float('nan'))*100:8.0f}%")
    print()
print("legend: B_good = achieved honest colour-agreeing coverage of secondary subject (current pipeline).")
print("        ceil(a) = best achievable by re-choosing depth only (colour fixed from wall A).")
print("        ceil(c) = best achievable with free compromise colour (option c upside).")
print("        straddle = colour-uniformity of landing blobs (100%=shape not the cap; low=shape caps c).")
