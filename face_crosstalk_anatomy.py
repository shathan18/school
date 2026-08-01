"""Where does the cross-talk actually GO? Anatomy of the missing intersection.

`face_duty_sweep.py` found the two-tone face pair scores good 0.58% / bad 0.80% -- a
good/bad RATIO of 0.72, which looks respectable, but the ABSOLUTE duty is ~20x below the
Hokusai control (good 11.5% / bad 15.2%, g/b 0.76) rendered by `render_pair.py` at an
identical recipe (14 panels, 30-60 deg, damage 0.5, credit 1.0, seed 2, ~300 shards).

Same ratio, twentyfold less overlap. A ratio can look healthy while the piece is barely an
intersection at all, because the ratio only describes the QUALITY of whatever cross-talk
exists, not its QUANTITY. Ranking on g/b alone would have hidden this completely -- the
same trap as `joint_intersection_pct`, one level up.

So this script decomposes the cross-talk budget instead of scoring it. For each wall it
renders ONLY the panels whose primary wall is the other one (the definition of cross-talk
used by `colour_agreeing_duty`) and asks where those photons land:

    subj%      how much of the wall the target's subject occupies (ink area)
    xt_wall%   how much of the WHOLE wall the cross-talk darkens
    xt_subj%   how much of the SUBJECT it darkens        -> the ceiling on good
    xt_white%  how much of the NON-subject (paper) it darkens -> dirt on clean white
    good/bad   the dE-25 split of xt_subj

The decisive comparison is xt_wall%: if it is high for the faces but xt_subj% is low, the
cross-talk exists and is simply missing the subject (a placement problem, fixable). If
xt_wall% is itself near zero, there is almost no cross-talk to place (a density problem,
not fixable by re-aiming -- only by more shards or more ink).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shadowart.geometry.projection import primary_wall_of
from shadowart.targets import color as C

import face_render300 as FR
import face_duty_sweep as DS

OUT = Path("out_faces_hc/duty")


def anatomy(b) -> dict:
    """Per-wall cross-talk budget for one built layout."""
    panels, targets = b["panels"], b["targets"]
    prim = {p.name: primary_wall_of(b["layout"], b["table"], p) for p in panels}
    white_thr = b["layout"].white_threshold
    res = {}
    for wall in ("A", "B"):
        q = b["panel_T"].copy()
        for gi, p in enumerate(panels):
            if prim.get(p.name) == wall:
                q[gi] = 1.0                       # drop the panels that BUILD this wall
        xr = b["renderer"].render_color_np(q)[wall]
        subj = C.subject_mask(targets[wall], white_thr)
        dark = (1.0 - xr.mean(-1)) > 0.05         # same dark_thr as colour_agreeing_duty
        npix = subj.size
        onsub = dark & subj
        denom = max(int(subj.sum()), 1)
        if onsub.any():
            d = C.delta_e(xr[onsub], targets[wall][onsub])
            good = 100.0 * float((d < FR.MATCH_TOL_REPORT).sum()) / denom
            bad = 100.0 * float((d >= FR.MATCH_TOL_REPORT).sum()) / denom
        else:
            good = bad = 0.0
        res[wall] = dict(
            subj_pct=100.0 * float(subj.sum()) / npix,
            xt_wall_pct=100.0 * float(dark.sum()) / npix,
            xt_subj_pct=100.0 * float(onsub.sum()) / denom,
            xt_white_pct=100.0 * float((dark & ~subj).sum()) / max(int((~subj).sum()), 1),
            good=good, bad=bad,
            n_crosstalk_panels=sum(1 for p in panels if prim.get(p.name) != wall),
        )
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = FR.write_targets()
    print("=" * 104)
    print("CROSS-TALK ANATOMY -- is the missing intersection a PLACEMENT or a DENSITY problem?")
    print("=" * 104)
    print(f"{'pair':34s} {'w':>2s} {'subj%':>6s} {'xt_wall%':>8s} {'xt_subj%':>8s} "
          f"{'xt_white%':>9s} {'good%':>6s} {'bad%':>6s} {'xtP':>4s}")
    print("-" * 104)
    dump = {}
    for pair in FR.PAIRS:
        b = DS.build(pair, seed=2, panel_count=FR.PANEL_COUNT,
                     angle_range=FR.ANGLE_RANGE, credit_weight=FR.CREDIT_WEIGHT)
        a = anatomy(b)
        dump[pair[0]] = a
        for wall in ("A", "B"):
            r = a[wall]
            print(f"{(pair[0] if wall == 'A' else ''):34s} {wall:>2s} {r['subj_pct']:>6.1f} "
                  f"{r['xt_wall_pct']:>8.2f} {r['xt_subj_pct']:>8.2f} {r['xt_white_pct']:>9.2f} "
                  f"{r['good']:>6.2f} {r['bad']:>6.2f} {r['n_crosstalk_panels']:>4d}")
    (OUT / "crosstalk_anatomy.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/crosstalk_anatomy.json")
    print("\nReading: xt_wall% near zero => DENSITY problem (too few shards / too little ink);")
    print("         xt_wall% high but xt_subj% low => PLACEMENT problem (cross-talk misses subject).")


if __name__ == "__main__":
    main()
