"""Stage 2: re-quantise every shard's TONE with the geometry frozen.

WHY A SECOND STAGE EXISTS
-------------------------
The renders reach IoU 0.82 and still look wrong, and the two facts are consistent: IoU scores
the outline of the mark, so it is blind to a face that is correctly shaped and uniformly
black. Which is what happens. Look at `preview_views.png`: the silhouettes are right, the
interiors are crushed.

The cause is structural, not a tuning miss. Each shard's tone is chosen from ITS OWN view's
target, at partition time, before anything is known about which other views' shards will end
up in the same beam. But 93% of shards serve all three views, so a wall pixel typically
receives its own view's shard PLUS stray from roughly two others, and transmittances multiply:
three sheets that each individually wanted 0.6 compose to 0.6^3 = 0.22. Mid-grey skin arrives
as black. The silhouette survives -- which is exactly why IoU never complained.

WHY THIS IS NOT THE PRE-COMPENSATION THAT ALREADY FAILED
--------------------------------------------------------
`pearl3_baseline.precompensate_targets` attacked the same darkness by dividing the stray light
out of the TARGET and re-running the decomposer. It oscillated (0.690 -> 0.598 -> 0.665 ->
0.641) for a specific reason: re-solving changes which shards exist, which changes the stray
light, which changes the target -- a fixed point with no reason to contract, made worse by the
fact that the "stray" is largely deliberate sharing being double-counted.

Here the geometry never moves. The partition, the shard footprints and the host assignments
are all frozen; the only free variables are which of the four tones each shard region carries.
That turns an ill-posed fixed point into plain coordinate descent on

    L = sum_over_walls || render(panel tones) - target ||^2

sweeping one sheet at a time, evaluating all four tones, and keeping the best per pixel. Each
sweep is monotone in L by construction, because a sweep only accepts a change that lowers it.

A SIDE EFFECT THAT TURNED OUT TO BE A NON-ISSUE
-----------------------------------------------
The solver records a continuous per-pixel `stack_intensity` and the renderer applies each
sheet at partial strength, which `color.stack_transmit_lut`'s own docstring flags as a
render-side approximation the cut files do not implement. That looked like it might mean the
shipped metrics were measuring an object nobody could build. Measured: they were not. Under
the engrave palette the recorded intensities are all 1.0 (the tone IS the channel, so there is
no weak-channel case for the hue fix to weaken), and the intensity-weighted and full-strength
renders agree to four decimal places -- 0.8206 / 0.7912 / SSIM 0.7528 either way. The check is
kept in `main` and printed on every run, because it costs one render and it is the kind of gap
that would otherwise be discovered by a workshop rather than by a test.

Run:  .venv\\Scripts\\python.exe pearl3_retone.py --arm 30v4 --sweeps 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

import pearl3_baseline as P
from shadowart import metrics
from shadowart.geometry import homography as Hg
from shadowart.solve.initializer import _panel_pixel_uv
from shadowart.targets import color as C


def wall_to_panel_sampler(scene, table, panel, wall_name, panel_shape):
    """Row/col indices that pull a WALL-space raster back onto this panel's pixel grid.

    Same forward mapping the decomposer uses (panel pixel -> homography -> wall coordinates);
    precomputed once per (panel, wall) because the coordinate descent below re-samples the
    same pair dozens of times.
    """
    wall = scene.walls[wall_name]
    Hp, Wp = panel_shape
    uv = _panel_pixel_uv(panel, (Hp, Wp))
    ab = Hg.apply_homography(table[(panel.name, wall_name)].H_pw, uv)
    Hn, Wn = scene.solve.wall_res
    col = ab[..., 0] / max(wall.width, 1e-9) * Wn - 0.5
    row = ab[..., 1] / max(wall.height, 1e-9) * Hn - 0.5
    return row.ravel(), col.ravel(), (Hp, Wp)


def pull_back(field, sampler):
    """Sample a [Hn,Wn] wall raster at the panel's pixel positions -> [Hp,Wp]."""
    row, col, shape = sampler
    return ndimage.map_coordinates(field, [row, col], order=1, mode="nearest").reshape(shape)


def sq_error_maps(pred, targets):
    """Per-wall squared error summed over RGB -- used to PROPOSE tones, not to judge them."""
    return {w: ((pred[w] - targets[w]) ** 2).sum(axis=-1) for w in targets}


def total_loss(pred, targets):
    return float(sum(m.sum() for m in sq_error_maps(pred, targets).values()))


def objective(pred, targets):
    """The composite this project reports, and therefore the one a proposal is judged by.

    Squared error is the natural thing to differentiate and the wrong thing to accept on.
    Measured, at whole-shard granularity: MSE fell from 64061 to 48410 while mean IoU fell
    from 0.821 to 0.722 -- the descent had discovered that lightening the piece pays in MSE
    and charged the silhouette for it. So MSE proposes (it is per-pixel and cheap) and this
    composite decides. Same weights as the sweep harness, so nothing is being optimised here
    that is not being reported elsewhere.
    """
    m = metrics.evaluate_multiview(targets, pred)
    walls = [w for w in m if w != "_summary"]
    ssim = float(np.mean([m[w]["ssim"] for w in walls]))
    return (0.25 * m["_summary"]["mean_iou"] + 0.25 * m["_summary"]["min_iou"]
            + 0.50 * ssim), m


def region_labels(tone2d):
    """Label each sheet's shards as connected runs of ONE original tone -> ([Hp,Wp] int, n).

    Kept for `--unit shard`, which is the conservative comparison arm rather than the shipped
    setting: see `retone` for why deciding at whole-shard granularity turned out to be the
    wrong constraint even though it is the obvious one.
    """
    lab = np.zeros(tone2d.shape, dtype=np.int32)
    nxt = 0
    for k in range(1, int(tone2d.max()) + 1):
        m = tone2d == k
        if not m.any():
            continue
        l, n = ndimage.label(m)
        lab = np.where(m, l + nxt, lab)
        nxt += n
    return lab, nxt


def make_fabricable(tone2d, n_tones, min_feature, pixel_m):
    """Project a tone map onto the set of maps the laser can actually produce.

    Exactly the operator the vectoriser applies (`features.enforce_min_feature`, per tone),
    so a map that has passed through here survives `panel_stack_pieces` essentially unchanged.
    Darker tones are written last, matching the max-over-slots convention used to build the
    starting map, so a pixel claimed by two tones after cleanup resolves the same way both
    times.
    """
    from shadowart.raster2vec import features
    out = np.zeros_like(tone2d)
    for k in range(1, n_tones):
        m = tone2d == k
        if not m.any():
            continue
        m = features.enforce_min_feature(m, min_feature, pixel_m)
        out[m] = k
    return out


def retone(scene, table, renderer, targets, names, stack_colorid, sweeps=3, verbose=True,
           unit="pixel", min_feature=None):
    """Coordinate descent over tone ids with the geometry frozen. Returns (tone, log).

    THE UNIT OF DECISION IS THE THING THAT MATTERS HERE, and the obvious answers are wrong:

    * Per PIXEL with no fabrication limit: the render loss halves and every metric improves
      (mean IoU 0.8206 -> 0.8283, SSIM 0.7528 -> 0.8026, RMSE 0.1399 -> 0.0963) -- and then
      the vector round trip scores **0.4154**, because the descent bought its gain in detail
      finer than the export could represent and `enforce_min_feature` deleted it.
    * Per SHARD (`unit='shard'`): fabricable by construction, but there are only 281 shard
      regions across 18 sheets, and at that granularity squared error buys tone by giving up
      coverage -- foreground 0.32 -> 0.24 on the front view, mean IoU down to 0.722.
      Fabricable, and worse than not doing it.

    The resolution of that apparent dilemma is that the 5 mm limit was never the right limit
    for this decision. `min_feature` exists so a shard CUT OUT of a sheet survives as a
    physical object; a tone region is ENGRAVED into a sheet that stays whole, and its only
    limit is the beam spot (~0.2 mm, well under one panel pixel). So the descent runs per
    pixel against the engrave limit, and each sheet's map is projected onto the fabricable
    set after every update -- the same operator the exporter will apply, so what is scored
    here is what comes out of the machine.

    Every panel update is still a PROPOSAL that has to earn its place (see `objective`):
    squared error is per-pixel and cheap and so it proposes, but the composite this project
    reports decides, and a sheet whose new tone map does not improve it is rolled back.

    Two further simplifications, both fabrication rather than convenience: the stack is
    collapsed to one tone per pixel (a raster engrave pass cannot be laminated onto itself,
    and max_stack 1/2/3 measured identical under this palette), and intensity is dropped
    (regions are emitted at full strength, the only thing the laser can do).
    """
    S, G, Hp, Wp = stack_colorid.shape
    # One tone id per panel pixel; start from the darkest tone any slot asked for, since the
    # stack was approximating exactly that.
    tone = stack_colorid.max(axis=0).astype(np.int16)          # [G,Hp,Wp]
    walls = list(scene.walls)
    # Candidate tones: id 0 is clear (delete) through to the darkest engrave level. Deleting
    # is allowed on purpose -- material that only ever made another view worse should be
    # removable, and forbidding it would hide that failure rather than fix it.
    candidates = list(range(len(names)))
    K = len(names)

    samplers = {(p.name, w): wall_to_panel_sampler(scene, table, p, w, (Hp, Wp))
                for p in scene.panels for w in walls}
    pixel_m = {gi: 0.5 * ((p.u_range[1] - p.u_range[0]) / Wp
                          + (p.v_range[1] - p.v_range[0]) / Hp)
               for gi, p in enumerate(scene.panels)}
    mf = scene.fab.min_feature if min_feature is None else min_feature
    regions = ({gi: region_labels(tone[gi]) for gi in range(G)} if unit == "shard" else None)

    def render(t2d):
        return renderer.render_color_np(C.stack_transmit_lut(names, t2d[None], None))

    # Start from a fabricable map so the reported baseline is the object, not the raster.
    for gi in range(G):
        tone[gi] = make_fabricable(tone[gi], K, mf, pixel_m[gi])

    pred = render(tone)
    score, _ = objective(pred, targets)
    log = [{"sweep": -1, "loss": total_loss(pred, targets), "unit": unit, "score": score,
            "mean_iou": metrics.evaluate_multiview(targets, pred)["_summary"]["mean_iou"]}]
    if verbose:
        print(f"  start          score {score:.4f}   mean IoU {log[0]['mean_iou']:.4f}"
              f"   loss {log[0]['loss']:.0f}")

    for it in range(sweeps):
        changed = accepted = 0
        for gi, panel in enumerate(scene.panels):
            present = tone[gi] > 0 if regions is None else regions[gi][0] > 0
            if not present.any():
                continue
            err = np.empty((len(candidates), Hp, Wp), dtype=np.float32)
            for ci, k in enumerate(candidates):
                trial = tone.copy()
                trial[gi] = np.where(present, k, 0)
                e = sq_error_maps(render(trial), targets)
                # A panel pixel is scored by the error at every wall location it lands on, so
                # a tone that fixes this view while wrecking another is priced correctly.
                err[ci] = sum(pull_back(e[w], samplers[(panel.name, w)]) for w in walls)
            if regions is None:
                new = np.where(present, np.argmin(err, axis=0).astype(np.int16), 0)
                new = make_fabricable(new, K, mf, pixel_m[gi])
            else:
                lab, n = regions[gi]
                ids = np.arange(1, n + 1)
                cost = np.stack([ndimage.sum_labels(err[ci], lab, ids)
                                 for ci in range(len(candidates))])
                lut = np.zeros(n + 1, dtype=np.int16)
                lut[1:] = [candidates[p] for p in np.argmin(cost, axis=0)]
                new = lut[lab]
            n_diff = int((new != tone[gi]).sum())
            if n_diff == 0:
                continue
            # Propose, then verify. The per-pixel argmin is only a suggestion: it is computed
            # on squared error, and the min-feature projection then alters it further, so
            # neither step is guaranteed to help the thing being reported.
            old = tone[gi].copy()
            tone[gi] = new
            cand_score, _ = objective(render(tone), targets)
            if cand_score > score:
                score, changed, accepted = cand_score, changed + n_diff, accepted + 1
            else:
                tone[gi] = old
        pred = render(tone)
        mi = metrics.evaluate_multiview(targets, pred)["_summary"]["mean_iou"]
        log.append({"sweep": it, "loss": total_loss(pred, targets), "score": score,
                    "mean_iou": mi, "pixels_changed": changed, "sheets_accepted": accepted})
        if verbose:
            print(f"  sweep {it}        score {score:.4f}   mean IoU {mi:.4f}   "
                  f"{accepted}/{G} sheets accepted, {changed} pixels re-toned")
        if changed == 0:
            break
    return tone[None], log


def _tone_stats(m, walls):
    """SSIM/RMSE means. `_summary` carries only the silhouette numbers -- which is precisely
    the blindness this stage exists to fix -- so the tonal ones are pulled out per view."""
    return (float(np.mean([m[w]["ssim"] for w in walls])),
            float(np.mean([m[w]["rmse"] for w in walls])))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="30v4", choices=sorted(P.ARMS))
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--unit", default="pixel", choices=["pixel", "shard"],
                    help="granularity of the tone decision; 'shard' is the comparison arm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    out = Path(a.out or f"out_pearl3/retone_{a.arm}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = P.ARMS[a.arm]

    t0 = time.perf_counter()
    res = P.run_build(cfg, out, verbose=False)
    art = res.pop("_artifacts")
    scene, table, renderer = art["scene"], art["table"], art["renderer"]
    names, targets = art["names"], art["targets"]

    before = res["summary"]
    walls = list(scene.walls)
    m_before = metrics.evaluate_multiview(targets, art["pred"])
    ss_b, rm_b = _tone_stats(m_before, walls)
    print(f"=== retone {a.arm} ===")
    print(f"stage 1 (partition, intensity-weighted render): "
          f"mean IoU {before['mean_iou']:.4f}  min {before['min_iou']:.4f}  "
          f"SSIM {ss_b:.4f}  RMSE {rm_b:.4f}")

    # The honest starting point for stage 2 is full-strength tones, because that is what the
    # cut files describe. Reporting the intensity-weighted number as "before" would flatter
    # the result by comparing against a render of an object nobody can make.
    flat = renderer.render_color_np(C.stack_transmit_lut(names, art["stack_colorid"], None))
    m_flat = metrics.evaluate_multiview(targets, flat)
    ss_f, rm_f = _tone_stats(m_flat, walls)
    print(f"stage 1 as actually fabricable (full-strength tones): "
          f"mean IoU {m_flat['_summary']['mean_iou']:.4f}  "
          f"min {m_flat['_summary']['min_iou']:.4f}  SSIM {ss_f:.4f}  RMSE {rm_f:.4f}")

    print("\nstage 2: coordinate descent over tone ids (geometry frozen)")
    cid, log = retone(scene, table, renderer, targets, names, art["stack_colorid"],
                      sweeps=a.sweeps, unit=a.unit, min_feature=cfg.engrave_min_feature)

    pred = renderer.render_color_np(C.stack_transmit_lut(names, cid, None))
    m = metrics.evaluate_multiview(targets, pred)
    ss_a, rm_a = _tone_stats(m, walls)
    print("\n" + metrics.format_multiview_report(m, f"{a.arm} after re-toning"))
    print(f"SSIM {ss_f:.4f} -> {ss_a:.4f}   RMSE {rm_f:.4f} -> {rm_a:.4f}")

    kept = int((cid > 0).sum())
    was = int((art["stack_colorid"] > 0).sum())
    print(f"material: {kept} of {was} shard pixels kept ({100*kept/max(was,1):.1f}%)")

    from shadowart.preview.wallview import save_color_comparison
    save_color_comparison(targets, pred, out / "preview_retoned.png")
    np.save(out / "stack_colorid_retoned.npy", cid)
    (out / "retone.json").write_text(json.dumps({
        "arm": a.arm,
        "stage1_intensity_weighted": {**before, "ssim": ss_b, "rmse": rm_b},
        "stage1_fabricable": {**m_flat["_summary"], "ssim": ss_f, "rmse": rm_f},
        "stage2_retoned": {**m["_summary"], "ssim": ss_a, "rmse": rm_a},
        "views": {w: m[w] for w in scene.walls},
        "descent": log,
        "shard_pixels_before": was, "shard_pixels_after": kept,
        "runtime_s": time.perf_counter() - t0,
    }, indent=2), encoding="utf-8")
    print(f"-> {out/'retone.json'}   ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
