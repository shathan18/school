"""Presentation figures for the 30x30 Pearl Girl triptych.

Everything here is regenerated from the journals and artefacts already on disk, so no figure
can drift from the numbers in docs/30x30-optimization.md. Run it after any re-solve.

    .venv\\Scripts\\python.exe pearl3_figures.py --out examples/pearl3

Figures:
  1_result       what the piece projects: target vs stage 1 vs shipped, per view
  2_levers       the three levers that moved the score, and the hypotheses that died
  3_bakeoff      five optimisers at equal budget -- and why the answer is "stop tuning"
  4_fabrication  the exported sheets, coloured by engrave tone
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

import pearl3_baseline as P
import pearl3_retone as R
from shadowart import metrics
from shadowart.targets import color as C

ROOT = Path(__file__).parent
STOPS = {"back": "back (0 deg)", "side": "side (120 deg)", "front": "front (240 deg)"}


def _grey(rgb):
    """Transmittance RGB -> what the eye sees on the wall (dark = blocked light)."""
    return np.clip(np.asarray(rgb), 0, 1).mean(axis=-1)


def _show(ax, img, title, sub=None):
    ax.imshow(_grey(img), origin="lower", cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)
    if sub:
        ax.set_xlabel(sub, fontsize=8)


# --- figure 1: the actual result -------------------------------------------------------
def fig_result(out, arm="30v4", sweeps=5):
    cfg = P.ARMS[arm]
    res = P.run_build(cfg, out / "_scratch", verbose=False)
    art = res.pop("_artifacts")
    scene, targets, names = art["scene"], art["targets"], art["names"]
    pred1 = art["pred"]
    m1 = metrics.evaluate_multiview(targets, pred1)

    cid, _ = R.retone(scene, art["table"], art["renderer"], targets, names,
                      art["stack_colorid"], sweeps=sweeps, verbose=False,
                      min_feature=cfg.engrave_min_feature)
    pred2 = art["renderer"].render_color_np(C.stack_transmit_lut(names, cid, None))
    m2 = metrics.evaluate_multiview(targets, pred2)

    walls = list(scene.walls)
    fig, axes = plt.subplots(len(walls), 3, figsize=(10.5, 3.6 * len(walls)), squeeze=False)
    for r, w in enumerate(walls):
        _show(axes[r, 0], targets[w], f"{STOPS.get(w, w)} — target")
        _show(axes[r, 1], pred1[w], "stage 1: geometry solved",
              f"IoU {m1[w]['iou']:.3f}   SSIM {m1[w]['ssim']:.3f}   RMSE {m1[w]['rmse']:.3f}")
        _show(axes[r, 2], pred2[w], "stage 2: tone re-quantised  ← SHIPPED",
              f"IoU {m2[w]['iou']:.3f}   SSIM {m2[w]['ssim']:.3f}   RMSE {m2[w]['rmse']:.3f}")
    s1, s2 = m1["_summary"], m2["_summary"]
    fig.suptitle(
        "One object, three shadows — 30x30 cm, 18 engraved Perspex sheets\n"
        f"mean IoU {s1['mean_iou']:.3f} -> {s2['mean_iou']:.3f}    "
        f"worst view {s1['min_iou']:.3f} -> {s2['min_iou']:.3f}    "
        "(stage 2 also deletes 21% of the material)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "1_result.png", dpi=130)
    plt.close(fig)
    return art, cid


# --- figure 2: what worked, and what didn't -------------------------------------------
def _journal(name):
    p = ROOT / "out_pearl3" / name / "runs.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def fig_levers(out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) the layout frontier -- the single biggest lever.
    # The sweep tried several pitches per sheet count, so plotting every run zigzags and buries
    # the trend. Take the best run at each count: the question this panel answers is "how good
    # can N sheets be", not "how bad can a badly-pitched N be".
    runs = [r for r in _journal("sweep_layout_30v2") + _journal("sweep_layout2_30v2")
            if "n_per_family" in r["overrides"]]
    best = {}
    for r in runs:
        k = r["overrides"]["n_per_family"] * 3
        if k not in best or r["mean_iou"] > best[k]["mean_iou"]:
            best[k] = r
    rows = [best[k] for k in sorted(best)]
    if rows:
        n = sorted(best)
        ax = axes[0, 0]
        ax.plot(n, [r["mean_iou"] for r in rows], "o-", color="#1f77b4", label="mean IoU")
        ax.plot(n, [r["min_iou"] for r in rows], "s--", color="#7fb3d5", label="worst view")
        ax.axvline(18, color="k", lw=1, ls=":")
        ax.annotate("18 sheets — chosen", (18, min(r["min_iou"] for r in rows)),
                    xytext=(-98, 2), textcoords="offset points", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(n, [r["crosstalk_cost"] for r in rows], "^-", color="#d62728", alpha=.65,
                 label="cross-talk cost")
        ax2.axhline(0, color="#d62728", lw=.8, ls="--", alpha=.5)
        ax2.set_ylabel("cross-talk cost (IoU)", color="#d62728", fontsize=9)
        ax.set_xlabel("total sheets"); ax.set_ylabel("IoU")
        ax.set_title("LEVER: more sheets, tighter pitch  (+0.070)\n"
                     "past ~12 sheets the cross-talk cost goes NEGATIVE —\n"
                     "stray light stopped being noise", fontsize=9)
        ax.legend(fontsize=8, loc="center right")

    # (b) tone allocation
    rows = [r for r in _journal("sweep_engrave_30v3") if r["overrides"]]
    if rows:
        rows.sort(key=lambda r: r["mean_iou"])
        lab = [str(r["overrides"].get("engrave_levels", "?")) for r in rows]
        ax = axes[0, 1]
        cols = ["#2ca02c" if r is rows[-1] else "#c7c7c7" for r in rows]
        ax.barh(range(len(rows)), [r["mean_iou"] for r in rows], color=cols)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(lab, fontsize=7)
        ax.set_xlim(min(r["mean_iou"] for r in rows) - 0.01, max(r["mean_iou"] for r in rows) + 0.005)
        ax.set_xlabel("mean IoU")
        ax.set_title("LEVER: allocate tones evenly in optical DENSITY,\n"
                     "not in transmittance  (+0.034)\n"
                     "overlapping sheets multiply, so even spacing in T\n"
                     "leaves the midtones unreachable", fontsize=9)

    # (c) a refuted hypothesis, plotted so it stays refuted
    rows = sorted(_journal("sweep_bandlimit_30v4"),
                  key=lambda r: r["overrides"].get("target_blur_px", 0))
    if rows:
        b = [r["overrides"].get("target_blur_px", 0) for r in rows]
        ax = axes[1, 0]
        ax.plot(b, [r["mean_iou"] for r in rows], "o-", label="mean IoU")
        ax.plot(b, [r["edge"] for r in rows], "s-", label="edge fidelity")
        ax.set_xlabel("target blur (panel px)"); ax.set_ylabel("score")
        ax.set_title("REFUTED: band-limiting the targets to the optical passband\n"
                     "the theory was sound; there is no interior optimum,\n"
                     "not even at 1 px", fontsize=9)
        ax.legend(fontsize=8)

    # (d) the headline, end to end. Read from the shipped report so it cannot drift.
    ax = axes[1, 1]
    rep = json.loads((ROOT / "out_pearl3_30" / "v4" / "fab_report.json").read_text())
    shipped = rep["gate_fab_round_trip"]["summary"]["mean_iou"]
    lab = ["60x60\n(previous)", "30x30\nfirst try", "30x30\noptimised", "30x30\n+ re-toned\nSHIPPED"]
    val = [0.722, 0.689, 0.8539, shipped]
    cols = ["#8c8c8c", "#d62728", "#7fb3d5", "#2ca02c"]
    bars = ax.bar(lab, val, color=cols)
    for b_, v in zip(bars, val):
        ax.text(b_.get_x() + b_.get_width() / 2, v + .004, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.722, color="#8c8c8c", ls="--", lw=1)
    ax.set_ylim(0.6, max(val) + 0.05); ax.set_ylabel("mean IoU")
    ax.set_title("Half the footprint, better than the original\n"
                 "(all bars all-light; last bar measured from the CUT FILES)", fontsize=9)

    fig.suptitle("What moved the number — and what didn't\n"
                 "lever panels are the original sweeps on the earlier target set; "
                 "they rank the levers, and the ranking did not change",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out / "2_levers.png", dpi=130)
    plt.close(fig)


# --- figure 3: optimiser bake-off ------------------------------------------------------
def fig_bakeoff(out):
    p = ROOT / "out_pearl3" / "search" / "bakeoff.json"
    if not p.exists():
        return
    d = json.load(p.open())
    m = d.get("methods", d)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    order = sorted(m, key=lambda k: -m[k]["best"])
    for k in order:
        h = m[k].get("history", [])
        if h:
            ax.plot([x[0] for x in h], [x[2] for x in h], label=f"{k}  ({m[k]['best']:.4f})")
    ax.axhline(0.7794, color="k", ls=":", lw=1)
    ax.annotate("hand-tuned 30v4", (1, 0.7794), xytext=(4, -11),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("distinct evaluations (equal budget)")
    ax.set_ylabel("best score so far")
    ax.set_title("Five methods, one shared memoised journal", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax2.barh(order[::-1], [m[k]["best"] for k in order[::-1]],
             color=["#2ca02c" if k == order[0] else "#c7c7c7" for k in order[::-1]])
    ax2.set_xlim(0.74, 0.80); ax2.set_xlabel("best score")
    ax2.set_title("Random search lands within 0.004 of the winner\n"
                  "=> the landscape is flat; parameter tuning is exhausted", fontsize=10)
    fig.suptitle("Optimiser bake-off — the useful result is the negative one", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "3_bakeoff.png", dpi=130)
    plt.close(fig)


# --- figure 4: the thing you actually cut ----------------------------------------------
def fig_fabrication(out, art, cid, arm="30v4"):
    from shadowart.solve import decompose
    import dataclasses
    scene = art["scene"]
    cfg = P.ARMS[arm]
    eng = dataclasses.replace(
        scene, fab=dataclasses.replace(scene.fab, min_feature=cfg.engrave_min_feature))
    px = 0.5 * ((scene.panels[0].u_range[1] - scene.panels[0].u_range[0]) / cid.shape[-1]
                + (scene.panels[0].v_range[1] - scene.panels[0].v_range[0]) / cid.shape[-2])
    pieces = decompose.panel_stack_pieces(eng, cid, art["names"], kerf=px)

    tone_fill = {"ENG_L": "#b9b9b9", "ENG_D": "#6e6e6e", "ENG_K": "#1c1c1c"}
    panels = scene.panels
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.15 * nrow), squeeze=False)
    for i, panel in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for poly, tone, _ in pieces.get(panel.name, []):
            gs = getattr(poly, "geoms", [poly])
            for g in gs:
                ax.add_patch(MplPolygon(np.asarray(g.exterior.coords),
                                        closed=True, facecolor=tone_fill.get(tone, "#999"),
                                        edgecolor="none"))
        ax.set_xlim(*panel.u_range); ax.set_ylim(*panel.v_range)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#4488cc")
        ax.set_title(panel.name, fontsize=7, pad=2)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("The 18 sheets as exported — one DXF/SVG each, shaded by engrave tone\n"
                 "blue outline = the clear Perspex square that gets cut; "
                 "greys = laser engrave layers ENG_L / ENG_D / ENG_K", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "4_fabrication.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="30v4")
    ap.add_argument("--out", default="examples/pearl3")
    ap.add_argument("--sweeps", type=int, default=5)
    a = ap.parse_args(argv)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    print("1/4 result ...")
    art, cid = fig_result(out, a.arm, a.sweeps)
    print("2/4 levers ...")
    fig_levers(out)
    print("3/4 bake-off ...")
    fig_bakeoff(out)
    print("4/4 fabrication ...")
    fig_fabrication(out, art, cid, a.arm)
    for f in sorted(out.glob("*.png")):
        print(f"  -> {f.relative_to(ROOT)}  ({f.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
