"""Turn a solved build into a package a laser cutter can actually run.

`pearl3_fab.py` names its output after the solver's internal panel ids -- `panel_F0_0.dxf`,
`panel_F2_1.svg`. Those are correct and meaningless: nothing in the name says which of the
three views the sheet serves, which way it is oriented, or where it sits in the stack, and a
six-sheet weave assembles exactly one way. This renames every file to carry that, and writes
the cut sheet next to it.

    .venv\\Scripts\\python.exe pearl3_package.py --arm 30v6 --src out_pearl3_30/v6

Reads only what is already on disk plus the scene geometry (which is cheap -- no solve), so it
can be re-run any time without touching the result.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from pearl3_baseline import ARMS, build_scene, _family_angles

ROOT = Path(__file__).parent

# Files that are deliverables in their own right rather than per-sheet cut data.
EXTRAS = {
    "scene_interactive.html": ("{p}_3d_scene_interactive.html",
                               "3D view: sheets, lamps, and the three projected walls"),
    "scene_panels.html":      ("{p}_3d_sheets_only.html",
                               "3D view: the sheets alone, no walls -- easier to read the weave"),
    "preview_views.png":      ("{p}_projections_preview.png",
                               "what the three shadows look like, full resolution"),
    "fab_report.json":        ("{p}_verification_report.json",
                               "all three acceptance gates, with per-view metrics"),
    "metrics.json":           ("{p}_metrics.json", "stage-1 solver metrics"),
}


def _stack_offset_mm(panel, angle_rad) -> float:
    """Signed distance of the sheet from the turntable axis, along its own pitch direction."""
    n = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
    return float(np.dot(np.asarray(panel.anchor), n)) * 1000.0


def plan(arm: str, src: Path):
    """Work out the human-readable name for every sheet. Returns a list of dicts."""
    cfg = ARMS[arm]
    scene = build_scene(cfg)
    angles = _family_angles(cfg)
    report = json.loads((src / "fab_report.json").read_text())
    man = json.loads((src / "cut" / "manifest.json").read_text())

    served = {p["panel"]: p for p in report["gate_ablation"]["per_panel"]}
    regions = {p["name"]: p for p in man["panels"]}
    prefix = f"pearl{len(scene.panels)}"

    rows = []
    for panel in scene.panels:
        fi = int(panel.name.split("_")[0][1:])
        ang = angles[fi]
        off = _stack_offset_mm(panel, panel.angle)
        abl = served.get(panel.name, {})
        rows.append({
            "id": panel.name,
            "view": abl.get("best_view", "?"),
            "angle_deg": ang,
            "offset_mm": off,
            "drop": abl.get("mean_iou_drop", 0.0),
            "size_mm": regions[panel.name]["size_mm"],
            "slots": regions[panel.name]["slots"],
            "engrave": regions[panel.name]["engrave_regions"],
        })

    # Number the sheets grouped by the view they serve, then by position in the stack, so the
    # cut order and the assembly order are the same order.
    order = {v: i for i, v in enumerate(cfg.views)}
    rows.sort(key=lambda r: (order.get(r["view"], 9), r["offset_mm"]))
    for i, r in enumerate(rows, 1):
        sign = "-" if r["offset_mm"] < 0 else "+"
        r["seq"] = i
        r["stem"] = (f"{prefix}_S{i:02d}_{r['view'].upper()}"
                     f"_ang{round(r['angle_deg']):03d}"
                     f"_off{sign}{abs(round(r['offset_mm'])):03d}mm")
    return cfg, scene, man, report, rows, prefix


def write_package(arm: str, src: Path, dst: Path):
    cfg, scene, man, report, rows, prefix = plan(arm, src)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "cut").mkdir(exist_ok=True)

    for r in rows:
        for ext in ("dxf", "svg"):
            s = src / "cut" / f"panel_{r['id']}.{ext}"
            if s.exists():
                shutil.copy2(s, dst / "cut" / f"{r['stem']}.{ext}")

    for name, (pat, _) in EXTRAS.items():
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / pat.format(p=prefix))

    # The OBJ names its material library by filename, so renaming one means patching the other.
    obj, mtl = src / "assembly.obj", src / "assembly.mtl"
    if obj.exists():
        new_mtl = f"{prefix}_3d_assembly.mtl"
        text = obj.read_text()
        if mtl.exists():
            shutil.copy2(mtl, dst / new_mtl)
            text = text.replace("mtllib assembly.mtl", f"mtllib {new_mtl}")
        (dst / f"{prefix}_3d_assembly.obj").write_text(text)

    (dst / "CUTTING.md").write_text(cut_sheet(cfg, scene, man, report, rows, prefix),
                                    encoding="utf-8")
    return dst, rows


def cut_sheet(cfg, scene, man, report, rows, prefix) -> str:
    g = report["geometry"]
    gate_b = report["gate_fab_round_trip"]["summary"]
    # The summary carries only the IoU pair; the rest is a mean over the per-view block.
    views = report["gate_fab_round_trip"]["views"]
    avg = {k: sum(v[k] for v in views.values()) / len(views) for k in ("ssim", "edge_fidelity")}
    tones = {t["layer"]: t["target_transmittance"] for t in man["tones"]}
    total = sum(sum(r["engrave"].values()) for r in rows)

    L = []
    a = L.append
    a(f"# Cut package - {cfg.name}\n")
    a(f"{len(scene.panels)} sheets of **3 mm clear cast acrylic (Perspex)**, "
      f"each **{man['panels'][0]['size_mm'][0]:.1f} x {man['panels'][0]['size_mm'][1]:.1f} mm**.\n")
    a("Every sheet is a different cut. They are not interchangeable, and the weave assembles "
      "exactly one way.\n")

    a("\n## Sheet list\n")
    a("| # | file (`.dxf` and `.svg`) | serves | sheet angle | offset from axis | slots | engraved regions |")
    a("| - | ------------------------ | ------ | ----------- | ---------------- | ----- | ---------------- |")
    for r in rows:
        n = sum(r["engrave"].values())
        a(f"| {r['seq']} | `{r['stem']}` | {r['view']} | {r['angle_deg']:.0f}\u00b0 | "
          f"{r['offset_mm']:+.0f} mm | {r['slots']} | {n} |")
    a(f"\n{total} engraved regions in total. Offset is measured from the turntable axis, "
      "perpendicular to the sheet.\n")

    a("\n## Layers, and the order to run them\n")
    a("Both formats carry the same five layers. **Run them in this order.**\n")
    a("| order | layer | operation | target transmittance | appearance |")
    a("| ----- | ----- | --------- | -------------------- | ---------- |")
    appear = {"ENG_L": "lightest engrave", "ENG_D": "mid engrave", "ENG_K": "darkest engrave"}
    for i, layer in enumerate(man["machine_order"], 1):
        if layer in tones:
            a(f"| {i} | `{layer}` | raster engrave | {tones[layer]:.2f} | {appear[layer]} |")
        else:
            op = "vector through-cut"
            note = ("the four joint slots" if layer == "CUT_SLOT" else "the outer square, last")
            a(f"| {i} | `{layer}` | {op} | - | {note} |")
    a("\nAnything on no layer stays **clear, unengraved** (transmittance 1.0).\n")
    a(f"> **Engrave first, cut last.** {man['order_rationale'].capitalize()}.\n")
    a("> **Calibrate before the real sheets.** The three transmittances above are optical "
      "targets, not power/speed settings - those are specific to your machine and to the "
      "acrylic. Engrave a step wedge on scrap, measure it, and match the three tones. Getting "
      "these wrong is the one error the geometry cannot absorb.\n")

    a("\n## Assembly\n")
    a(f"The sheets interlock into three families of {cfg.n_per_family}, one family per view, "
      f"at **{cfg.pitch*1000:.0f} mm** pitch:\n")
    fam = {}
    for r in rows:
        fam.setdefault((r["view"], r["angle_deg"]), []).append(r)
    for (view, ang), members in fam.items():
        ids = ", ".join(f"S{m['seq']:02d} ({m['offset_mm']:+.0f} mm)" for m in members)
        a(f"- **{view}** - sheets at {ang:.0f}\u00b0: {ids}")
    a(f"\nSlots are cut `thickness / sin \u03b8` wide, so the widest is "
      f"{report['gate_weave']['max_slot_width_mm']:.2f} mm for 3.0 mm material - that is correct, "
      "not an error. A slot at an angle has to be wider than the sheet is thick.\n")
    a(f"Assembled footprint **{g['footprint_m'][0]*100:.1f} x {g['footprint_m'][1]*100:.1f} cm**, "
      f"swept circle {g['swept_diameter_m']*100:.0f} cm.\n")
    a("Open the interactive 3D file below before assembling - it is far quicker than reading "
      "coordinates.\n")

    a("\n## Installation geometry\n")
    a("| | |")
    a("| - | - |")
    a(f"| lamp behind the piece | {g['lamp_to_body_m']:.3f} m |")
    a(f"| piece to wall | {g['body_to_wall_m']:.3f} m |")
    a(f"| lamp height | {g['lamp_height_m']:.3f} m |")
    a(f"| projected image | {cfg.image:.2f} m square ({g['magnification']:.2f}x) |")
    a(f"| viewing stops | {', '.join(f'{s:.0f}' for s in cfg.stops_deg)}\u00b0 "
      f"-> {', '.join(cfg.views)} |")
    a("\nThe lamp must be a **small, bright point source**. A large or diffuse source blurs "
      "every edge by the penumbra "
      f"({min(g['penumbra_sigma_mm'].values()):.1f}-{max(g['penumbra_sigma_mm'].values()):.1f} mm "
      "at the wall as designed) and the tonal structure goes with it.\n")

    a("\n## Other files\n")
    a("| file | what it is |")
    a("| ---- | ---------- |")
    a(f"| `{prefix}_3d_assembly.obj` + `.mtl` | the assembled piece as 3D geometry |")
    for name, (pat, desc) in EXTRAS.items():
        a(f"| `{pat.format(p=prefix)}` | {desc} |")

    a("\n## Expected result\n")
    a(f"Measured by re-rendering **from these cut files**: mean IoU "
      f"**{gate_b['mean_iou']:.4f}**, worst view **{gate_b['min_iou']:.4f}**, "
      f"SSIM {avg['ssim']:.4f}, edge fidelity {avg['edge_fidelity']:.4f}.\n")
    a("With six sheets there is no redundancy: single-panel ablation costs between "
      f"{min(r['drop'] for r in rows):.3f} and {max(r['drop'] for r in rows):.3f} mean IoU. "
      "Every sheet is load-bearing, so a mis-cut sheet is not a cosmetic problem.\n")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="30v6", choices=sorted(ARMS))
    ap.add_argument("--src", default="out_pearl3_30/v6")
    ap.add_argument("--out", default=None, help="default: <src>/deliverables")
    a = ap.parse_args(argv)

    src = ROOT / a.src
    dst = ROOT / a.out if a.out else src / "deliverables"
    dst, rows = write_package(a.arm, src, dst)

    print(f"{len(rows)} sheets -> {dst.relative_to(ROOT)}")
    for r in rows:
        print(f"  S{r['seq']:02d}  {r['stem']}.dxf / .svg")
    for f in sorted(dst.glob("*")):
        if f.is_file():
            print(f"  ---  {f.name}  ({f.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
