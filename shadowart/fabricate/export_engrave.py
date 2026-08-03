"""Cut files for the *laser-engraved clear perspex* build.

This is a different machine job from `export_color`, and the difference is the whole point
of the 30x30 rebuild. `export_color` was written for dyed acrylic: it sorted shards by colour
and emitted one file per colour, because each colour was a physically different sheet and the
operator cut each stack separately. Engraved perspex inverts that. Every panel is a single
clear sheet that goes through the machine exactly once; the tone is produced by *how hard the
laser fires over a region*, not by which sheet you loaded. So the natural unit is the panel,
not the tone, and each panel file carries:

    CUT_OUTLINE   carrier rectangle          -- through-cut, once
    CUT_SLOT      cross-lap weave slots      -- through-cut, once
    ENG_L / ENG_D / ENG_K   shard regions    -- raster engrave, one pass each, ascending power

Ordering matters on the machine and is baked into the layer order below: engrave first while
the sheet is still a rigid rectangle held flat by its own stiffness, cut last. Cutting first
would leave a slotted, floppy carrier that lifts off the bed and defocuses the engrave.

The tone -> power mapping is deliberately NOT written here. Engraver response is
machine-, material- and lens-specific, and inventing numbers would be worse than useless
because they look authoritative. `manifest()` emits the target transmittance for each layer
and points the operator at the step-wedge calibration in `targets/engrave.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import svgwrite

from ..raster2vec.contours import polygon_rings

# aci colours; also the machine ordering (engrave passes, then the through-cuts)
_ENGRAVE_ACI = {"ENG_L": 4, "ENG_D": 3, "ENG_K": 2}
_CUT_ACI = {"CUT_OUTLINE": 5, "CUT_SLOT": 1}


def _mm(ring, u0, v0):
    return [((u - u0) * 1000.0, (v - v0) * 1000.0) for u, v in ring]


def split_by_tone(pieces, tone_names):
    """[(poly, channel, slot)] -> {tone_name: [poly]}, keeping only known engrave tones.

    Shards land on stack slots, but the engrave palette gives each shard exactly one tone
    (a laser pass cannot be laminated onto itself), so slots collapse away here. Anything
    with an unrecognised channel is dropped rather than silently engraved at the wrong power.
    """
    out = {t: [] for t in tone_names}
    for item in pieces:
        poly, channel = item[0], item[1]
        if channel in out:
            out[channel].append(poly)
    return out


def export_panel(drawing, tone_polys, out_dir, formats=("dxf", "svg")):
    """One panel -> one DXF and/or SVG carrying cut layers + one layer per engrave tone."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "dxf" in formats:
        written.append(_dxf(drawing, tone_polys, out_dir / f"panel_{drawing.name}.dxf"))
    if "svg" in formats:
        written.append(_svg(drawing, tone_polys, out_dir / f"panel_{drawing.name}.svg"))
    return written


def _dxf(drawing, tone_polys, path):
    doc = ezdxf.new()
    doc.units = ezdxf.units.MM
    for name, aci in list(_ENGRAVE_ACI.items()) + list(_CUT_ACI.items()):
        doc.layers.add(name, color=aci)
    msp = doc.modelspace()
    u0, v0 = drawing.u_range[0], drawing.v_range[0]

    for tone in _ENGRAVE_ACI:                     # engrave passes first
        for poly in tone_polys.get(tone, []):
            ext, holes = polygon_rings(poly)
            msp.add_lwpolyline(_mm(ext, u0, v0), close=True, dxfattribs={"layer": tone})
            for h in holes:
                msp.add_lwpolyline(_mm(h, u0, v0), close=True, dxfattribs={"layer": tone})

    for s in drawing.slots:                       # then the through-cuts
        msp.add_lwpolyline(_mm(s, u0, v0), close=True, dxfattribs={"layer": "CUT_SLOT"})
    msp.add_lwpolyline(_mm(drawing.outline, u0, v0), close=True,
                       dxfattribs={"layer": "CUT_OUTLINE"})
    doc.saveas(path)
    return str(path)


def _svg(drawing, tone_polys, path, levels=None):
    """Preview/plot SVG. Fill greys are the *target transmittance*, so the file reads as a
    picture of the finished panel held up to the light rather than as an abstract layer soup."""
    u0, u1 = drawing.u_range
    v0, v1 = drawing.v_range
    w_mm, h_mm = (u1 - u0) * 1000.0, (v1 - v0) * 1000.0
    dwg = svgwrite.Drawing(str(path), size=(f"{w_mm}mm", f"{h_mm}mm"),
                           viewBox=f"0 0 {w_mm} {h_mm}")

    def pts(ring):
        return [((u - u0) * 1000.0, (v1 - v) * 1000.0) for u, v in ring]   # flip v: SVG y is down

    from ..targets import engrave as _eng
    levels = _eng.DEFAULT_LEVELS if levels is None else levels
    shade = {n: int(round(255 * t)) for n, t in zip(_eng.LEVEL_NAMES, levels)}

    for tone in _ENGRAVE_ACI:
        polys = tone_polys.get(tone, [])
        if not polys:
            continue
        g = shade.get(tone, 128)
        grp = dwg.g(id=tone, fill=f"rgb({g},{g},{g})", stroke="none")
        for poly in polys:
            ext, holes = polygon_rings(poly)
            grp.add(dwg.polygon(points=pts(ext)))
            for h in holes:
                grp.add(dwg.polygon(points=pts(h), fill="white"))
        dwg.add(grp)

    cut = dwg.g(id="CUT", fill="none", stroke="red", stroke_width=0.2)
    for s in drawing.slots:
        cut.add(dwg.polygon(points=pts(s)))
    cut.add(dwg.polygon(points=pts(drawing.outline), stroke="blue"))
    dwg.add(cut)
    dwg.save()
    return str(path)


def export_all(drawings, pieces_by_panel, tone_names, out_dir, formats=("dxf", "svg"),
               levels=None):
    """Whole build -> per-panel cut files + manifest.json. Returns (paths, counts)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, counts = [], {}
    for d in drawings:
        tones = split_by_tone(pieces_by_panel.get(d.name, []), tone_names)
        if "dxf" in formats:
            paths.append(_dxf(d, tones, out_dir / f"panel_{d.name}.dxf"))
        if "svg" in formats:
            paths.append(_svg(d, tones, out_dir / f"panel_{d.name}.svg", levels))
        counts[d.name] = {t: len(p) for t, p in tones.items()}
    man = manifest(drawings, counts, tone_names, levels)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
    paths.append(str(out_dir / "manifest.json"))
    return paths, counts


def manifest(drawings, counts, tone_names, levels=None):
    from ..targets import engrave as _eng
    levels = _eng.DEFAULT_LEVELS if levels is None else levels
    return {
        "material": "clear cast acrylic (perspex)",
        "process": "raster engrave for tone, vector through-cut for outline and slots",
        "machine_order": ["ENG_L", "ENG_D", "ENG_K", "CUT_SLOT", "CUT_OUTLINE"],
        "order_rationale": ("engrave while the sheet is still a rigid rectangle; cutting the "
                            "slots first leaves a floppy carrier that lifts and defocuses"),
        "tones": [
            {"layer": n, "target_transmittance": float(t),
             "note": "power/speed is machine-specific -- calibrate with the step wedge in "
                     "shadowart/targets/engrave.py before running the real sheets"}
            for n, t in zip(tone_names, levels)
        ],
        "unengraved": {"layer": None, "target_transmittance": 1.0,
                       "note": "clear perspex, left untouched"},
        "panels": [{"name": d.name,
                    "size_mm": [round((d.u_range[1] - d.u_range[0]) * 1000, 2),
                                round((d.v_range[1] - d.v_range[0]) * 1000, 2)],
                    "slots": len(d.slots),
                    "engrave_regions": counts.get(d.name, {})}
                   for d in drawings],
    }
