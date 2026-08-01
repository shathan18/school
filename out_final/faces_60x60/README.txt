THREE FACES on THREE WALLS -- 60x60 single-blue-dye build
========================================================

Shards: 42 total on 14 of 24 panels (threshold theta=0.1).
Panels: 24 clear-perspex sheets, each 60 x 60 cm x 3 mm, on a symmetric 120-deg 3-wall rig
        at 3x magnification (walls 1.8 m, ~3 m room). See faces_60x60_shards.obj for the layout.

FABRICATION
 1. cut_sheets/<panel>.dxf (or .svg) -- laser-cut the blue shard outlines from a 60x60 clear sheet.
    Each closed outline = one shard. Units are millimetres; the 600x600 rectangle is the sheet.
 2. Dye every shard with the SAME blue Pentart ink (one dip). Tone on the walls comes from how
    many dyed shards a light ray passes through across the 24 depth-panels -- NOT from layering.
 3. Mount each panel's shards on its 60x60 carrier at the depth/angle in the .obj, in rig order.

NOTES
 * Shard shapes depend only on theta, not on how dark the dye is -- these files are final for any
   ink strength; dye darkness only changes how deep-blue the projected faces look.
 * Expressions (neutral/happy/sad) read distinctly; faces are intentionally blocky (2cm laser
   feature + single dye + 3x magnification). ~5% blue corner ghosting is inherent to 3 shared lights.
