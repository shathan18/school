THREE FACES on THREE WALLS -- FINAL GRAYSCALE 60x60 build (genuine multiplexing)
==============================================================================

Panels: 5 clear-perspex sheets, each 60x60 cm x 3mm, in an EGG-CRATE GRID -- 2 parallel families
        (2 sheets at 60 deg + 3 sheets at 150 deg, 14 cm pitch). Parallel within a family so every
        intersection is ONE-from-each-family = strictly 2-WAY (max 2 panels per slot; no 3-way).
        3 lights at 120 deg, 3x magnification, walls 1.8 m. Sharp (small) light source.
Shards: 702 total -- light 389, medium 256, black 57. Min shard 5x5 mm (light
        grey won't show smaller). Cell 5 mm.
Palette: clear (white) + 3 dye tones. Per your dye tests, tones defined by stacking-to-black:
         light (~6-8 layers=black), medium (~3-4), black (darkest, ~2). Optical densities ADD when
         shards stack across the depth-panels, so overlaps read darker/true-black.

GENUINE MULTIPLEXING: the SAME shards build all 3 faces; each lit by only its own lamp shows a
 different expression. Removing shards degrades multiple faces (not just one).

WHY GRAYSCALE (this vs the earlier binary build): direct tone (light/medium/black) reproduces the
 faces instead of forcing tone through noisy binary stippling. IoU 0.64 -> ~0.78, MSE halved, and
 the leftover texture is INSIDE features (background stays clean). Grounded in Baran et al. 2012
 (layered attenuators for multiple grayscale shadows) and ShadowPix 2012 (halftoning).

FABRICATION
 1. cut_sheets/<panel>.svg or .dxf -- each has 3 LAYERS (light/medium/black). Laser-engrave the
    shard outlines on a 60x60 clear sheet.
 2. Hand-colour each region with its tone's alcohol-ink dye (dither to hit the tone).
 3. Mount each panel at its azimuth (see .obj). Clear (white) regions get no dye.
