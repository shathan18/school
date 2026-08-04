# Cut package - 30x30, 6 sheets (2 per family)

6 sheets of **3 mm clear cast acrylic (Perspex)**, each **295.8 x 295.8 mm**.

Every sheet is a different cut. They are not interchangeable, and the weave assembles exactly one way.


## Sheet list

| # | file (`.dxf` and `.svg`) | serves | sheet angle | offset from axis | slots | engraved regions |
| - | ------------------------ | ------ | ----------- | ---------------- | ----- | ---------------- |
| 1 | `pearl6_S01_BACK_ang075_off-025mm` | back | 75° | -25 mm | 4 | 990 |
| 2 | `pearl6_S02_BACK_ang075_off+025mm` | back | 75° | +25 mm | 4 | 647 |
| 3 | `pearl6_S03_SIDE_ang135_off-025mm` | side | 135° | -25 mm | 4 | 768 |
| 4 | `pearl6_S04_SIDE_ang135_off+025mm` | side | 135° | +25 mm | 4 | 532 |
| 5 | `pearl6_S05_FRONT_ang015_off-025mm` | front | 15° | -25 mm | 4 | 32 |
| 6 | `pearl6_S06_FRONT_ang015_off+025mm` | front | 15° | +25 mm | 4 | 660 |

3629 engraved regions in total. Offset is measured from the turntable axis, perpendicular to the sheet.


## Layers, and the order to run them

Both formats carry the same five layers. **Run them in this order.**

| order | layer | operation | target transmittance | appearance |
| ----- | ----- | --------- | -------------------- | ---------- |
| 1 | `ENG_L` | raster engrave | 0.85 | lightest engrave |
| 2 | `ENG_D` | raster engrave | 0.62 | mid engrave |
| 3 | `ENG_K` | raster engrave | 0.35 | darkest engrave |
| 4 | `CUT_SLOT` | vector through-cut | - | the four joint slots |
| 5 | `CUT_OUTLINE` | vector through-cut | - | the outer square, last |

Anything on no layer stays **clear, unengraved** (transmittance 1.0).

> **Engrave first, cut last.** Engrave while the sheet is still a rigid rectangle; cutting the slots first leaves a floppy carrier that lifts and defocuses.

> **Calibrate before the real sheets.** The three transmittances above are optical targets, not power/speed settings - those are specific to your machine and to the acrylic. Engrave a step wedge on scrap, measure it, and match the three tones. Getting these wrong is the one error the geometry cannot absorb.


## Assembly

The sheets interlock into three families of 2, one family per view, at **50 mm** pitch:

- **back** - sheets at 75°: S01 (-25 mm), S02 (+25 mm)
- **side** - sheets at 135°: S03 (-25 mm), S04 (+25 mm)
- **front** - sheets at 15°: S05 (-25 mm), S06 (+25 mm)

Slots are cut `thickness / sin θ` wide, so the widest is 3.46 mm for 3.0 mm material - that is correct, not an error. A slot at an angle has to be wider than the sheet is thick.

Assembled footprint **29.9 x 29.9 cm**, swept circle 30 cm.

Open the interactive 3D file below before assembling - it is far quicker than reading coordinates.


## Installation geometry

| | |
| - | - |
| lamp behind the piece | 0.493 m |
| piece to wall | 2.507 m |
| lamp height | 0.874 m |
| projected image | 1.80 m square (6.09x) |
| viewing stops | 15, 135, 255° -> back, side, front |

The lamp must be a **small, bright point source**. A large or diffuse source blurs every edge by the penumbra (3.6-3.9 mm at the wall as designed) and the tonal structure goes with it.


## Other files

| file | what it is |
| ---- | ---------- |
| `pearl6_3d_assembly.obj` + `.mtl` | the assembled piece as 3D geometry |
| `pearl6_3d_scene_interactive.html` | 3D view: sheets, lamps, and the three projected walls |
| `pearl6_3d_sheets_only.html` | 3D view: the sheets alone, no walls -- easier to read the weave |
| `pearl6_projections_preview.png` | what the three shadows look like, full resolution |
| `pearl6_verification_report.json` | all three acceptance gates, with per-view metrics |
| `pearl6_metrics.json` | stage-1 solver metrics |

## Expected result

Measured by re-rendering **from these cut files**: mean IoU **0.8460**, worst view **0.8358**, SSIM 0.7506, edge fidelity 0.5702.

With six sheets there is no redundancy: single-panel ablation costs between 0.052 and 0.166 mean IoU. Every sheet is load-bearing, so a mis-cut sheet is not a cosmetic problem.

