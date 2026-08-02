# Development Log — Laser-Engraved Grayscale Shadow-Art

A chronological record of the project **from the point we switched to the new laser-engraving method**
(replacing the earlier CMYK colour-layering approach). Aug 2026.

**Interactive write-up:** https://claude.ai/code/artifact/90d30fa9-7b86-4ab1-86fe-17d979a5972a

---

## Stage 1 — New fabrication basis: laser-engraved grayscale

Dropped CMYK (fragile, colour registration never held). New basis: each shard is **clear perspex whose
tone is set directly by the laser** — engraving the surface to a controlled frost density (more laser =
more opaque = darker). No inks, no colour registration, no layer limit. *(Dye was considered but is not
part of the confirmed process.)*

- Palette quantised to **four levels**: clear · light · medium · black.
- Optical density **adds** where a ray passes through stacked shards.
- Panels: **60×60 cm** clear sheets; shards **≥ 5 mm** (below that the light grey won't register).

## Stage 2 — Grayscale beat binary; egg-crate made it fabricable

- Representing mid-tones as **grey** instead of scattered black dots was the real stipple-killer:
  **IoU 0.64 → 0.79**.
- Fabricability fix: two **parallel families** of sheets cross to form an **egg-crate** so every joint
  is exactly **two** sheets (no three-way intersections). Verified 2-way on every build.

## Stage 3 — Panel-layout optimisation

- Swept panel **count** and **pitch**: **6 panels at 6 cm pitch** is the robust optimum (fewer loses
  fidelity, wider loses multiplexing).
- **30×30 vs 60×60 (fair test, both covering the full figure):** ties on silhouette
  (**IoU 0.81 vs 0.80**), but 30×30 uses far fewer / coarser shards (259 vs 2278) and loses fine detail.
  → **Kept 60×60** for portraits.

## Stage 4 — Turntable concept + stop-angle optimisation

Replaced the 3-wall / 3-lamp rig with **one fixed light, one fixed wall, and a rotating base** —
optically identical, far simpler hardware. Three stops → three pictures.

- 2-stage sweep (layout × stop angles). **Even-120° spacing, grid phased 15°** wins → **stops at
  15° / 135° / 255°**. (Only the full-quality solve reveals it — the fast proxy misranks.)
- Genuine multiplexing, not independent casters: **76% of shards serve ≥2 views, 44% all three**.

## Stage 5 — Faces (prototype)

Three expressions of one man — **laughing / stern / afraid** — on a single assembly. Its job was to
prove grayscale multiplexing carries fine photographic detail and keeps expressions distinct. It does:
**IoU 0.72, distinctness 0.97, 1041 shards, 6 panels.** → `out_final/faces2_60x60/`

## Stage 6 — Girl with a Pearl Earring (grand finale) + the pearl

Back / side / front of Vermeer's girl from a single shard cluster: **IoU 0.83, 76%/44% multiplexing,
669 shards.** → `out_final/pearl_girl3_turntable/`

- The iconic **pearl earring** is exactly what a shadow reconstruction loses first. Isolated it in each
  view with **SAM** (Segment Anything, `sam-vit-huge`, run offline) and emphasised it (natural-size
  bright disc + thin dark ring) so it reads as a distinct feature — at no cost to overall fidelity.
- **Source quality dominates:** highest-resolution cut-outs beat a lower-res matched triptych
  (0.83 vs 0.77) on the same config.

## Stage 7 — Noise handling (what didn't work)

- **Heavy de-streak — rejected.** Morphological opening cleaned background "rain" but ate real detail on
  tonal portraits (**IoU 0.83 → 0.55**). Faint streaking is left as acceptable texture instead.

## Stage 8 — Same method, more subjects

- **CS × Technion** faculty logos (2 views, 4 panels): grayscale rescued the CS mark that blobbed out
  under binary. IoU 0.68. → `out_final/cs_technion_60x60/`
- **Pearl Girl front + back** (2-view), the build that preceded the three-view finale. IoU 0.76.
  → `out_final/pearl_girl_60x60/`

## Stage 9 — Fabrication deliverables + write-up

Every build folder (`out_final/<build>/`) ships:

- `*_shards.obj` + `.mtl` — each shard a separate object, grouped by tone (light/medium/black),
  distinct materials; opens cleanly in Rhino.
- `cut_sheets/<panel>.svg` and `.dxf` — one file per 60×60 panel, layered by tone.
- `ASSEMBLY_MAP.png` — plan view: which sheet, what azimuth, what offset, which edge faces out
  (cross-checked against the OBJ/SVG, 5/5 geometry checks).
- `preview_walls.png` + `README.txt`.

Plus the single interactive page (linked above) with the 360° orbit viewer.

---

### Results at a glance

| Build | Views | IoU | Distinct. | ≥2-view shards | Shards |
|---|---|---|---|---|---|
| Faces (prototype) | laughing / stern / afraid | 0.72 | 0.97 | — | 1041 |
| **Pearl Girl (finale)** | back / side / front | **0.83** | 0.90 | 76% | 669 |
| Pearl Girl (2-view) | front / back | 0.76 | — | — | 684 |
| CS × Technion | 2 faculty logos | 0.68 | 0.71 | — | ~775 |

*All builds: 6 panels · 60×60 cm · 2-way egg-crate · clear + 2 greys + black · 5 mm min shard ·
single light + turntable (3 stops).*

---

## Contribution — what's new vs. classic 3-wall shadow art

Classic multi-view shadow art (e.g. Mitra & Pauly, *Shadow Art*, SIGGRAPH Asia 2009) carves a **single
connected 3D object** that throws **binary** shadows from **three fixed lights onto three fixed walls**.
Layered-attenuator work (cf. Baran et al. 2012, layered attenuators for prescribed shadows; ShadowPix)
stacks semi-transparent sheets for grayscale multi-shadow. Our build differs on several axes:

1. **One wall, one light, a turntable — not 3 walls + 3 lamps.** We show this is optically equivalent to
   the N-wall rig (rotating the assembly by θ under a fixed light ≡ presenting a wall at azimuth θ),
   collapsing the whole installation to a single lamp / wall / motor — simpler to build, calibrate, and
   exhibit.
2. **Stop angles and grid phase become free optimisation variables** — a degree of freedom the
   fixed-wall rig doesn't have. We sweep them (even-120°, phase 15° wins) and show it changes the result.
3. **Flat, laser-shaded shards in a 2-way egg-crate — not a solid carved object.** Everything is flat
   60×60 cm sheets that slot together (no 3-way joints), laser-cuttable and hand-assemblable — a
   fabrication model distinct from voxel carving.
4. **Grayscale by laser-engraved attenuation + optical stacking**, enabling continuous-tone
   **photographic portraits** (Vermeer, expressive faces) rather than bold binary silhouettes.
5. **Genuine multiplexing, quantified.** We separate *real* multiplexing (shards contributing to several
   views) from decoupled independent casters via a per-shard duty metric (% serving ≥2 views), and
   optimise for it — 76% / 44% on the finale. Prior work reports shadow fidelity but not sharing.
6. **Joint tomographic inverse solve** over a shared per-panel opacity field (vs. greedy per-shard
   heuristics), plus a **feature-preservation pass** (SAM-isolated salient detail — the pearl).

*One-line framing:* **grayscale, laser-fabricated, genuinely-multiplexed shadow art on a single-light
turntable — with the rotation geometry itself optimised.**

## Roadmap to publication level — what's left

The engineering is largely in place; the missing pieces are validation, comparison, and analysis.

**Physical validation (highest priority)**
- Laser-cut, engrave, assemble, light, and **photograph the real casts**; compare photos to the honest
  renders (registration + tone calibration). Renders are faithful, but a paper needs real-world evidence.
- **Tone calibration:** measure the actual laser-frost optical densities and verify the 4-level palette
  and the *additive-stacking* assumption on physical samples.
- **Tolerance study:** sensitivity to cut precision, panel-placement error, laser-tone repeatability,
  and light-source size (penumbra).

**Comparisons & ablations**
- Re-implement / compare against **classic 3-wall shadow art** and **layered attenuators** on identical
  targets; quantify our advantage (hardware simplicity, grayscale detail, multiplexing).
- Formalise our sweeps as **ablation tables** (grayscale vs binary · panel count/pitch · stop-angle &
  phase · source resolution · 30 vs 60 cm) with one consistent metric set.

**Analysis / theory**
- State the forward model precisely and prove the **turntable ≡ N-wall equivalence**.
- Characterise the **multiplexing bound** from non-negativity / attenuation coupling: how much
  conflicting detail three images can share, and how it scales as #views grows (4 / 5 / 6 stops?).
- Degrees-of-freedom vs. constraints counting.

**Evaluation**
- A small **perceptual study**: do viewers recognise each view and read them as distinct?
- Report SSIM / IoU **plus a perceptual metric**, against the baselines above.

**Write-up & release**
- Related-work positioning; honest **limitations** (resolution ceiling, residual streaking, source
  dependence); and **code + configs released** for reproducibility.
