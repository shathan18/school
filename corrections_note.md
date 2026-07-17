# Corrections to figures quoted in earlier drafts

**Purpose.** Three numbers in material we drafted for you were later found to be
measured the wrong way. This note supersedes them together — all three were quoted
in the same documents, so they need correcting as a set rather than piecemeal. A
fourth item is a new finding that reframes what those numbers were really about.

None of these change the core method or its validated result (signed-damage
assignment still works, and still generalises to bold / flat / posterised /
palette-compatible imagery). What changes is that three specific figures were
**optimistic because they measured the easy thing instead of the honest thing.**
Each correction below states the wrong number, the right number, and *why* the
first one was wrong.

---

## 1. Reconstruction completeness: **85% → 57% / 54%**

- **Quoted:** ~85% "image completeness."
- **Corrected:** **57%** (Wall A) / **54%** (Wall B).
- **Why it was wrong:** the 85% was a **coverage** metric — it asked "is this part
  of the subject darkened at all?" and scored a pixel as complete whenever *any*
  shadow fell there, regardless of colour. A shard casting the wrong colour still
  counted as 100% complete. Replacing it with a **colour-match** metric
  (‖predicted − source‖ < 0.30 in RGB, and required to beat a blank wall) drops it
  to ~54–57%. That is the honest fraction of the image that is reconstructed *in
  something close to the right colour*, not merely touched by shadow.

## 2. Face reconstruction: "saturates / colour-limited" — **retracted**

- **Quoted:** portrait faces "saturate around 0.53 SSIM" and are **colour-limited**
  (implying more colours/channels were the missing ingredient).
- **Corrected:** **retracted.** Faces are **resolution-limited**, not colour-limited.
- **Why it was wrong:** we had only tested up to the ~300-shard budget. Pushing shard
  count far higher shows the face keeps improving and only resolves recognisable
  features at **~2750 shards — roughly 9× our 300-shard budget.** So the limit is the
  number of pieces we can physically cut and laminate, not the palette. Within a
  buildable budget, photographic faces stay blurry; that blur is a **fabrication-scale
  limit**, and it is why bold / flat / posterised sources are the right choice for a
  300-shard build.

## 3. Cross-wall "joint-intersection": **27–31% → ~3%**

- **Quoted:** ~27–31% "joint-intersection" — presented as the sculpture doing genuine
  double duty, one wall's material also helping the other wall's image.
- **Corrected:** **~3%** genuine agreement (for an arbitrary image pair).
- **Why it was wrong:** the 27–31% counted every stray shadow that merely **landed on
  the other image's subject**, regardless of colour. When we require the stray shadow
  to also carry the **colour that spot actually wants**, about **90% of it disappears**
  — it was the wrong colour, i.e. visual contamination that happened to overlap
  content. The classic example: a red shard landing on a golden croissant was scored
  as "helpful" for being darker than white, while actually reading as a red stain.
  Genuine colour-agreeing double duty on an arbitrary pair is ~3%.

---

## 4. New finding — what those numbers were really measuring

Correcting #3 led to the more interesting result, which we'd recommend foregrounding
in any future write-up:

**Genuine cross-wall double duty is bounded by how colour-compatible the two source
images are — not by the algorithm.**

- **Arbitrary pair** (e.g. apples / breakfast): honest double duty ≈ **0.3–3%**. The two
  images simply don't want the same colours at the geometrically-linked points, so no
  amount of solver cleverness can manufacture agreement that isn't there.
- **Deliberately compatible pair:** the strongest we tested is *Girl with a Pearl Earring*
  **front vs back** — the same figure in the same palette. Honest double duty rises to
  **~24% (mean, Wall A; up to ~28% per seed)**. A same-scene / opposite-light pair (Monet
  *San Giorgio Maggiore* morning vs dusk) reaches **~16%** by the same measure. Both have
  measured headroom toward ~25–40%.
- **The move costs no algorithm change.** Swapping the arbitrary pair for the compatible
  one — same solver, same settings — lifts the honest figure from **0.3% → ~24%**. Pair
  selection, not solver tuning, is the dominant lever.

**Design consequence:** choose the two images to be palette-compatible *up front* — ideally
the same subject or scene. That single choice buys more real double duty than any change we
have made to the optimiser.

*Full-quality reconstruction of the front/back Girl pair, with the corrected metrics and a
side-by-side source-vs-shadow comparison, is in the accompanying interactive deliverable.
(One honest caveat carried from correction #2: the face reads as a soft blur at a buildable
~200-shard budget — a fabrication-scale limit, not a colour one.)*

---

### One-line summary of the correction set

| # | Figure | Was | Is | Root cause |
|---|--------|-----|-----|-----------|
| 1 | Completeness | 85% | 57% / 54% | coverage measured, not colour |
| 2 | Faces | "colour-limited, saturates" | resolution-limited (~2750 shards ≈ 9× budget) | tested only to 300 shards |
| 3 | Double duty | 27–31% | ~3% (arbitrary pair) | counted "landed on content", not "right colour" |
| 4 | *(new)* Double duty is set by **image-pair compatibility** | — | 0.3% arbitrary → ~24% compatible (Girl front/back), no algorithm change | — |
