# Choosing image pairs that actually reconstruct

**The single most important rule: use a CENTERED SUBJECT on a plain background, not a
full-frame image.** ShadowArt casts a shadow of a subject; it cannot reproduce a
background, and a full-frame painting spends the whole shard budget trying to — so it
collapses into a few flat colour blobs (we measured this: full-frame Monet / Cézanne
paintings reconstruct as unrecognizable blobs, RMSE 0.10–0.30 but visually meaningless).

The moment the subject is isolated on a plain ground, the same pipeline produces a clearly
**recognizable** wall image. Two independent verified examples in this repo:

| Pair | What it is | Composite | edge-fidelity A / B | Reads as… |
|------|-----------|-----------|--------------------|-----------|
| **Girl front/back** (`girl_front_nobg.png` / `girl_back_nobg.png`) | one figure, two views, background pre-removed | **2.26** | 0.58 / 0.81 | clearly the portrait + its profile |
| **Greek amphorae** (`amphora_taleides.jpg` / `amphora_andokides.jpg`) | two black-figure vases, dark subject on light ground | **1.93** | 0.69 / 0.26 | clearly two vases, terracotta/black split |

Compare to the full-frame paintings that do **not** work: Monet Rouen 1.95* / Cézanne
1.73* / Vermeer full-frame 1.61 — the composite scores look similar but the images are
blobs, because the score rewards average colour and the background dominates the frame.
(*RMSE looked great precisely because a flat wash of the right average colour scores well —
exactly the failure `metrics.py` warns about.)

This matches `corrections_note.md`: the method is for **bold / flat / posterized** subjects
with a clear silhouette, and reconstruction is fabrication-scale-limited, so simpler subjects
win at a buildable shard budget.

---

## Why a compatible PAIR still matters

Within the centered-subject regime, pick the two subjects to be **palette-compatible** —
ideally the same subject/scene or the same colour family. Per `corrections_note.md` §4 this
is what buys genuine cross-wall "double duty" (~24% for a same-subject pair vs ~0.3% for an
arbitrary one) with no solver change. The two working pairs above are both compatible by
construction: the Girl pair is *literally the same figure* twice; the amphorae share the
black-figure terracotta palette.

---

## The workflow

### 1. Isolate the subject

If your images already have the subject cut out on white (like the `*_nobg.png` files),
skip this. Otherwise:

```bash
py -m shadowart.cli segment --in my_a.jpg my_b.jpg --out-dir examples
#   writes examples/my_a_nobg.png / my_b_nobg.png
#   --bg-tol / --sat-tol widen what counts as background (raise for a darker/greyer ground)
```

or fold it into the run itself:

```bash
py -m shadowart.cli run ... --remove-bg ...   # writes <out>/nobg_a.png, nobg_b.png first
```

**Limits of the built-in remover** (classical, no ML — numpy/scipy only): it flood-fills the
background inward from the image border, so it is reliable only for a **reasonably uniform,
plain** background (white / light grey / a flat colour). It will misfire on:
- a **gradient** studio background (e.g. Met object photos) — it strips only the lightest
  edge and leaves wisps;
- a **dark** background behind a **light** subject — it can invert (remove the subject).

It guards against the worst case: if the detected subject is <10% or >92% of the frame the
result is flagged **untrustworthy** and the original is kept unchanged. For those images,
make a manual cut-out (any editor → subject on a white canvas) or, for a dark high-contrast
subject, just run it as-is — a strong dark silhouette reconstructs well even with a little
background left in (see the amphora result).

### 2. Run with a restart budget

```bash
# Flagship: same figure, two views (background already removed)
py -m shadowart.cli run --scene scenes/example.yaml --color --color-mode overlap \
    --target-a examples/girl_front_nobg.png --target-b examples/girl_back_nobg.png \
    --restarts 8 --out out_girl

# Two palette-matched objects (dark subject; no removal needed)
py -m shadowart.cli run --scene scenes/example.yaml --color --color-mode overlap \
    --target-a examples/amphora_taleides.jpg --target-b examples/amphora_andokides.jpg \
    --restarts 8 --out out_amphora
```

`--restarts N` keeps the best of N shard layouts by the composite score
(`solve/search.py::score_layout` = SSIM + edge-fidelity − cross-talk). Add `--palette noir`
for a tonal black-and-white read, `--search-panels --panel-restarts 4` to also search panel
geometry, or `--time-budget 60` to cap wallclock instead of a fixed count.

---

## Sourcing your own pairs (what to look for)

- **Centered single subject**, silhouette clearly separable from the background.
- **Plain background** (white/light) if you want the auto-remover to work; otherwise plan a
  manual cut-out.
- **Two subjects that share a palette** — same object/scene, or the same limited colour set.
- **Bold, flat, high-contrast** beats subtle/photographic (faces stay soft at a buildable
  budget — that's a fabrication limit, not a colour one).

Good public-domain sources of isolated subjects:
- [The Met Open Access](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access)
  — object photos (vases, sculpture, tools) on plain grounds; API gives direct image URLs.
- [Wikimedia Commons — objects on white background](https://commons.wikimedia.org/wiki/Category:Objects_on_white_background)
- [PhyloPic](https://www.phylopic.org/) — public-domain silhouettes (animals, figures),
  already black-on-transparent, ideal for the mono path.

The two amphorae here came from the Met:
[Taleides amphora](https://www.metmuseum.org/art/collection/search/254578) ·
[Andokides amphora](https://www.metmuseum.org/art/collection/search/255154).
The Girl front/back pair is a pre-cut derivative of Vermeer's
[Girl with a Pearl Earring](https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring).
