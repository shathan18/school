# AI-generated image pairs engineered for ShadowArt

Instead of forcing existing art through the pipeline, we design two images *to* the medium's
measured constraints, then generate them. This doc is the prompt package: what the data requires,
three concepts, ready-to-paste prompt pairs, and how to verify.

## What our data says an ideal pair requires (each point is defensible)

1. **Flat, few-colour, bold — no photographic detail.** At the ~300-shard fabrication budget each
   shard is one flat colour; faces / fine tonal detail need **~2750 shards (≈9× budget)** and fail
   (`corrections_note.md` §2). The method is for "bold / flat / posterized subjects with a clear
   silhouette" (`docs/image_pairs.md`).
2. **Centred subject on a PLAIN WHITE ground — not full-frame.** ShadowArt casts a shadow of a
   *subject*; it cannot reproduce a background, and a full-frame image spends the whole budget on
   the background and collapses to blobs (`docs/image_pairs.md` rule #1). Verified winners are
   centred subjects on light grounds (Girl front/back composite **2.26**, amphorae **1.93**).
3. **Colour compatibility drives double duty.** A shard carries one colour from its own wall, so its
   stray shadow only *helps* the other wall where both images want a compatible colour at
   geometrically-linked points (`report_lecturer.md`). Measured: genuine double duty is
   **~0.3–3% for an arbitrary pair** vs **~24% for a same-subject / same-palette pair** — *with no
   algorithm change* (`corrections_note.md` §4). The reliable lever is a **shared, restricted
   palette used in similar proportions** (so wherever a shard lands it likely meets an agreeing
   colour). Parallel/mirrored composition is a bonus, but exact positional correspondence is set by
   the projection geometry, not the artwork — don't overclaim it.
4. **Palette must map to the fabricable CMYK perspex** (`shadowart/targets/color.py` `PERSPEX`):
   single sheets = **Cyan / Magenta / Yellow / Black**; two stacked sheets give the secondaries
   **red-orange (M+Y), blue-violet (C+M), green (C+Y)**; **white = no shard**. Design to ≈3–5 of:
   white, black, cyan, magenta, yellow, red-orange, blue, green.
5. **Recognizable when coarse** → strong silhouette, iconic simple form, thick outlines.

**→ Prompt rules:** flat screenprint/woodblock; centred single subject on white; one shared palette
of 3–4 named CMYK-reachable colours; big contiguous fields; bold thick outlines; explicitly forbid
gradients / shading / photorealism / background / extra colours.

## Concepts

| # | Concept | Palette feel | Why |
|---|---------|--------------|-----|
| **A** | **Two Koi** (matched pair / yin–yang) — *recommended* | warm orange/red/black | colourful **and** high agreement (both same warm palette), iconic silhouette |
| B | Sun & Moon (day/night) | shared warm (gold moon) | clean two-wall narrative; centred discs |
| C | Two Faces in Profile (dialogue) | near-mono (black + 1 accent) | **max** agreement (huge shared black field), cheapest cut; corner = facing |

**Recommend A (Two Koi):** best balance of colour with high colour-agreement, a genuine
"belongs-together" story for two walls, unmistakable at 300 shards. Use **C** to *max out* the
double-duty number if near-monochrome is acceptable.

---

## Pair A — TWO KOI (recommended)

**Shared palette (identical both walls):** white *(no shard)* · black **K** (CMYK 0/0/0/100) ·
vermilion-orange **M+Y** (≈0/60/100/0) · deep red (≈0/90/85/0).

**Wall A prompt**
```
Flat two-colour screenprint / Japanese ukiyo-e woodblock of a single stylized koi carp, curving
upward and to the right, centred on a plain solid white background. Bold solid shapes only: body
filled flat vermilion-orange, a few thick black outline strokes for scales and fins, one deep-red
patch on the head, crisp black contour. No water, no scenery, no background — just the fish on
white. Poster art: hard edges, solid flat colour fills, high contrast, thick confident outlines,
minimal internal detail. Limited palette only: white, black, vermilion-orange, deep red. Reads
clearly as a koi at a glance.
```

**Wall B prompt**
```
Flat two-colour screenprint / Japanese ukiyo-e woodblock of a single stylized koi carp, curving
downward and to the left — the mirror partner of the other fish — centred on a plain solid white
background. Identical style and palette: body flat vermilion-orange, thick black outline strokes
for a few scales and fins, one deep-red patch on the head, crisp black contour. No water, no
background — just the fish on white. Poster art: hard edges, solid flat colour fills, high
contrast, thick outlines, minimal detail. Same limited palette: white, black, vermilion-orange,
deep red. Together the two fish read as a matched pair circling each other.
```

**Rationale (→ constraint):**
- Flat fills / hard edges / 4 colours → **§1** (budget needs flat regions & few colours).
- *Identical* palette + mirrored layout → **§3** (agreement is the measured lever, ~24% vs ~3%).
- Single koi, thick outline, on white → **§2 + §5** (budget serves subject; strong coarse silhouette).

---

## Pair B — SUN & MOON (day/night)

**Shared palette (warm, so they agree):** white *(no shard)* · black **K** (0/0/0/100) ·
golden-yellow **Y** (0/0/100/0) · warm orange **M+Y** (≈0/55/100/0).

**Wall A prompt (Sun)**
```
Flat minimalist poster of a stylized SUN: one bold solid golden-yellow disc centred on plain solid
white, ringed by simple flat orange triangular rays, thick black outline. Screenprint/woodblock
style: hard edges, solid flat colour, high contrast. No gradient, no glow, no shading, no
background. Limited palette only: white, black, golden-yellow, warm orange. Symmetric, iconic,
reads instantly as a sun.
```

**Wall B prompt (Moon)**
```
Flat minimalist poster of a stylized crescent MOON with a few small stars, centred on plain solid
white, in the SAME warm palette as its partner sun: the moon a bold solid golden-yellow/orange
crescent, stars as simple flat orange four-point shapes, thick black outline. Screenprint/woodblock
style: hard edges, solid flat colour, high contrast. No gradient, no glow, no shading, no
background. Same limited palette: white, black, golden-yellow, warm orange. Iconic, reads instantly
as night.
```

**Rationale:** warm disc centred in both, same 4 colours → **§3** (gold moon, not blue, so palettes
match); flat poster, no glow/gradient → **§1**; centred disc on white, bold → **§2/§5**; day-vs-night
across perpendicular walls = viewer stands between them (theme fit).

---

## Pair C — TWO FACES IN PROFILE (max double-duty / near-mono)

**Shared palette:** white *(no shard)* · black **K** (0/0/0/100) · terracotta-red **M+Y** (≈0/65/90/0).

**Wall A prompt**
```
Flat high-contrast screenprint of a single human head in PROFILE facing right, as one bold solid
black silhouette centred on plain solid white, with one flat terracotta-red shape for the
collar/shoulder. Stencil/woodblock style: hard edges, solid flat fills. No facial detail beyond the
silhouette outline (nose/lips/brow are the contour only), no gradient, no shading, no background.
Limited palette only: white, black, terracotta red. Bold, iconic, reads instantly as a face in
profile.
```

**Wall B prompt** *(mirror)*
```
Flat high-contrast screenprint of a single human head in PROFILE facing left — the mirror partner —
one bold solid black silhouette centred on plain solid white, one flat terracotta-red collar shape.
Identical style/palette. No facial detail beyond the outline, no gradient, no shading, no
background. Same limited palette: white, black, terracotta red. The two profiles face each other
across the corner.
```

**Rationale:** two big black silhouettes → largest shared single-colour field → **§3** max agreement
(mostly K, easiest single-sheet colour); silhouette-only, no face detail → **§1** (faces are the
~2750-shard failure mode — cut out on purpose); mirrored profiles across the corner = the concept
*is* the geometry.

---

## Tell the generator to AVOID (the measured failure modes)

Negative prompt / "do not include": *photorealistic, 3D render, realistic face or skin, fine facial
features, gradients, soft shading, glow/bloom, drop shadows, ambient occlusion, painterly brush
texture, fur/scale/skin texture, blur, bokeh, depth of field, colour noise/grain, background scenery
/ landscape / pattern / frame, more than 4 colours, watermark.*
Positive anchors to repeat: *flat vector, screenprint, woodblock, poster, bold, hard edges, solid
colour fills, limited palette, high contrast, thick outlines, centred subject on plain white.*
(no-gradient/no-detail → §1; on-white/no-background → §2; ≤4 shared colours → §3; bold silhouette → §5.)

Generator notes: prompts are portable. Midjourney — add `--style raw --no gradient,shading,
background,text`. DALL·E / GPT-image — the sentences above work as-is; ask for "vector-flat, solid
colours". SDXL — pair with a low CFG and a "flat vector illustration" LoRA if available.

## Verify (defensible test)

1. **Prep:** save each as PNG. If the background isn't white, put the subject back on white:
   `py -m shadowart.cli segment --in a.png b.png --out-dir examples`. If gradients slipped in,
   posterize to K=4–5 first (the validated flatten step).
2. **Compatibility headline:** add the pair to `PAIRS` in `out_thickness_test/ceiling_straddle.py`
   and run it. Target the honest colour-agreeing **B_good in the ~15–25% band**, clearly above the
   ~3% arbitrary-pair floor, with a **high straddle %** (shape isn't the cap — flat regions worked).
3. **It reads:** `py -m shadowart.cli run --scene scenes/example.yaml --color --color-mode overlap
   --target-a examples/<A>.png --target-b examples/<B>.png --restarts 8 --out out_pair`; check
   composite ~1.9–2.3 and edge-fidelity ~0.6–0.8 (near the verified winners), and that
   `out_pair/preview_final.png` reads at 300 shards.
4. **Ablation for the report:** also run an *arbitrary* pair (e.g. one koi + the sun) through
   ceiling_straddle to show the designed pair beats it on B_good — proving the gain came from the
   shared-palette design, not the algorithm.

---

# V2 — more complex COMPOSITION  (superseded: went too far on visual complexity; see V3)

The v1 pairs generated too simply / too identically. Three fixes, all still inside the medium's
limits:

- **Never say "mirror."** The two prompts must describe **genuinely different subjects/scenes**;
  they share only the *style + palette clause* (paste it verbatim into both). To lock the palette
  while the content differs, generate wall B with wall A as a **style reference**
  (Midjourney `--sref <A>`, or img2img / "in the exact same flat style and palette as this image").
- **"More complex" = more bold flat SHAPES, not more detail.** Compose a small SCENE of 3–6 large
  flat elements (figure + a few symbolic props). Keep every element a big solid shape with a thick
  outline; still no shading, no texture, no fine faces (those are the ~2750-shard failure mode).
- **Deeper theme = a two-act myth.** The two walls are one story: rise/fall, before/after,
  this-becomes-that. Keep **similar colour proportions** across both (e.g. ~50% dark, ~30% warm,
  ~20% white) so double duty (§3) survives the differing shapes.

## Pair D — ICARUS: Ascent & Fall  (recommended v2)

Myth of ambition and hubris — one figure, two acts. Palette: white *(no shard)* · black **K**
(0/0/0/100) · golden-yellow **Y** (0/0/100/0) · vermilion-orange **M+Y** (≈0/60/100/0).

**Wall A — The Ascent**
```
Flat woodblock / screenprint scene, a centred compound subject on a plain solid white background
with wide white margins: a bold BLACK-silhouette winged figure climbing UPWARD, large feathered
wings spread wide (feathers as bold flat black shapes), reaching toward a big solid golden-yellow
SUN disc in the upper corner ringed by a few flat vermilion-orange triangular rays. Only four flat
colours: white, black, golden-yellow, vermilion-orange. Hard edges, solid flat colour fills, thick
outlines, high contrast. No shading, no gradient, no background scenery, no fine facial detail — the
figure is one solid silhouette. Bold, mythic, reads clearly at a glance.
```
**Wall B — The Fall**
```
Companion piece in the IDENTICAL flat style and four-colour palette (white, black, golden-yellow,
vermilion-orange): a flat woodblock / screenprint scene, a centred compound subject on plain solid
white with wide white margins, showing the SAME black-silhouette winged figure now TUMBLING
DOWNWARD head-first, its wings breaking apart into scattered bold flat black feather shapes; a small
golden-yellow sun far above, and stylized vermilion-orange-and-black WAVES as a few bold curved
shapes across the bottom (discrete shapes, not a full-bleed sea). Hard edges, solid flat fills,
thick outlines, high contrast. No shading, no gradient, no fine detail. Ascent and fall of one
figure.
```
Rationale: 4 flat colours + compound-but-bold shapes → §1; identical palette & similar dark/warm
proportions → §3; centred compound subject with white margins → §2; solid-silhouette figure → §5;
ascent-vs-fall = two DISTINCT scenes of one myth (fixes the "identical" problem) → theme.

## Pair E — KOI & DRAGON: the Dragon Gate  (transformation)

Chinese legend: the carp that leaps the falls becomes a dragon — perseverance & becoming. Palette:
white · black **K** · vermilion **M+Y** (≈0/60/100/0) · deep blue **C+M** (≈95/70/0/10).

**Wall A — The Leap**
```
Flat ukiyo-e woodblock, centred subject on plain solid white with white margins: a bold VERMILION
koi carp leaping UPWARD out of a few bold flat deep-blue waterfall / wave shapes, thick black
outline and a few black scale strokes, one small blue splash. Only four flat colours: white, black,
vermilion, deep blue. Hard edges, solid flat colour fills, high contrast, thick outlines. No
gradient, no shading, no background. Bold and dynamic.
```
**Wall B — The Dragon**
```
Companion piece in the IDENTICAL style and four-colour palette (white, black, vermilion, deep blue):
a flat ukiyo-e woodblock of an Eastern DRAGON coiling upward (the transformed koi), a long bold
vermilion serpentine body with flat black scale strokes, deep-blue mane and whiskers as bold flat
ribbons, clutching a small blue orb, centred on plain solid white with white margins. Hard edges,
solid flat fills, high contrast, thick outlines. No gradient, no shading, no background. The koi and
the dragon read as the before-and-after of one transformation.
```
Rationale: two DISTINCT forms (fish → dragon), one legend → theme + fixes duplication; shared 4
colours & proportions → §3; bold flat serpentine/koi silhouettes → §1/§5; on white → §2.

## Pair F — THE SUN & THE MOON  (tarot, symbolic)

Consciousness/day vs mystery/night — a classic symbolic couple, folk-art tarot style (inherently
flat & bold). Palette: white · black **K** · golden-yellow **Y** · deep blue **C+M** (≈95/70/0/10).

**Wall A — The Sun**
```
Flat folk-art tarot-card illustration, centred symbolic composition on plain solid white with white
margins: a large radiant golden-yellow SUN with a simple bold face (eyes and mouth as plain black
shapes, NOT realistic), straight-and-wavy flat rays, and two bold flat golden sunflowers on black
stems below. Only four flat colours: white, black, golden-yellow, deep blue. Hard edges, solid flat
fills, thick black outlines, high contrast. No gradient, no shading, no background. Bold, symbolic.
```
**Wall B — The Moon**
```
Companion piece in the IDENTICAL style and four-colour palette (white, black, golden-yellow, deep
blue): a flat folk-art tarot-card illustration on plain solid white with white margins: a
golden-yellow crescent inside a full MOON disc with a simple bold face, two flat black tower shapes
either side, a bold black wolf silhouette howling below, and a winding deep-blue path. Hard edges,
solid flat fills, thick outlines, high contrast. No gradient, no shading, no background. Bold,
symbolic — the mysterious counterpart to the Sun.
```
Rationale: distinct symbolic scenes, same 4 colours → §3 + theme; compound flat shapes (not detail)
→ §1; simple flat faces avoid the ~2750-shard face trap → §1; centred on white → §2.

---

# V3 — SIMPLE image, DEEP meaning  (the target)

The correction to V2: keep the picture **dead simple** (one bold flat icon per wall — exactly what
300 shards wants), and put **all the depth in the pairing**. A lone koi or a lone sun is
decorative; two simple icons chosen so that **together they mean something** turn the corner
installation into a metaphor the viewer completes by standing between the walls.

Design rule: **one icon per wall, shared 3-colour palette, and a concept where the two icons form a
duality / cause / metaphor.** The meaning lives in the *relationship*, not the drawing.

## Pair G — THE MOTH & THE FLAME  (recommended)

**Meaning:** the fatal pull of desire toward what consumes us — longing, obsession, the sublime and
self-destruction. The two walls meet at a right angle; the empty corner *between* them is the
distance the moth is forever crossing toward the flame. The viewer standing there stands inside the
metaphor. (Simple images; the depth is the gap.)
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · golden-yellow **Y** (0/0/100/0) ·
warm orange **M+Y** (≈0/60/100/0).

**Wall A — The Flame**
```
Flat screenprint of a single lit CANDLE with a bold flame, centred on plain solid white with wide
white margins. The flame one solid golden-yellow shape with an orange core; the candle a simple flat
black column; a thin black wick. Only these flat colours: white, black, golden-yellow, orange. Hard
edges, solid flat fills, thick outlines, high contrast. No glow, no gradient, no shading, no
background. Simple and iconic — a single candle flame.
```
**Wall B — The Moth**
```
Companion piece in the IDENTICAL flat style and palette (white, black, golden-yellow, orange): a
single MOTH seen from above with wings spread, centred on plain solid white with wide white margins
— body and wing outlines solid black, wing markings a few flat orange and golden-yellow shapes. Hard
edges, solid flat fills, thick outlines, high contrast. No gradient, no shading, no background.
Simple and iconic — a single moth.
```
Meaning → theme; one bold icon each on white → §1/§2/§5; shared warm palette → §3; distinct subjects
(flame vs moth) → no duplication.

## Pair H — THE DOVE & THE SERPENT

**Meaning:** "Be wise as serpents and innocent as doves." Not good vs evil as enemies — the two
natures a whole person must hold together: cunning and gentleness, knowledge and innocence. They
face each other across the corner as halves of one self.
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · one warm accent — deep red **M** (≈0/90/70/0).

**Wall A — The Dove**
```
Flat woodblock / linocut of a single DOVE in flight, wings raised, carrying a small sprig, centred
on plain solid white with wide white margins. Bold solid black silhouette with a crisp outline and
one small deep-red accent (the sprig). Only 3 flat colours: white, black, deep red. Hard edges,
solid flat fills, thick outline, high contrast. No shading, no gradient, no background. Simple and
iconic.
```
**Wall B — The Serpent**
```
Companion piece in the IDENTICAL style and 3-colour palette (white, black, deep red): a single
coiled SERPENT, centred on plain solid white with wide white margins — a bold solid black serpentine
coil with a crisp outline and one small deep-red accent (the forked tongue). Hard edges, solid flat
fills, thick outline, high contrast. No shading, no gradient, no background. Simple and iconic.
```
Meaning (archetypal duality) → theme; two bold single silhouettes → §1/§5; shared black+accent → §3;
on white → §2.

## Pair I — MEMENTO MORI: THE ROSE & THE SKULL

**Meaning:** vanitas — beauty and life are fleeting; the skull is the mirror of the rose. Two of the
oldest symbols in art: together they say *remember you will die — so live.* The bloom on one wall,
its end on the other.
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · crimson red **M** (≈0/95/75/0).

**Wall A — The Rose**
```
Flat screenprint of a single ROSE in bloom on a short stem with two leaves, centred on plain solid
white with wide white margins. Petals solid crimson red; stem and leaves solid black; thick black
outline. Only 3 flat colours: white, black, crimson red. Hard edges, solid flat fills, high
contrast. No shading, no gradient, no background. Simple and iconic.
```
**Wall B — The Skull**
```
Companion piece in the IDENTICAL style and 3-colour palette (white, black, crimson red): a single
human SKULL, front view, centred on plain solid white with wide white margins — a bold flat white
skull shape whose eye sockets, nose and teeth are solid black shapes, with one small crimson accent.
Graphic sugar-skull / screenprint look. Hard edges, solid flat fills, high contrast. No shading, no
gradient, no realistic bone texture, no background. Simple and iconic.
```
Meaning (life/death vanitas) → theme; skull drawn as flat black-on-white shapes (NOT a realistic
face) → §1 avoids the ~2750-shard trap; shared 3 colours → §3; on white → §2.

**Recommended: G (Moth & Flame)** — simplest images, warmest shared palette (best double-duty), and
the corner geometry literally *is* the metaphor.

## Pair J — THE ANCHOR & THE BIRD  (roots & wings)

**Meaning:** the two gifts we need — something to hold us steady and something to set us free;
weight and lift, staying and leaving. Across the corner: the anchor that keeps you and the wing that
releases you.
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · deep blue **C+M** (≈95/70/0/10).

**Wall A — The Anchor**
```
Flat screenprint of a single ship's ANCHOR with a coiled rope, centred on plain solid white with
wide white margins. The anchor one bold solid black shape; the rope a few flat deep-blue curved
shapes; thick black outline. Only 3 flat colours: white, black, deep blue. Hard edges, solid flat
fills, high contrast. No shading, no gradient, no background. Simple and iconic.
```
**Wall B — The Bird**
```
Companion piece in the IDENTICAL style and 3-colour palette (white, black, deep blue): a single BIRD
in upward flight, wings raised, centred on plain solid white with wide white margins — a bold solid
black silhouette with one small deep-blue accent. Hard edges, solid flat fills, thick outline, high
contrast. No shading, no gradient, no background. Simple and iconic.
```
Meaning (stability vs freedom) → theme; two bold single silhouettes → §1/§5; shared black+blue → §3;
on white → §2.

## Pair K — THE KEY & THE CAGE  (freedom within reach)

**Meaning:** freedom is often already in our hands — we hold our own key. The cage door stands
OPEN; whether that means escape or the fear of leaving it is for the viewer, standing between the
two, to decide.
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · golden-yellow **Y** (0/0/100/0).

**Wall A — The Key**
```
Flat screenprint of a single old ornate KEY, centred on plain solid white with wide white margins —
the key one bold solid golden-yellow shape with a thick black outline and simple black cut-outs in
the bow. Only 3 flat colours: white, black, golden-yellow. Hard edges, solid flat fills, high
contrast. No shading, no gradient, no background. Simple and iconic.
```
**Wall B — The Cage**
```
Companion piece in the IDENTICAL style and 3-colour palette (white, black, golden-yellow): a single
empty BIRDCAGE with its little door hanging OPEN, centred on plain solid white with wide white
margins — bold flat black bars and dome, one golden-yellow accent (the open latch / a hanging ring).
Hard edges, solid flat fills, high contrast. No shading, no gradient, no background. Simple and
iconic.
```
Meaning (captivity vs the open door) → theme; shared black+gold → §3; bold simple icons on white →
§1/§2/§5.

## Pair L — THE SOWER & THE SCYTHE  (reap what you sow)

**Meaning:** sowing and reaping — beginning and end, hope and harvest, life given and taken. The
scythe is both the farmer's tool and Death's; you reap what you sow. The cycle turns across the
corner.
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · golden-yellow **Y** (0/0/100/0).

**Wall A — The Sower**
```
Flat woodblock of a single SOWER — a bold BLACK-silhouette figure mid-stride scattering seed, the
seed a scatter of small flat golden-yellow shapes, centred on plain solid white with wide white
margins. Only 3 flat colours: white, black, golden-yellow. Hard edges, solid flat fills, thick
outline, high contrast. No facial detail (the figure is one solid silhouette), no shading, no
gradient, no background. Simple and iconic.
```
**Wall B — The Scythe**
```
Companion piece in the IDENTICAL style and 3-colour palette (white, black, golden-yellow): a single
SCYTHE crossed with a bound sheaf of golden-yellow WHEAT, centred on plain solid white with wide
white margins — blade and handle bold solid black, the wheat flat golden-yellow with black stalks,
thick outlines. Hard edges, solid flat fills, high contrast. No shading, no gradient, no background.
Simple and iconic.
```
Meaning (the cycle: life given & taken) → theme; shared black+gold in matched proportions → §3; bold
flat icons, figure as silhouette (no face) → §1/§2/§5.

---

# V4 — ICONIC & SECULAR  (each image famous-painting-strong; NO religion, NO mythology)

Reset: each image should hit like a *famous painting* on its own — a singular, culturally-loaded
icon — and the link is **conceptual** (two treatments of one idea, thing & echo, before/after,
question & answer), not a narrative of two characters. **Banned: any religion, any mythology (Greek,
Norse, all); copyrighted film/TV characters.** Sources ranged: famous secular art, iconic
photographs, landmark architecture, scientific/natural icons, history, graphic design, non-Western
culture.

## 10 concept pairs, ranked by iconic power

1. **The Great Wave × The Mushroom Cloud** — the natural sublime × the man-made sublime; two forces
   that dwarf us, one curling like the other. *[Hokusai × 20th-c photo]* Flatten risk: LOW.
2. **Van Gogh's "Starry Night" × A Spiral Galaxy** — the vision × the reality of a turbulent cosmos;
   the swirl he painted, the swirl that exists. *[Famous painting × science]* Risk: MEDIUM (Starry
   Night's power is partly brushwork; keep the big swirls + star discs).
3. **Tank Man × The Flower in the Rifle** — two icons of one idea: the individual's peaceful defiance
   of overwhelming force; force vs the refusal of force. *[Iconic photographs]* Risk: LOW.
4. **The DNA Double Helix × The Fingerprint** — the code every human shares × the pattern no two
   share; the universal blueprint and the singular self. *[Scientific icons]* Risk: LOWEST (already
   flat line-graphics — the safe breathtaking bet).
5. **The Blue Marble (whole Earth) × The Lunar Bootprint** — the vast × the intimate; the planet and
   the single step that let us look back at it. *[Iconic photographs / achievement]* Risk: MEDIUM
   (bootprint reads abstract; make the tread bold).
6. **The Sydney Opera House × The Nautilus Shell** — architecture × the natural form it echoes; design
   drawn from nature's blueprint. *[Landmark architecture × natural icon]* Risk: LOW.
7. **The Paper Crane × The Warplane** — fragile hope × the machine of war (Sadako's thousand cranes
   over Hiroshima). *[Non-Western symbol × history]* Risk: LOW.
8. **The Red Poppy × Barbed Wire** — remembrance × the trench that earned it; the flower that grew on
   the battlefield. *[Cultural symbol × history]* Risk: MEDIUM (wire must be a bold zigzag, not fine).
9. **The Titanic × The Iceberg** — human triumph × nature's indifference; the "unsinkable" and the
   thing that sank it. *[History]* Risk: LOW.
10. **Rosie the Riveter × The Raised Fist** — two icons of collective empowerment; the poster and the
    gesture of a movement. *[Modern graphic design × protest symbol]* Risk: MEDIUM (Rosie's face —
    keep it the flat poster, not realistic).

*(Cliché test passed: none are sun/moon, yin-yang, or two-lovers tropes.)*

## Top-3 full prompt pairs

### 1 — THE GREAT WAVE × THE MUSHROOM CLOUD
**Palette:** white *(foam / no shard)* · black **K** (0/0/0/100) · deep Prussian blue **C+M**
(≈95/70/0/25) · pale grey **cool-grey** (≈0/3/12/18).
**Wall A — The Great Wave**
```
Flat Japanese ukiyo-e woodblock (Hokusai's Great Wave reimagined flat), centred on plain solid white
with wide white margins: one giant cresting WAVE — a bold flat deep Prussian-blue body with white
foam claws curling over the top, thick black outline, a low pale-grey horizon sliver. Only 4 flat
colours: white, black, deep Prussian blue, pale grey. Hard edges, solid flat colour fills, high
contrast. No shading, no gradient, no background. Reads instantly as the great wave.
```
**Wall B — The Mushroom Cloud**
```
Companion piece in the IDENTICAL flat woodblock style and 4-colour palette (white, black, deep
Prussian blue, pale grey): one towering atomic MUSHROOM CLOUD rising from the horizon, its billowing
cap and stem as bold flat deep-blue-and-grey shapes with curling white highlights that echo the
wave's foam, thick black outline. Centred on plain solid white with wide white margins. Hard edges,
solid flat fills, high contrast. No shading, no gradient, no background. The cloud rises like a
second wave.
```
Rationale: same blue/white/grey with curling white "foam/billow" at corresponding heights → double
duty (§3); woodblock flat fills + 4 colours + bold silhouette → §1; one subject on white → §2/§5;
two sublimes echoing → concept.

### 2 — VAN GOGH'S STARRY NIGHT × A SPIRAL GALAXY
**Palette:** white *(stars / no shard)* · black **K** (0/0/0/100) · deep night-blue **C+M**
(≈95/70/0/20) · gold-yellow **Y** (0/0/100/0).
**Wall A — The Starry Night**
```
Flat poster / stained-glass reimagining of Van Gogh's "The Starry Night", centred on plain solid
white with wide white margins: bold flat swirling deep-night-blue sky curls, a few large gold-yellow
star discs and one gold crescent, a single black cypress flame-shape at one edge, thick black
outlines. Only 4 flat colours: white, black, deep night-blue, gold-yellow. Hard edges, solid flat
colour fills, high contrast. No brushwork texture, no gradient, no shading, no background. Reads
instantly as Starry Night.
```
**Wall B — The Spiral Galaxy**
```
Companion piece in the IDENTICAL flat style and 4-colour palette (white, black, deep night-blue,
gold-yellow): one grand SPIRAL GALAXY face-on — bold flat deep-night-blue spiral arms curling around
a gold-yellow core, scattered gold star discs, thick black outline, centred on plain solid white
with wide white margins. Hard edges, solid flat fills, high contrast. No gradient, no shading, no
background. The painted sky and the real cosmos — the same turbulence.
```
Rationale: both are blue swirls studded with gold star-discs at corresponding positions → double
duty near-maximal (§3); flat swirl shapes (de-textured) + 4 colours → §1; on white → §2; vision ×
reality → concept. Risk: keep Van Gogh as bold flat curls, NOT brushstrokes.

### 3 — TANK MAN × THE FLOWER IN THE RIFLE
**Palette:** white *(no shard)* · black **K** (0/0/0/100) · golden-yellow **Y** (0/0/100/0).
**Wall A — Tank Man**
```
Flat high-contrast screenprint / poster, centred on plain solid white with wide white margins: one
small lone FIGURE in bold black silhouette facing a row of massive TANK silhouettes (a few bold flat
black shapes), the tiny figure holding two small golden-yellow bags. Only 3 flat colours: white,
black, golden-yellow. Hard edges, solid flat fills, thick outlines, high contrast. No shading, no
gradient, no background, no facial detail. Reads instantly as one person stopping the tanks.
```
**Wall B — The Flower in the Rifle**
```
Companion piece in the IDENTICAL flat poster style and 3-colour palette (white, black,
golden-yellow): a single hand in bold black silhouette placing a golden-yellow FLOWER into the
barrel of a black RIFLE, centred on plain solid white with wide white margins. Hard edges, solid
flat fills, thick outlines, high contrast. No shading, no gradient, no background. One small bloom
against the machine of force.
```
Rationale: the golden-yellow held object (bags / flower) at corresponding positions → double duty
(§3); bold black silhouettes on white + 3 colours → §1/§5; on white → §2; two icons of peaceful
defiance (question of force / answer of refusal) → concept. Risk: LOW (both are silhouette-native).

*(AVOID list + verification: same as the top of this doc — negative-prompt the gradients / detail /
extra colours; then ceiling_straddle.py targeting B_good 15–25%, and a `--color-mode overlap
--restarts 8` run to confirm it reads.)*
