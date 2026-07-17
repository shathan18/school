# ShadowArt — Where the Project Stands
**A guide for the team.** Written after the algorithm changed substantially; if you last looked at this a while ago, the shard-to-panel logic is different now.

---

## What the system does, in one paragraph

You give it **two different pictures**. It designs **one physical object** — a set of flat acrylic panels, standing at various angles, holding lots of small coloured shards. You put that object in the corner of a room with **two lamps**. Light the object from one direction and its shadow on the left wall forms **picture 1**. Light it from the other direction and its shadow on the right wall forms **picture 2**. Same object, two completely different images.

Both lamps are on **at the same time**. That fact is the source of every hard problem in this project — more on that below.

---

## Terms you'll need (defined once, used throughout)

- **Panel** — a flat sheet standing vertically in the room, at some angle. Shards get mounted on it. Also called a "plane."
- **Shard** — one small piece of coloured acrylic. The picture is built out of ~50–100 of these per wall.
- **Cross-talk** — the stray shadow. A shard placed to help build the *apple* on Wall A **also** throws a shadow onto Wall B, because both lamps are on. That second shadow is cross-talk. **It cannot be turned off** — it's physics, not a bug.
- **Bad cross-talk** — stray shadow landing on the *empty white background* of the other picture. Looks like a random splotch. This is the visible noise.
- **Good cross-talk / joint-intersection** — stray shadow landing somewhere *useful* on the other picture (a dark region, or a region wanting a similar colour). This is the panel "doing double duty" — serving both walls at once. **This is the interesting part of the concept.**
- **CMYK** — Cyan, Magenta, Yellow, blacK. Printers build any colour by layering these four inks. We do the same with transparent coloured acrylic: stack a cyan shard and a yellow shard and the light through them reads green.
- **Voronoi partition** — a way of cutting a shape into cells. Scatter some seed points; every pixel belongs to whichever seed is nearest. You get organic, irregular tiles (like cracked glass or a giraffe's coat). This is how we cut a picture into shards.
- **RMSE / SSIM** — quality scores comparing our simulated shadow against the target picture. **RMSE = colour error, lower is better.** **SSIM = structural similarity, higher is better** (does it *look* like the picture, shapes and edges intact).

---

## INPUT

### The two source images
Two ordinary image files. The code crops each to its subject, scales it to 92% of the wall, and centres it.

### Parameters — who decides what

**The USER sets these** (in `scenes/example.yaml`):

| Parameter | Current value | Notes |
|---|---|---|
| Wall size | 1.8 × 1.8 m | derived from the optics — not arbitrary |
| Light positions | 3.0 m out from each wall | derived (controls blur + magnification) |
| Material thickness | 3 mm acrylic | checked — this is real purchasable stock |
| Number of panels | 8 (shipped) / 14 (our experiments) | **just picked. Not derived.** |
| Shard size (`fragment_size`) | 0.135 m | derived from wall size |
| Shard count ceiling | 220 per wall | **just picked** — and it never actually kicks in (see below) |
| Colour threshold | 0.15 | **just picked** |
| Max colours stacked per shard | 3 | **just picked** |

**The ALGORITHM decides these:**
- Which panels go where, at what angle (if the search is used)
- How many shards, and how big each one is (adapts to picture detail)
- What colour each shard is
- **Which panel each shard gets mounted on** ← *this is the part we changed, and it's the whole story*

---

## ALGORITHM — step by step

### Steps 1–4 are standard setup. Step 5 is our actual contribution.

### 1. Panel count — **not clever. Just a number we chose.**
It's a fixed count (8 in the shipped config, 14 in our experiments). There's **no search over how many panels are needed**, and no stopping rule. We planned one and never built it. Be honest about this if asked.

### 2. Panel angles and positions — **random, then filtered**
Each candidate panel gets a **random** angle and a **random** position. It's then rejected unless it passes three checks:
- it sits inside the room, between each lamp and its wall
- it stands at least **0.5 m clear of both walls** (so the object doesn't look smeared against the glass)
- its shadow isn't magnified more than 3× (otherwise one giant blurry panel covers the whole wall)

We generate 16 candidates and keep the best one, repeatedly, choosing whichever adds the most *new* coverage of the picture. So the *selection* is smart, but each candidate is a random draw. **Different random seed → different sculpture.**

### 3. Cutting the picture into shards — **yes, it's Voronoi**
Plainly: **we use a Voronoi partition.** Concretely, in `decompose.py`:
1. Lay a regular grid of points over the picture, then **jiggle each point randomly** (so the result looks organic, not gridded).
2. **Every pixel joins whichever seed point is nearest** — that's the Voronoi step. Each resulting cell is one shard.
3. If a shard comes out too big, **re-run Voronoi inside it** to split it further (up to 3 levels deep).
4. Shards in *detailed* parts of the picture (edges, colour changes) are automatically made **smaller**; flat areas get big chunky shards. The code detects detail with an edge filter.
5. Tiny slivers aren't thrown away — they're absorbed into their neighbours, so the shards **tile the picture exactly, with no holes**.

**Important finding:** we swept shard count from 28 up to 394 per wall. **More shards makes the result *worse*, not better.** SSIM peaks at ~55 shards (Wall A) and falls steadily after:

| shards (Wall A) | SSIM (higher=better) |
|---|---|
| 28 | 0.801 |
| **55** | **0.796 ← sweet spot region** |
| 99 | 0.767 |
| 206 | 0.737 |
| 394 | 0.709 |

The 220 "ceiling" we'd set for fabrication reasons turns out to be irrelevant — quality degrades **long before** you hit it. **Fewer, bigger shards win.**

### 4. Shard colour — CMYK
For each shard: take the **most common colour** in that region (the mode, not the average — averaging would invent colours that aren't in the picture), convert it to CMYK, and keep the channels above a **0.15** threshold (up to 3). Those become physical layers of coloured acrylic stacked at that spot.

**Note a key limitation:** a shard's colour is chosen from **its own wall's picture only**. The other wall is never consulted. This matters — see limitations.

### 5. **Which panel does each shard go on? — THIS IS THE CONTRIBUTION**

Here's the insight that makes everything work.

A shard has to sit *somewhere along the light ray* between the lamp and the spot it's meant to darken. Sliding it **nearer or further along that ray doesn't change where its shadow lands on its own wall** — but it *completely changes where its stray shadow lands on the other wall*.

In our code, **choosing which panel to mount a shard on IS choosing that depth.** So the panel choice is the **one and only steering wheel** for cross-talk.

**And the original code turned that steering wheel at random** (`rng.choice`). That single line is why the noise varied wildly (14%–37%) depending on the random seed.

**What we do now:** for each shard, we look at every panel that could physically hold it, and we **compute where the stray shadow would land on the other wall** — then pick the panel that does **least damage** there. We compute this cheaply (no re-rendering — just a matrix multiply and ~200 pixel lookups per option).

We tried two versions of "damage":
- **Harm-only:** *"don't hurt the other picture."* The algorithm's answer: aim all stray shadows **off the wall entirely**. Noise vanished — but so did all the double-duty. It solved the problem by refusing to interact.
- **Signed (the good one):** *"don't hurt the other picture — but you get **credit** if your stray shadow lands somewhere genuinely useful."* Now it has a reason to *aim* the shadows at helpful places instead of just discarding them.

---

## THE CORE RESULT

**10 random seeds per arm. Panel placement identical across arms — only the assignment rule differs.**

| Version | Bad cross-talk (noise) ↓ | Good cross-talk (double-duty) ↑ | Wall A RMSE ↓ | Wall A SSIM ↑ | Wall B RMSE ↓ | Wall B SSIM ↑ | Panels used |
|---|---|---|---|---|---|---|---|
| **Random** (original `rng.choice`) | 23.1% | 15.4% | 0.243 | 0.778 | 0.240 | 0.783 | 13.7/14 |
| **Harm-only damage** | **2.5%** | 2.5% ✗ | **0.212** | **0.852** | **0.182** | **0.874** | 7.3/14 |
| **Signed damage (credit 0.5)** ⭐ | **4.6%** | **31.3%** | 0.218 | 0.809 | 0.188 | 0.843 | 12.4/14 |

### Read that table like this:

**Signed damage beats the original random code on BOTH axes at once:**
- **~5× less noise** (23.1% → 4.6%)
- **~2× more genuine double-duty** (15.4% → 31.3%)
- and **better picture quality on every measure** (RMSE down, SSIM up on both walls)

**The harm-only version is a trap.** Its picture-quality numbers are the best in the table — but look at the double-duty column: **2.5%**. It got clean pictures by making the panels stop serving both walls at all. You'd have built two separate sculptures that happen to share a room. **The thing that makes this project interesting would be gone.**

**Bonus:** harm-only also collapsed onto **7.3 of 14 panels** (half the structure standing empty). Signed damage uses **12.4/14** — the collapse was a symptom of the one-sided objective, and it fixed itself.

### The headline

> **Stray cross-talk cannot be eliminated — that's physics. But it can be AIMED. And aiming it beats trying to suppress it.**

We also found the trade-off is **partially separable, not fundamental**. Turning the credit dial up gets you *more* double-duty at the cost of *more* noise:

| credit weight | bad ↓ | good ↑ |
|---|---|---|
| 0.5 | 4.6% | 31.3% |
| 1.0 | 7.0% | 32.7% |
| 2.0 | 9.0% | 33.8% |

So there **is** a real trade-off at the margin — but the original random code was sitting nowhere near the good part of that curve. **0.5 is the sweet spot.**

---

## OUTPUT — what the pipeline actually produces

| File | What it is |
|---|---|
| `preview_final.png` | Target vs simulated shadow, both walls |
| `scene_interactive.html` | 3D model you can orbit in a browser |
| `cut/structure/` | DXF + SVG for laser-cutting the clear panels **with their slots** |
| `cut/stack0..2/` | DXF + SVG cut files, one per colour, per layer |
| `shards.obj` / `shards.ply` | 3D meshes |
| `shards.obj` rig | wall/panel/light markers to get orientation right on import |
| console | RMSE / SSIM / edge-fidelity scores, shard counts |

---

## ⚠️ WHAT CHANGED RECENTLY — read this if you're catching up

1. **The colour bug (fixed).** Golden-orange was rendering as **maroon** — and worse, golden-orange and pure red produced the *identical* colour. The cause: we were applying colour channels at *full strength* regardless of how much of that colour the pixel actually needed. Now it's **intensity-weighted**. Golden now looks golden.
2. **The shard-to-panel assignment (the big one).** Was `rng.choice` — literally random. Now it's the damage-minimising choice described above. **This is the actual contribution.**
3. **Shard density.** We used to believe more shards = more detail. **Wrong.** Measured: quality *peaks* at ~55 shards/wall and degrades after. We now run at about half the old density.
4. **The installation was rescaled** — bigger walls (1.8 m), longer light throw (3 m), smaller object.

## 🚨 AND ONE THING THAT IS **NOT** DONE

**None of the improvements are wired into the command-line tool.** If you run `shadowart run --color --color-mode overlap` right now, you get **the old 8 fixed panels and the old random assignment.** All the new work lives in the library but is only reachable from the test scripts in `out_thickness_test/`. **Somebody needs to wire it into `cli.py`.** This is the top of the to-do list.

---

## HONEST LIMITATIONS — state these, don't hide them

1. **Nothing has been physically built.** Everything here is simulation. We have not cut acrylic, assembled panels, or pointed a real lamp at anything. The renderer models penumbra (soft shadow blur) and material thickness, but **a real build could reveal problems we haven't imagined.**

2. **Tested on very few image pairs.** Mainly apple/breakfast and two CMYK posters. These are **simple, flat-colour graphics**. We do not know how this behaves on photographs, faces, or anything with subtle gradients.

3. **Many constants were just picked, not derived.** Panel count (14), candidate count (16), angle bounds, the 0.5 m standoff, the 3× magnification cap, the 0.15 colour threshold, the 220 shard ceiling, the damage weights (0.5/0.5). **If challenged on any of these, the honest answer is "we chose it, and here's roughly why" — not "we derived it."**

4. **The angle bound conflict (unresolved).** Our best results widen panel angles to 5°–85°, but the *original* 20°–70° limit existed for a **fabrication** reason: two panels crossing at a shallow angle need a very wide slot (`thickness / sin θ`). **We have not re-checked the slot widths for the wider range.** This must be verified before anything is cut.

5. **The "credit" term is arguably measuring the wrong thing.** It rewards a shard for landing somewhere "closer to the target than blank white." But a **red** shard on a **golden** croissant scores as "helpful" by that rule — red *is* closer to gold than white is — yet visually it's still contamination. **You can see this in the renders: the stray shadows moved off the background and onto the croissant.** So the 31.3% "good cross-talk" figure counts *landing on the subject*, not necessarily *helping* it. Treat that number with care.

6. **The marginal trade-off is real.** Pushing for more double-duty does bring noise back (see the credit-weight table). This is not free.

7. **Fundamental limit:** a shard has **one colour**, taken from its own wall's picture. When its stray shadow hits the other wall, it arrives with **the wrong colour**. It can be *aimed* somewhere less harmful — it can never arrive *correct*, unless the two pictures happen to want compatible colours at geometrically-linked points. **True double-duty is only possible where the two images happen to agree.**

---

## TO-DO

1. **Wire the search + damage assignment into `cli.py`** (currently the CLI runs the old code)
2. Re-check oblique slot widths for the 5°–85° angle range before cutting
3. Fix the credit term to reward *colour agreement*, not merely "darker than white"
4. Test on more image pairs, including photographs
5. Build a small physical prototype
