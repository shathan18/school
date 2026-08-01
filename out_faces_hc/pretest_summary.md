# Face pre-test -- do stark two-tone faces survive 300 shards?

> **SUPERSEDED by `face_render300.py`.** This script's core claim -- that an ideal
> uniform 12x12 shard grid is an *optimistic upper bound* -- is **false**. The real
> solver render of Poe and Dostoevsky is cleanly recognisable (face-box detail 0.745,
> at ~208 shards, under the 300 budget), while this script ranked the two-tone group
> *below* the smooth-oil controls. A uniform grid wastes cells on clear-white areas
> that cost the solver no shards, and cannot align its cell edges to the ink boundary.
> Use the render, not this table, to screen images.

## Caveat recorded up front

This direction is **grayscale**. Hue is discarded, so the colour-compatibility /
colour-agreeing double-duty result (`shadowart-noise.md`, `report_team.md`) does
**not** transfer to any pair drawn from this pool. Choosing grayscale faces means
giving up the double-duty argument on those walls. Flagged deliberately.

## Method

- Budget 300 shards / 2 walls = **150 per wall**; an ideal
  uniform tiling is a **12x12** grid, so one shard subtends **8.3% of
  head width**. Eyes, brows and mouth are all at or below that size -- which is why a
  uniform-grid argument predicts failure. The real solver does not tile uniformly.
- Row 3 of the contact sheet was *claimed* to be an optimistic upper bound. It is not;
  see the banner above. It is closer to a lower bound for flat, hard-edged sources.
- All detail metrics are restricted to a hand-set **face box** (brow-to-chin), because a
  first pass over the whole head crop ranked the *Mona Lisa* top: at feature scale the
  dominant energy is the long sharp edge of the hair/dress mass, which a coarse grid
  reproduces well. That was measuring silhouette, not face.
- Gates set from the physics beforehand: `detail_retain >= 0.3`,
  `n_bands >= 2`, `band_depth >= 0.15`.

## Results

| who | group | detail_retain | bands (sim/src) | band_depth | rank-1 id | margin | verdict |
|-----|-------|--------------:|:---------------:|-----------:|:---------:|-------:|---------|
| Edgar Allan Poe | two_tone | 0.560 | 2/1 | 1.000 | Y | +0.251 | PASS |
| Van Gogh self | oil_control | 0.518 | 1/2 | 0.333 | Y | +0.248 | FAIL |
| Mona Lisa | oil_control | 0.467 | 1/2 | 0.389 | Y | +0.526 | FAIL |
| Girl w/ Pearl Earring | oil_control | 0.457 | 2/2 | 0.298 | Y | +0.512 | PASS |
| Mallarme | two_tone | 0.442 | 2/3 | 0.159 | Y | +0.422 | PASS |
| Dostoevsky | two_tone | 0.414 | 1/2 | 0.328 | Y | +0.278 | FAIL |
| Munch self | oil_control | 0.385 | 0/1 | 0.000 | Y | +0.360 | FAIL |
| Ibsen | two_tone | 0.189 | 4/4 | 1.000 | Y | +0.269 | FAIL |
| Wagner | two_tone | 0.170 | 4/4 | 0.687 | Y | +0.386 | FAIL |

## Group means

| group | n | detail_retain | n_bands | band_depth |
|-------|--:|--------------:|--------:|-----------:|
| two_tone (test) | 5 | 0.355 | 2.60 | 0.635 |
| oil_control | 4 | 0.457 | 1.00 | 0.255 |

Two-tone advantage in face-box detail retention: **-0.102**

`pretest_contact.png` is the actual evidence; the numbers above are proxies.