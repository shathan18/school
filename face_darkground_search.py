"""Stage 1 of the dark-ground portrait search: cast a wide net, screen WITHOUT crop boxes.

WHY THIS EXISTS
---------------
The GRAY_L paper tint in `face_paper_floor.py` was a workaround, not a fix. It bought
intersection (2/10 -> 7/10 planes serving both images) by making the bare paper into
material, at the cost of clamping the target's contrast to GRAY_L (~0.78) and burying the
face under cross-talk: face detail 0.74 -> 0.35.

But the tint was only needed because I cast the piece badly. Measured, clear paper:

    subject      native ink area    good% (wall B)
    poe               44%               0.34
    mallarme          69%              11.74

35x more duty from ink area alone, at FULL contrast. Cross-talk needs material to exist,
and ink IS the material. So the real lever is casting: find portraits that are natively
DARK-GROUND (light head on a black field, or a black hair/coat mass that fills the frame)
and no tint is required at all.

Screening here is deliberately BOX-FREE. The crop boxes are hand-placed and were wrong for
years (see `face_crop_check.py`); requiring one per candidate would make a wide search
impossible and would re-introduce exactly that error. So stage 1 ranks on properties that
need no box, and only the finalists get hand-boxed in stage 2.

Run:  python face_darkground_search.py            # fetch + screen
      python face_darkground_search.py --no-fetch # re-screen what is already on disk
Out:  examples/faces_dg/<key>.jpg
      out_faces_hc/search/darkground_contact.png   <- LOOK AT THIS
      out_faces_hc/search/darkground.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import face_pretest as FP

SRC = Path("examples/faces_dg")
OUT = Path("out_faces_hc/search")
API = "https://commons.wikimedia.org/w/api.php"
UA = "shadowart-research/1.0 (academic sculpture study; contact via repo)"
WORK = 256
# Wikimedia only serves PRE-RENDERED thumbnail widths without rate-limiting. Asking for an
# arbitrary width (I used 1100) forces on-demand rasterisation and gets you 429'd off the
# service. 800 is on the cached list.
THUMB_W = 800


def say(*parts) -> None:
    """print() that cannot die on a Japanese print title under a cp1252 console."""
    s = " ".join(str(p) for p in parts)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(s.encode(enc, "replace").decode(enc, "replace"))

# Commons search terms. Aimed at flat-mass, heavy-black, PUBLIC-DOMAIN portrait prints of
# men (project rule: no women / religion / politics / mythology as subjects).
# Vallotton d.1925, Sharaku fl.1794, Munch d.1944 -> all clear of life+70.
QUERIES = [
    ("vallotton", 'Vallotton woodcut portrait', 12),
    ("sharaku",   'Sharaku kabuki actor portrait print', 12),
    ("okubie",    'okubi-e actor head print ukiyo-e', 10),
    ("munch",     'Munch woodcut portrait man', 8),
    ("woodcut",   'woodcut portrait man profile black white', 12),
    ("mezzotint", 'mezzotint portrait head man', 10),
    ("linocut",   'linocut portrait man high contrast', 8),
]
BAD_WORDS = re.compile(
    r"(madonna|christ|jesus|saint|st\.|virgin|buddha|god|angel|nude|"
    r"woman|women|girl|lady|female|mrs|miss|"
    r"president|king|queen|emperor|politic|war|flag|"
    r"venus|apollo|muse|myth|nymph)", re.I)


@dataclass
class Shot:
    key: str
    title: str
    path: str
    ink: float = 0.0          # fraction of frame that is NOT paper-white after 2-tone
    n_blobs: int = 0          # connected ink components >= 0.5% of frame
    biggest: float = 0.0      # largest ink component, fraction of frame
    edge_rate: float = 0.0    # ink-boundary length / ink area -> hatching detector
    contrast: float = 0.0     # 98th - 2nd percentile of luminance
    score: float = 0.0


def _api(params: dict, tries: int = 5) -> dict:
    """Commons API with backoff. The first version of this made one call PER FILE and got
    HTTP 429'd off the service after ~8 hits; everything below is batched instead."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            print(f"    (429, backing off {delay:.0f}s)")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch() -> None:
    """Search Commons and pull a downscaled copy of each hit. Skips anything already on
    disk, so re-runs are cheap and the pool is reproducible.

    ONE api call per query term: `generator=search` + `prop=imageinfo` returns titles and
    thumbnail URLs together. Image bytes come from upload.wikimedia.org, which is a
    separate budget from the API."""
    SRC.mkdir(parents=True, exist_ok=True)
    for tag, term, n in QUERIES:
        say(f"  [{tag}] {term}")
        try:
            data = _api({"action": "query", "generator": "search",
                         "gsrsearch": f"{term} filetype:bitmap", "gsrnamespace": 6,
                         "gsrlimit": n * 3, "prop": "imageinfo",
                         "iiprop": "url", "iiurlwidth": THUMB_W})
            pages = list(data.get("query", {}).get("pages", {}).values())
        except Exception as e:
            say(f"  ! search failed for {tag!r}: {e}")
            continue
        time.sleep(1.5)

        kept = 0
        for pg in pages:
            if kept >= n:
                break
            title = pg.get("title", "")
            if BAD_WORDS.search(title):
                say(f"    - skip (subject rule): {title}")
                continue
            stem = title.split(":", 1)[-1].rsplit(".", 1)[0].lower()
            key = f"{tag}_{re.sub(r'[^a-z0-9]+', '_', stem)[:38]}"
            dst = SRC / f"{key}.jpg"
            if dst.exists():
                kept += 1
                continue
            try:
                url = pg["imageinfo"][0]["thumburl"]
            except (KeyError, IndexError):
                continue
            if _download(url, dst):
                say(f"    + {key}")
                kept += 1
            else:
                say(f"    ! failed: {title}")


def _download(url: str, dst: Path, tries: int = 4) -> bool:
    delay = 3.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read()
            Image.open(io.BytesIO(raw)).convert("RGB").save(dst, quality=92)
            time.sleep(0.8)
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                return False
            say(f"    (429 on download, backing off {delay:.0f}s)")
            time.sleep(delay)
            delay *= 2
        except Exception:
            return False
    return False


def measure(p: Path) -> dict:
    """Box-free structure metrics on the whole print.

    `ink` is the headline: it is the fraction of the frame that will be MATERIAL, and
    material is what cross-talk requires. `edge_rate` separates flat mass from hatching --
    a hatched face has the same ink area as a flat one but many times the boundary length,
    and hatching is destroyed by 300 shards.
    """
    img = Image.open(p).convert("RGB")
    img.thumbnail((WORK, WORK), Image.LANCZOS)
    g = np.asarray(img, dtype=np.float32).mean(-1) / 255.0
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    contrast = float(hi - lo)
    g = np.clip((g - lo) / max(1e-6, hi - lo), 0.0, 1.0)

    two = FP.posterize_gray(g, 2)
    ink = two < 0.5                                   # dark tone == cut material
    frac = float(ink.mean())

    lab, n = ndi.label(ink)
    if n:
        sizes = np.bincount(lab.ravel())[1:] / ink.size
        big = sizes[sizes >= 0.005]
        biggest = float(sizes.max())
    else:
        big, biggest = np.array([]), 0.0

    per = float((ink ^ ndi.binary_erosion(ink)).sum())
    edge_rate = per / max(1.0, ink.sum())

    return dict(ink=frac, n_blobs=int(big.size), biggest=biggest,
                edge_rate=float(edge_rate), contrast=contrast)


def score(s: Shot) -> float:
    """Rank for "will intersect AND survive 300 shards", with no crop box available.

    ink        -> plateau reward: below 0.55 there is not enough material to cross-talk
                  (poe, 0.44, gave 0.34% duty); above ~0.85 the print is a black slab.
    biggest    -> one dominant mass, the lecturer's 300-shard criterion.
    edge_rate  -> penalise hatching; flat masses have short boundaries per unit area.
    contrast   -> a print that is already low-contrast has nothing to posterise.
    """
    ink_term = float(np.clip((s.ink - 0.45) / 0.25, 0.0, 1.0)) * \
               float(np.clip((0.95 - s.ink) / 0.15, 0.0, 1.0))
    mass_term = float(np.clip(s.biggest / 0.35, 0.0, 1.0))
    flat_term = float(np.clip((0.30 - s.edge_rate) / 0.22, 0.0, 1.0))
    con_term = float(np.clip(s.contrast / 0.65, 0.0, 1.0))
    return ink_term * (0.25 + 0.75 * mass_term) * (0.25 + 0.75 * flat_term) * con_term


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_fetch:
        print("fetching from Wikimedia Commons ...")
        fetch()

    # the incumbents, so the new pool is judged against something known
    for c in FP.CANDIDATES:
        if c.group == "two_tone" and Path(c.path).exists():
            dst = SRC / f"incumbent_{c.key}.jpg"
            if not dst.exists():
                Image.open(c.path).convert("RGB").save(dst, quality=92)

    shots = []
    for p in sorted(SRC.glob("*.jpg")):
        try:
            m = measure(p)
        except Exception as e:
            say(f"  ! measure {p.name}: {e}")
            continue
        s = Shot(key=p.stem, title=p.stem, path=str(p), **m)
        s.score = score(s)
        shots.append(s)

    shots.sort(key=lambda s: -s.score)
    say(f"\n{'key':46s}{'ink':>7s}{'big':>7s}{'edge':>7s}{'con':>7s}{'blobs':>7s}{'score':>8s}")
    say("-" * 88)
    for s in shots:
        mark = "  <== incumbent" if s.key.startswith("incumbent") else ""
        say(f"{s.key[:45]:46s}{s.ink:>7.3f}{s.biggest:>7.3f}{s.edge_rate:>7.3f}"
            f"{s.contrast:>7.3f}{s.n_blobs:>7d}{s.score:>8.3f}{mark}")

    (OUT / "darkground.json").write_text(
        json.dumps([asdict(s) for s in shots], indent=2), encoding="utf-8")

    top = shots[:24]
    cols = 6
    rows = (len(top) + cols - 1) // cols
    fig, ax = plt.subplots(rows * 2, cols, figsize=(2.4 * cols, 2.6 * rows))
    ax = np.atleast_2d(ax)
    for i, s in enumerate(top):
        r, c = divmod(i, cols)
        img = Image.open(s.path).convert("RGB")
        img.thumbnail((WORK, WORK), Image.LANCZOS)
        g = np.asarray(img, dtype=np.float32).mean(-1) / 255.0
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = np.clip((g - lo) / max(1e-6, hi - lo), 0, 1)
        ax[2 * r, c].imshow(g, cmap="gray", vmin=0, vmax=1)
        ax[2 * r, c].set_title(f"{i}. {s.key[:26]}", fontsize=6)
        ax[2 * r + 1, c].imshow(FP.posterize_gray(g, 2), cmap="gray", vmin=0, vmax=1)
        ax[2 * r + 1, c].set_title(f"ink {s.ink:.2f}  S {s.score:.2f}", fontsize=6)
    for a_ in ax.ravel():
        a_.axis("off")
    fig.suptitle("Dark-ground portrait search -- top row: source, bottom row: 2-tone (what "
                 "the solver aims at). 'ink' is the fraction that becomes MATERIAL.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "darkground_contact.png", dpi=110)
    print(f"\nwrote {OUT/'darkground_contact.png'}  ({len(shots)} candidates)")


if __name__ == "__main__":
    main()
