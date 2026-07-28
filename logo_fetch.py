"""Fetch university / faculty logos from Wikimedia Commons for the shard simulation.

Why logos at all: the face campaign proved the binding constraint is INK AREA, not
subject type -- white paper = clear perspex = no shard, and bare wall cannot intersect
the other wall. Logos are flat, hard-edged and 2-tone by construction, so they are the
best possible case for coarse shards; the open question is purely whether they carry
enough material.

Run:  python logo_fetch.py
Out:  examples/logos/<key>.png   (PNG, alpha flattened onto white)

Note on rights: these are trademarks. This is non-commercial academic coursework and the
files are hosted on Commons (simple-geometry logos are {{PD-textlogo}} there), but the
marks remain the property of their institutions. Do not reuse the renders commercially.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "shadowart-student-project/1.0 (academic coursework simulation)"}
OUT = Path("examples/logos")
THUMB_W = 512          # a standard width -> served from cache, avoids on-demand throttling

# Commons file titles, chosen from a `generator=search` sweep (see search_terms()).
WANTED = {
    "technion":   "File:Technion logo.svg",
    "technion5":  "File:Technion-logo5.png",
    "huji":       "File:Hebrew University Logo.svg",
    "tau":        "File:Tel Aviv university logo - English.png",
    "bgu":        "File:Logo-bg.svg",
}

SEARCH_TERMS = [
    "Technion computer science",
    "Faculty of Computer Science Technion",
    "Taub Faculty",
    "Technion faculty logo",
    "Weizmann Institute of Science logo",
]


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    return json.loads(_get(url))


def _get(url: str, tries: int = 5) -> bytes:
    """GET with exponential backoff. Commons returns 429 aggressively for SVG rendering;
    a plain retry loop is the difference between 2/5 and 5/5 downloads."""
    delay = 2.0
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            print(f"      429, backing off {delay:.0f}s ...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def search_terms() -> None:
    """Print candidates so the WANTED table above can be curated by hand, not guessed."""
    for term in SEARCH_TERMS:
        try:
            d = api(dict(action="query", format="json", generator="search",
                         gsrsearch=term, gsrlimit=8, gsrnamespace=6,
                         prop="imageinfo", iiprop="url|size|mime"))
            pages = (d.get("query", {}) or {}).get("pages", {}) or {}
            print(f"== {term} -> {len(pages)}")
            for p in pages.values():
                ii = p["imageinfo"][0]
                print(f"    {p['title'][:64]:64s} {ii['mime']:16s} {ii['width']}x{ii['height']}")
        except Exception as e:
            print(f"== {term} ERR {type(e).__name__} {e}")
        time.sleep(0.6)


def thumb_url(title: str) -> tuple[str, str]:
    """(url, mime). SVGs come back as a rendered PNG thumbnail at THUMB_W."""
    d = api(dict(action="query", format="json", titles=title,
                 prop="imageinfo", iiprop="url|size|mime", iiurlwidth=THUMB_W))
    page = next(iter((d.get("query", {}) or {}).get("pages", {}).values()))
    ii = page["imageinfo"][0]
    return ii.get("thumburl") or ii["url"], ii["mime"]


def fetch(key: str, title: str) -> Path | None:
    dest = OUT / f"{key}.png"
    if dest.exists():
        print(f"  {key:10s} cached  {dest}")
        return dest
    try:
        url, mime = thumb_url(title)
        raw = _get(url)
        tmp = OUT / f"_{key}.bin"
        tmp.write_bytes(raw)
        im = Image.open(tmp)
        # Logos are usually RGBA with a transparent ground. Flatten onto WHITE so the
        # "normal polarity" arm is honest dark-mark-on-white; inversion happens later.
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im.convert("RGB").save(dest)
        tmp.unlink()
        print(f"  {key:10s} OK      {im.size[0]}x{im.size[1]}  <- {title}  ({mime})")
        return dest
    except Exception as e:
        print(f"  {key:10s} FAIL    {type(e).__name__}: {e}  <- {title}")
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--search" in sys.argv:
        print("--- searching for a CS-faculty logo ---------------------------------")
        search_terms()
    print("--- downloading curated set -----------------------------------------")
    got = {}
    for k, t in WANTED.items():
        got[k] = fetch(k, t)
        time.sleep(1.5)                 # be a polite client; 429s are self-inflicted
    ok = [k for k, v in got.items() if v]
    print(f"\n{len(ok)}/{len(WANTED)} downloaded -> {OUT}: {', '.join(ok)}")


if __name__ == "__main__":
    main()
