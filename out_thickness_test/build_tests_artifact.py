# -*- coding: utf-8 -*-
"""Compact results page for the two redistribution/posterise tests (embeds both comparison PNGs)."""
import base64, os
def b64(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()
feat = b64("out_thickness_test/feature_mask/compare.png")
post = b64("out_thickness_test/posterise/compare.png")

HTML = f"""<title>Two tests: narrow feature mask · posterised face</title>
<style>
:root{{--bg:#EDEBE6;--card:#F7F5F0;--ink:#232020;--mut:#6B645C;--line:#DAD5CC;
  --accent:#7A5C3E;--null:#8A5A4C;--good:#5B7A4E;--grid:#E4DFD6;
  --sh:0 1px 2px rgba(35,32,32,.06),0 6px 20px rgba(35,32,32,.07);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171412;--card:#201C19;--ink:#EDE6DB;--mut:#A99C8C;
  --line:#332C26;--accent:#C79A66;--null:#D08A72;--good:#9BB37C;--grid:#2B2420;
  --sh:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.4);}}}}
:root[data-theme="light"]{{--bg:#EDEBE6;--card:#F7F5F0;--ink:#232020;--mut:#6B645C;--line:#DAD5CC;
  --accent:#7A5C3E;--null:#8A5A4C;--good:#5B7A4E;--grid:#E4DFD6;--sh:0 1px 2px rgba(35,32,32,.06),0 6px 20px rgba(35,32,32,.07);}}
:root[data-theme="dark"]{{--bg:#171412;--card:#201C19;--ink:#EDE6DB;--mut:#A99C8C;--line:#332C26;
  --accent:#C79A66;--null:#D08A72;--good:#9BB37C;--grid:#2B2420;--sh:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.4);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55;}}
.wrap{{max-width:940px;margin:0 auto;padding:clamp(26px,5vw,60px) clamp(16px,4vw,36px) 72px;}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 14px;}}
h1{{font-family:Georgia,"Palatino Linotype",serif;font-size:clamp(28px,5vw,44px);margin:0;letter-spacing:-.01em;line-height:1.05;text-wrap:balance;}}
.lede{{color:var(--mut);max-width:64ch;margin:16px 0 0;font-size:16px;}}
.test{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(18px,3vw,30px);margin-top:34px;box-shadow:var(--sh);}}
.tnum{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);font-weight:600;}}
.test h2{{font-family:Georgia,serif;font-size:22px;margin:6px 0 0;font-weight:600;}}
.verdict{{display:inline-flex;align-items:center;gap:8px;margin:16px 0 4px;font-weight:700;font-size:15px;padding:7px 14px;border-radius:9px;}}
.verdict.null{{color:var(--null);background:color-mix(in srgb,var(--null) 12%,transparent);}}
.verdict.mix{{color:var(--good);background:color-mix(in srgb,var(--good) 13%,transparent);}}
.test p{{margin:12px 0 0;}} .test p.tight{{max-width:66ch;}}
.tbl{{overflow-x:auto;margin-top:18px;border:1px solid var(--line);border-radius:10px;}}
table{{border-collapse:collapse;width:100%;min-width:440px;font-variant-numeric:tabular-nums;}}
th{{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);font-weight:600;text-align:right;padding:11px 14px 8px;border-bottom:1px solid var(--line);}}
th.l,td.l{{text-align:left;}}
td{{text-align:right;padding:9px 14px;font-size:13.5px;border-bottom:1px solid var(--grid);}}
tr:last-child td{{border-bottom:0;}}
td.dn{{color:var(--null);font-weight:600;}} td.up{{color:var(--good);font-weight:600;}}
.delta{{font-size:13px;color:var(--mut);margin-top:10px;}}
.delta b.null{{color:var(--null);}} .delta b.up{{color:var(--good);}}
figure{{margin:20px 0 0;}}
img{{width:100%;height:auto;display:block;border-radius:10px;border:1px solid var(--line);background:#fff;}}
figcaption{{color:var(--mut);font-size:12.5px;margin-top:9px;text-align:center;}}
.why{{margin-top:20px;padding:15px 18px;border-left:3px solid var(--accent);background:color-mix(in srgb,var(--accent) 6%,var(--card));border-radius:9px;font-size:14px;max-width:70ch;}}
.why b{{color:var(--accent);}}
.foot{{margin-top:40px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:16px;}}
</style>
<div class="wrap">
  <p class="eyebrow">Shadow-Art · Redistribution &amp; Posterise Tests</p>
  <h1>Can a face be pushed into a 300-shard budget?</h1>
  <p class="lede">Two ways to try to make a portrait resolve without adding shards: concentrate the
  shards we have onto the features, or simplify the target so what we have is enough. Control config
  (signed-damage, colour-agreement credit), 3 seeds, Girl-with-a-Pearl-Earring front view.</p>

  <div class="test">
    <p class="tnum">Test 1 · redistribution</p>
    <h2>Narrow feature-only mask — eyes / nose / lips</h2>
    <p class="tight">Same mechanism as the whole-face semantic mask (which was null at every budget):
    it moves shards, it does not add them. Narrower target (4.4% of the wall), so it could plausibly
    <em>hurt</em> by starving the rest.</p>
    <div class="verdict null">● NULL — and it does not hurt</div>
    <div class="tbl"><table>
      <thead><tr><th class="l">arm</th><th>feature SSIM</th><th>whole-face SSIM</th><th>global SSIM</th></tr></thead>
      <tbody>
        <tr><td class="l">control (no mask)</td><td>0.347 ± 0.038</td><td>0.393 ± 0.025</td><td>0.719 ± 0.011</td></tr>
        <tr><td class="l">feature-mask</td><td>0.356 ± 0.044</td><td>0.399 ± 0.027</td><td>0.719 ± 0.011</td></tr>
      </tbody>
    </table></div>
    <div class="delta">feature-SSIM Δ = <b class="null">+0.009</b>, against per-seed noise of ~0.044 → indistinguishable from zero.
    Global unchanged, so it doesn't starve the rest either. Confirmed at ~100 and ~250 shards/wall.</div>
    <figure><img src="{feat}" alt="control vs feature-mask reconstruction, feature box in red">
      <figcaption>Inside the red feature box the two reconstructions are near-identical — finer shards, same mush.</figcaption></figure>
  </div>

  <div class="test">
    <p class="tnum">Test 2 · simplify the target</p>
    <h2>Posterised face — flattened to 7 solid tones</h2>
    <p class="tight">Structure-preserving posterisation (luminance-band, keeps dark eyes/lips distinct).
    Same pipeline and seeds on the photographic vs posterised target, ~250 shards/wall.</p>
    <div class="verdict mix">◑ Real fidelity gain — but not a recognizability unlock</div>
    <div class="tbl"><table>
      <thead><tr><th class="l">target</th><th>face SSIM</th><th>global SSIM</th></tr></thead>
      <tbody>
        <tr><td class="l">photographic</td><td>0.418 ± 0.024</td><td>0.723 ± 0.012</td></tr>
        <tr><td class="l">posterised (7 tones)</td><td class="up">0.461 ± 0.014</td><td>0.709 ± 0.015</td></tr>
      </tbody>
    </table></div>
    <div class="delta">face-SSIM Δ = <b class="up">+0.042</b> (noise ~0.02) → a <b>real</b> gain, robust across seeds and budgets.
    Small global cost (−0.014). A flat target is genuinely easier for the medium to reproduce.</div>
    <figure><img src="{post}" alt="photographic vs posterised source and reconstruction">
      <figcaption>Top: photographic source → reconstruction. Bottom: posterised source → reconstruction.
      The posterised <em>source</em> reads clearly; its reconstruction is cleaner in tone but the face still does not resolve.</figcaption></figure>
    <div class="why"><b>Why the gain doesn't become recognizability.</b> Posterising flattens <em>tone</em>, but a
    face's recognizability lives in <em>small features</em> — eyes, nostrils, lip line — and posterising does
    not make those features any larger. The reconstruction is still limited by shard size, not by the target's
    tonal complexity, so the same ~9×-budget resolution wall stands. Posterising pays off when it turns an image
    into <em>bold flat shapes</em> (graphic/stencil imagery); a naturalistic portrait's features are too subtle
    for that, so a posterised face is <em>more faithful</em> but not more <em>recognizable</em> at this budget.</div>
  </div>

  <div class="foot">3 seeds each · control config: signed-damage, damage_weight 0.5, credit_weight 0.5, colour-agreement credit match_tol 0.30 ·
  images: out_thickness_test/feature_mask/compare.png, out_thickness_test/posterise/compare.png</div>
</div>
"""
p = "out_thickness_test/tests_report.html"
open(p, "w", encoding="utf-8").write(HTML)
print("wrote", p, len(HTML), "chars")
