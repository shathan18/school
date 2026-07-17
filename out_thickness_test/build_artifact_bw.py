# -*- coding: utf-8 -*-
"""Self-contained artifact for the B&W Girl front/back deliverable (monochrome theme)."""
import base64, json, os
OUT = "out_thickness_test/bw_final"
def b64(p):
    with open(os.path.join(OUT, p), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()
img = {k: b64(f"{k}.png") for k in ("srcA", "reconA", "srcB", "reconB")}
M = json.load(open(os.path.join(OUT, "metrics.json"))); ps = {r["seed"]: r for r in M["per_seed"]}
mean = M["mean"]; BEST = M["best_seed"]; b2 = ps[BEST]
maxbgA = max(r["bg"]["A"] for r in M["per_seed"])
def rows():
    out = []
    for s in (1, 2, 3, 4, 5):
        r = ps[s]; A, B, bg = r["A"], r["B"], r["bg"]
        cls = ' class="shown"' if s == BEST else ""; tag = ' <span class="tag">shown</span>' if s == BEST else ""
        out.append(f'<tr{cls}><td class="seed">{s}{tag}</td>'
            f'<td>{A["rmse"]:.3f}</td><td>{A["ssim"]:.3f}</td><td>{A["edge_fidelity"]:.2f}</td>'
            f'<td class="sep">{B["rmse"]:.3f}</td><td>{B["ssim"]:.3f}</td><td>{B["edge_fidelity"]:.2f}</td>'
            f'<td class="sep good">{bg["A"]:.1f}%</td><td class="good">{bg["B"]:.1f}%</td></tr>')
    return "\n".join(out)

HTML = f"""<title>The Same Portrait, in Monochrome — Dual-Wall Shadow Reconstruction</title>
<style>
:root{{--bg:#E9E8E5;--card:#F5F4F1;--card2:#EDECE9;--ink:#20211F;--mut:#5F615D;--line:#D5D4CF;
  --accent:#4E5C63;--soft:#8A979D;--good:#3C6A62;--grid:#E1E0DB;--sh:0 1px 2px rgba(20,20,18,.06),0 6px 20px rgba(20,20,18,.07);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--card:#1E1E1C;--card2:#252523;--ink:#EAE8E2;--mut:#A6A49C;
  --line:#323230;--accent:#A9BAC0;--soft:#7D8B91;--good:#8FB6AD;--grid:#2A2A28;--sh:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.4);}}}}
:root[data-theme="light"]{{--bg:#E9E8E5;--card:#F5F4F1;--card2:#EDECE9;--ink:#20211F;--mut:#5F615D;--line:#D5D4CF;
  --accent:#4E5C63;--soft:#8A979D;--good:#3C6A62;--grid:#E1E0DB;--sh:0 1px 2px rgba(20,20,18,.06),0 6px 20px rgba(20,20,18,.07);}}
:root[data-theme="dark"]{{--bg:#151513;--card:#1E1E1C;--card2:#252523;--ink:#EAE8E2;--mut:#A6A49C;--line:#323230;
  --accent:#A9BAC0;--soft:#7D8B91;--good:#8FB6AD;--grid:#2A2A28;--sh:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.4);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55;}}
.wrap{{max-width:1060px;margin:0 auto;padding:clamp(28px,5vw,68px) clamp(18px,4vw,40px) 80px;}}
.eyebrow{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 18px;}}
h1{{font-family:Georgia,"Iowan Old Style","Palatino Linotype",serif;font-size:clamp(32px,6vw,56px);line-height:1.02;font-weight:600;margin:0;letter-spacing:-.01em;text-wrap:balance;}}
.lede{{font-size:clamp(16px,2vw,19px);color:var(--mut);max-width:62ch;margin:20px 0 0;}} .lede b{{color:var(--ink);font-weight:600;}}
.rule{{height:1px;background:var(--line);border:0;margin:44px 0;}}
.h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);font-weight:600;margin:0 0 20px;}}
.walls{{display:grid;grid-template-columns:1fr 1fr;gap:26px;}} @media (max-width:760px){{.walls{{grid-template-columns:1fr;}}}}
.wall figure{{margin:0;}} .wcap{{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:0 2px 12px;}}
.wcap .name{{font-weight:600;font-size:15px;}} .wcap .sub{{color:var(--mut);font-size:12.5px;}}
.cmp{{position:relative;width:100%;aspect-ratio:1/1;border-radius:10px;overflow:hidden;background:var(--card);box-shadow:var(--sh);border:1px solid var(--line);user-select:none;touch-action:none;}}
.cmp img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;}}
.cmp .over{{clip-path:inset(0 calc(100% - var(--pos,50%)) 0 0);}}
.cmp .lab{{position:absolute;top:10px;font-size:10.5px;letter-spacing:.14em;font-weight:600;text-transform:uppercase;padding:4px 8px;border-radius:5px;background:color-mix(in srgb,var(--card) 80%,transparent);color:var(--ink);backdrop-filter:blur(3px);pointer-events:none;}}
.cmp .lab.l{{left:10px;}} .cmp .lab.r{{right:10px;color:var(--accent);}}
.cmp .divide{{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:2px;background:var(--accent);transform:translateX(-1px);pointer-events:none;}}
.cmp .grip{{position:absolute;top:50%;left:var(--pos,50%);width:38px;height:38px;margin:-19px 0 0 -19px;border-radius:50%;background:var(--accent);border:3px solid var(--card);box-shadow:var(--sh);display:grid;place-items:center;cursor:ew-resize;color:var(--card);}}
.cmp .grip svg{{width:18px;height:18px;}}
.cmp input[type=range]{{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize;}}
.cmp:focus-within .divide{{box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 40%,transparent);}}
.chips{{display:flex;flex-wrap:wrap;gap:7px;margin:14px 2px 0;}}
.chip{{font-size:12px;color:var(--mut);background:var(--card);border:1px solid var(--line);padding:4px 9px;border-radius:20px;font-variant-numeric:tabular-nums;}}
.chip b{{color:var(--ink);font-weight:600;}} .chip.k b{{color:var(--good);}}
.hint{{text-align:center;color:var(--mut);font-size:12.5px;margin:22px auto 0;max-width:70ch;}}
.finding{{display:grid;grid-template-columns:auto 1fr;gap:clamp(20px,4vw,48px);align-items:center;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:clamp(22px,3.5vw,36px);box-shadow:var(--sh);}}
@media (max-width:680px){{.finding{{grid-template-columns:1fr;}}}}
.bignum .n{{font-size:clamp(52px,9vw,86px);line-height:.9;font-weight:600;color:var(--accent);letter-spacing:-.02em;font-variant-numeric:tabular-nums;}}
.bignum .pm{{font-size:20px;color:var(--mut);font-weight:500;}} .bignum .cap{{font-size:12.5px;color:var(--mut);margin-top:10px;max-width:24ch;}}
.finding p{{margin:0 0 12px;}} .finding p:last-child{{margin-bottom:0;}}
.cf{{display:flex;align-items:center;gap:12px;margin-top:16px;flex-wrap:wrap;font-variant-numeric:tabular-nums;font-size:14px;}}
.cf .pill{{padding:5px 11px;border-radius:8px;border:1px solid var(--line);background:var(--card2);}}
.cf .pill b{{color:var(--ink);}} .cf .arrow{{color:var(--mut);}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card);box-shadow:var(--sh);}}
table{{border-collapse:collapse;width:100%;min-width:560px;font-variant-numeric:tabular-nums;}}
thead th{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);font-weight:600;text-align:right;padding:14px 12px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}}
thead th.grp{{color:var(--accent);text-align:left;}}
tbody td{{text-align:right;padding:9px 12px;font-size:13.5px;border-bottom:1px solid var(--grid);}}
tbody td.seed{{text-align:left;color:var(--mut);}} tbody td.sep{{border-left:1px solid var(--line);}} tbody td.good{{color:var(--good);font-weight:600;}}
tr.shown td{{background:color-mix(in srgb,var(--accent) 9%,transparent);}} tr.shown td.seed{{color:var(--ink);font-weight:600;}}
.tag{{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--card);background:var(--accent);padding:2px 6px;border-radius:4px;margin-left:6px;vertical-align:middle;}}
tfoot td{{text-align:right;padding:12px;font-size:13.5px;font-weight:600;border-top:2px solid var(--line);}}
tfoot td.seed{{text-align:left;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;font-size:11px;}} tfoot td.sep{{border-left:1px solid var(--line);}} tfoot td.good{{color:var(--good);}}
.notes{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}} @media (max-width:680px){{.notes{{grid-template-columns:1fr;}}}}
.note{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:16px 20px;font-size:14px;max-width:70ch;}}
.note b{{color:var(--accent);}} .note .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:0 0 8px;font-weight:600;}}
.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;}}
.mcard{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:16px 18px;}}
.mcard .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:0 0 8px;}}
.mono{{font-family:"SF Mono","Cascadia Code",Consolas,ui-monospace,monospace;font-size:12.5px;}} .mcard .v{{font-size:14px;line-height:1.6;}}
.foot{{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;}}}}
</style>
<div class="wrap">
  <p class="eyebrow">Dual-Wall Shadow Sculpture · Monochrome Study</p>
  <h1>The same portrait, in black &amp; white</h1>
  <p class="lede">The Girl-with-a-Pearl-Earring front/back pair again — this time as <b>grayscale</b>.
  With no hue to place, the shadow is built almost entirely from the <b>black channel at varying
  strength</b>, and the result is a clean tonal portrait: silhouette, turban and light/dark masses all read.</p>

  <hr class="rule">
  <p class="h2">The reconstruction · drag to compare</p>
  <div class="walls">
    <div class="wall">
      <div class="wcap"><span class="name">Wall A — front</span><span class="sub">grayscale</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconA']}" alt="Reconstructed shadow, front"><img class="over" src="{img['srcA']}" alt="Source, front">
        <span class="lab l">Source</span><span class="lab r">Shadow</span><div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal front source vs shadow"></div></figure>
      <div class="chips"><span class="chip">SSIM <b>{b2['A']['ssim']:.2f}</b></span><span class="chip">RMSE <b>{b2['A']['rmse']:.3f}</b></span><span class="chip">Edge <b>{b2['A']['edge_fidelity']:.2f}</b></span><span class="chip k">double-duty <b>{b2['bg']['A']:.1f}%</b></span></div>
    </div>
    <div class="wall">
      <div class="wcap"><span class="name">Wall B — back</span><span class="sub">grayscale</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconB']}" alt="Reconstructed shadow, back"><img class="over" src="{img['srcB']}" alt="Source, back">
        <span class="lab l">Source</span><span class="lab r">Shadow</span><div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal back source vs shadow"></div></figure>
      <div class="chips"><span class="chip">SSIM <b>{b2['B']['ssim']:.2f}</b></span><span class="chip">RMSE <b>{b2['B']['rmse']:.3f}</b></span><span class="chip">Edge <b>{b2['B']['edge_fidelity']:.2f}</b></span><span class="chip k">double-duty <b>{b2['bg']['B']:.1f}%</b></span></div>
    </div>
  </div>
  <p class="hint">Tonally it reads well — arguably cleaner than the colour version, with no hue-mixing to muddy it.
  The face itself stays soft: black &amp; white doesn't change the shard-resolution limit.</p>

  <hr class="rule">
  <p class="h2">The finding</p>
  <div class="finding">
    <div class="bignum"><span class="n">{mean['bgA']:.1f}</span><span class="pm">%&nbsp;±&nbsp;{mean['bgA_sd']:.1f}</span>
      <div class="cap">honest colour-agreeing double duty, Wall&nbsp;A — mean over 5 seeds (up to {maxbgA:.1f}%)</div></div>
    <div>
      <p><b>Removing colour did not raise double duty — it slightly lowered it.</b> One might expect two
      grayscale images to be trivially compatible, but the credit rewards matching <em>value</em> at the
      linked points, and hue was actually <em>helping</em>: in colour the shared gold garment and blue
      turban agreed at those points.</p>
      <div class="cf">
        <span class="pill">colour <b>23.6%</b></span><span class="arrow">→</span>
        <span class="pill">black &amp; white <b>{mean['bgA']:.1f}%</b></span>
        <span style="color:var(--mut);font-size:12.5px;">same pair, same solver</span>
      </div>
    </div>
  </div>

  <hr class="rule">
  <p class="h2">Every seed · fidelity &amp; double duty</p>
  <div class="tblwrap"><table>
    <thead>
      <tr><th class="seed" style="text-align:left">seed</th><th class="grp" colspan="3">Wall A (front)</th><th class="grp sep" colspan="3">Wall B (back)</th><th class="grp sep" colspan="2">double duty</th></tr>
      <tr><th class="seed"></th><th>rmse</th><th>ssim</th><th>edge</th><th class="sep">rmse</th><th>ssim</th><th>edge</th><th class="sep">Wall A</th><th>Wall B</th></tr>
    </thead>
    <tbody>
{rows()}
    </tbody>
    <tfoot><tr><td class="seed">mean ± sd</td><td>—</td><td>{mean['ssimA']:.3f}</td><td>—</td><td class="sep">—</td><td>{mean['ssimB']:.3f}</td><td>—</td><td class="sep good">{mean['bgA']:.1f} ± {mean['bgA_sd']:.1f}%</td><td class="good">{mean['bgB']:.1f} ± {mean['bgB_sd']:.1f}%</td></tr></tfoot>
  </table></div>

  <hr class="rule">
  <p class="h2">Two honest notes</p>
  <div class="notes">
    <div class="note"><p class="k">It's a black-channel build</p>Grayscale reduces the CMYK lamination to the
    <b>K (black) sheet alone</b>, at varying strength. That's a far simpler object to fabricate — one material,
    no colour registration — but it <b>forfeits the coloured-shard idea</b> the project is built around. A choice, not a free win.</p>
    <div class="note"><p class="k">The face is still soft</p>Black &amp; white does nothing for the shard-resolution
    wall: recognizability lives in <b>small features</b>, and monochrome doesn't make them larger. The tonal
    portrait reads; the eyes and mouth don't resolve, exactly as at colour.</div>
  </div>

  <hr class="rule">
  <p class="h2">How it was made</p>
  <div class="method">
    <div class="mcard"><p class="k">Source pair</p><p class="v"><em>Girl with a Pearl Earring</em><br>front &amp; back, grayscale, bg removed</p></div>
    <div class="mcard"><p class="k">Assignment</p><p class="v mono">signed-damage<br>damage_weight 0.5<br>credit_weight 0.5</p></div>
    <div class="mcard"><p class="k">Colour-agreement credit</p><p class="v mono">match_tol 0.30<br>full-RGB (= value here)</p></div>
    <div class="mcard"><p class="k">Build</p><p class="v mono">{M['shards']} shards · {M['panels_used']}/14 panels<br>walls 1.8 × 1.8 m</p></div>
  </div>
  <div class="foot"><span>Interactive 3-D scene: <span class="mono">out_thickness_test/bw_final/scene.html</span></span><span>Multi-seed validated · seed {BEST} shown</span></div>
</div>
<script>
(function(){{document.querySelectorAll('.cmp').forEach(function(c){{var r=c.querySelector('input[type=range]');function set(v){{c.style.setProperty('--pos',v+'%');}}r.addEventListener('input',function(){{set(r.value);}});set(r.value);}});}})();
</script>
"""
p = os.path.join(OUT, "deliverable.html")
open(p, "w", encoding="utf-8").write(HTML); print("wrote", p, len(HTML), "chars")
