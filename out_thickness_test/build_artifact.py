# -*- coding: utf-8 -*-
"""Assemble the self-contained Monet deliverable artifact (embeds the 4 wall images)."""
import base64, json, os
OUT = "out_thickness_test/monet_final"

def b64(p):
    with open(os.path.join(OUT, p), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

img = {k: b64(f"{k}.png") for k in ("srcA", "reconA", "srcB", "reconB")}
M = json.load(open(os.path.join(OUT, "metrics.json")))
ps = {r["seed"]: r for r in M["per_seed"]}
mean = M["mean"]

def rows():
    out = []
    for s in (1, 2, 3, 4, 5):
        r = ps[s]; A, B, bg = r["A"], r["B"], r["bg"]
        cls = ' class="shown"' if s == 3 else ""
        tag = ' <span class="tag">shown</span>' if s == 3 else ""
        out.append(
            f'<tr{cls}><td class="seed">{s}{tag}</td>'
            f'<td>{A["rmse"]:.3f}</td><td>{A["ssim"]:.3f}</td><td>{A["edge_fidelity"]:.2f}</td>'
            f'<td class="sep">{B["rmse"]:.3f}</td><td>{B["ssim"]:.3f}</td><td>{B["edge_fidelity"]:.2f}</td>'
            f'<td class="sep good">{bg["A"]:.1f}%</td><td class="good">{bg["B"]:.1f}%</td></tr>')
    return "\n".join(out)

HTML = f"""<title>Two Lights, Two Monets — Dual-Wall Shadow Reconstruction</title>
<style>
:root {{
  --ground:#E7E8E6; --panel:#F3F3F1; --panel-2:#ECEDEA; --ink:#242A2D; --muted:#5E696D;
  --line:#D3D6D4; --accent:#38606F; --accent-soft:#6C919E; --good:#3F7355; --grid:#E0E2DF;
  --shadow:0 1px 2px rgba(30,40,44,.06),0 6px 22px rgba(30,40,44,.07);
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --ground:#1D1618; --panel:#271F1E; --panel-2:#2E2523; --ink:#F0E6D8; --muted:#B29B87;
    --line:#3A2E2B; --accent:#DA9C50; --accent-soft:#C4703A; --good:#A6BC83; --grid:#332A27;
    --shadow:0 1px 2px rgba(0,0,0,.30),0 8px 30px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="light"] {{
  --ground:#E7E8E6; --panel:#F3F3F1; --panel-2:#ECEDEA; --ink:#242A2D; --muted:#5E696D;
  --line:#D3D6D4; --accent:#38606F; --accent-soft:#6C919E; --good:#3F7355; --grid:#E0E2DF;
  --shadow:0 1px 2px rgba(30,40,44,.06),0 6px 22px rgba(30,40,44,.07);
}}
:root[data-theme="dark"] {{
  --ground:#1D1618; --panel:#271F1E; --panel-2:#2E2523; --ink:#F0E6D8; --muted:#B29B87;
  --line:#3A2E2B; --accent:#DA9C50; --accent-soft:#C4703A; --good:#A6BC83; --grid:#332A27;
  --shadow:0 1px 2px rgba(0,0,0,.30),0 8px 30px rgba(0,0,0,.35);
}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}}
.serif{{font-family:Georgia,"Iowan Old Style","Palatino Linotype","Book Antiqua",serif;}}
.wrap{{max-width:1060px;margin:0 auto;padding:clamp(28px,5vw,68px) clamp(18px,4vw,40px) 80px;}}
.eyebrow{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);
  font-weight:600;margin:0 0 18px;}}
h1.serif{{font-size:clamp(32px,6vw,56px);line-height:1.02;font-weight:600;margin:0;
  letter-spacing:-.01em;text-wrap:balance;}}
.lede{{font-size:clamp(16px,2vw,19px);color:var(--muted);max-width:60ch;margin:20px 0 0;}}
.lede b{{color:var(--ink);font-weight:600;}}
.rule{{height:1px;background:var(--line);border:0;margin:44px 0;}}
.h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0 0 20px;}}

/* ---- comparison sliders ---- */
.walls{{display:grid;grid-template-columns:1fr 1fr;gap:26px;}}
@media (max-width:760px){{.walls{{grid-template-columns:1fr;}}}}
.wall figure{{margin:0;}}
.wcap{{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:0 2px 12px;}}
.wcap .name{{font-weight:600;font-size:15px;}}
.wcap .sub{{color:var(--muted);font-size:12.5px;letter-spacing:.02em;}}
.cmp{{position:relative;width:100%;aspect-ratio:1/1;border-radius:10px;overflow:hidden;
  background:var(--panel);box-shadow:var(--shadow);border:1px solid var(--line);
  user-select:none;touch-action:none;}}
.cmp img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;
  image-rendering:auto;pointer-events:none;}}
.cmp .over{{clip-path:inset(0 calc(100% - var(--pos,50%)) 0 0);}}
.cmp .lab{{position:absolute;top:10px;font-size:10.5px;letter-spacing:.14em;font-weight:600;
  text-transform:uppercase;padding:4px 8px;border-radius:5px;
  background:color-mix(in srgb,var(--panel) 78%,transparent);color:var(--ink);
  backdrop-filter:blur(3px);pointer-events:none;}}
.cmp .lab.l{{left:10px;}} .cmp .lab.r{{right:10px;color:var(--accent);}}
.cmp .divide{{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:2px;
  background:var(--accent);transform:translateX(-1px);pointer-events:none;box-shadow:0 0 0 1px rgba(0,0,0,.08);}}
.cmp .grip{{position:absolute;top:50%;left:var(--pos,50%);width:38px;height:38px;margin:-19px 0 0 -19px;
  border-radius:50%;background:var(--accent);border:3px solid var(--panel);
  box-shadow:var(--shadow);display:grid;place-items:center;cursor:ew-resize;color:var(--panel);}}
.cmp .grip svg{{width:18px;height:18px;}}
.cmp input[type=range]{{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize;}}
.cmp:focus-within .divide{{box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 40%,transparent);}}
.chips{{display:flex;flex-wrap:wrap;gap:7px;margin:14px 2px 0;}}
.chip{{font-size:12px;color:var(--muted);background:var(--panel);border:1px solid var(--line);
  padding:4px 9px;border-radius:20px;font-variant-numeric:tabular-nums;}}
.chip b{{color:var(--ink);font-weight:600;}}
.chip.k b{{color:var(--good);}}
.hint{{text-align:center;color:var(--muted);font-size:12.5px;margin:22px 0 0;letter-spacing:.02em;}}

/* ---- finding ---- */
.finding{{display:grid;grid-template-columns:auto 1fr;gap:clamp(20px,4vw,48px);align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:clamp(22px,3.5vw,36px);box-shadow:var(--shadow);}}
@media (max-width:680px){{.finding{{grid-template-columns:1fr;}}}}
.bignum{{font-variant-numeric:tabular-nums;}}
.bignum .n{{font-size:clamp(52px,9vw,86px);line-height:.9;font-weight:600;color:var(--accent);
  letter-spacing:-.02em;}}
.bignum .pm{{font-size:20px;color:var(--muted);font-weight:500;}}
.bignum .cap{{font-size:12.5px;color:var(--muted);margin-top:10px;max-width:20ch;}}
.finding p{{margin:0 0 12px;}}
.finding p:last-child{{margin-bottom:0;}}
.move{{display:flex;align-items:center;gap:14px;margin-top:18px;flex-wrap:wrap;}}
.move .from,.move .to{{font-variant-numeric:tabular-nums;font-weight:600;}}
.move .from{{color:var(--muted);}} .move .to{{color:var(--good);}}
.move .bar{{flex:1;min-width:120px;height:8px;border-radius:6px;position:relative;
  background:linear-gradient(90deg,var(--panel-2),color-mix(in srgb,var(--good) 55%,var(--panel-2)));
  border:1px solid var(--line);overflow:hidden;}}
.move .bar::after{{content:"";position:absolute;left:1.5%;top:0;bottom:0;width:2px;background:var(--muted);opacity:.6;}}
.move .lbl{{font-size:12px;color:var(--muted);letter-spacing:.02em;}}

/* ---- table ---- */
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);box-shadow:var(--shadow);}}
table{{border-collapse:collapse;width:100%;min-width:560px;font-variant-numeric:tabular-nums;}}
thead th{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  font-weight:600;text-align:right;padding:14px 12px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}}
thead th.grp{{color:var(--accent);text-align:left;}}
tbody td{{text-align:right;padding:9px 12px;font-size:13.5px;border-bottom:1px solid var(--grid);}}
tbody td.seed{{text-align:left;color:var(--muted);}}
tbody td.sep{{border-left:1px solid var(--line);}}
tbody td.good{{color:var(--good);font-weight:600;}}
tr.shown td{{background:color-mix(in srgb,var(--accent) 8%,transparent);}}
tr.shown td.seed{{color:var(--ink);font-weight:600;}}
.tag{{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--panel);
  background:var(--accent);padding:2px 6px;border-radius:4px;margin-left:6px;vertical-align:middle;}}
tfoot td{{text-align:right;padding:12px;font-size:13.5px;font-weight:600;border-top:2px solid var(--line);}}
tfoot td.seed{{text-align:left;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px;}}
tfoot td.sep{{border-left:1px solid var(--line);}}
tfoot td.good{{color:var(--good);}}

/* ---- method ---- */
.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;margin-top:8px;}}
.mcard{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 18px;}}
.mcard .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;}}
.mono{{font-family:"SF Mono","Cascadia Code",Consolas,ui-monospace,monospace;font-size:12.5px;}}
.mcard .v{{font-size:14px;line-height:1.6;}}
.retire{{color:var(--muted);font-size:13.5px;max-width:64ch;margin:22px auto 0;text-align:center;}}
.retire s{{color:var(--accent-soft);text-decoration-thickness:1px;}}
.foot{{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px;
  display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;}}
@media (prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important;transition:none!important;}}}}
</style>

<div class="wrap">
  <p class="eyebrow">Dual-Wall Shadow Sculpture · Reconstruction Study</p>
  <h1 class="serif">Two lights, two Monets</h1>
  <p class="lede">One woven-acrylic sculpture, lit from two directions, casts a different painting on
  each perpendicular wall — Monet's <b>San Giorgio Maggiore</b> at morning on Wall&nbsp;A, at dusk on
  Wall&nbsp;B. This is the current pipeline's full-quality output on the pair the data says works —
  and the reason it works is <b>the pair, not the algorithm</b>.</p>

  <hr class="rule">
  <p class="h2">The reconstruction · drag to compare</p>
  <div class="walls">
    <div class="wall">
      <div class="wcap"><span class="name">Wall A — morning</span><span class="sub">cool palette</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconA']}" alt="Reconstructed shadow, Wall A">
        <img class="over" src="{img['srcA']}" alt="Source painting, Wall A">
        <span class="lab l">Source</span><span class="lab r">Shadow</span>
        <div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal Wall A source vs shadow">
      </div></figure>
      <div class="chips">
        <span class="chip">SSIM <b>0.75</b></span><span class="chip">RMSE <b>0.148</b></span>
        <span class="chip">Edge <b>0.34</b></span><span class="chip k">double-duty <b>19.6%</b></span>
      </div>
    </div>
    <div class="wall">
      <div class="wcap"><span class="name">Wall B — dusk</span><span class="sub">warm palette</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconB']}" alt="Reconstructed shadow, Wall B">
        <img class="over" src="{img['srcB']}" alt="Source painting, Wall B">
        <span class="lab l">Source</span><span class="lab r">Shadow</span>
        <div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal Wall B source vs shadow">
      </div></figure>
      <div class="chips">
        <span class="chip">SSIM <b>0.69</b></span><span class="chip">RMSE <b>0.189</b></span>
        <span class="chip">Edge <b>0.36</b></span><span class="chip k">double-duty <b>5.1%</b></span>
      </div>
    </div>
  </div>
  <p class="hint">Left of the divider is Monet's painting; right is the shadow the sculpture actually casts.
  The two reconstructions share warm and cool tones — that shared palette is what makes real double-duty possible.</p>

  <hr class="rule">
  <p class="h2">The finding</p>
  <div class="finding">
    <div class="bignum">
      <span class="n">16.3</span><span class="pm">%&nbsp;±&nbsp;4.0</span>
      <div class="cap">honest colour-agreeing double-duty on Wall&nbsp;A — mean over 5 seeds (19.6% on the shown seed)</div>
    </div>
    <div>
      <p><b>Genuine double-duty — a stray shadow from one wall that lands on the other wall's image
      <em>and carries the colour that spot wants</em> — is bounded by how compatible the two source
      images are, not by the solver.</b></p>
      <p>Swapping an arbitrary pair for a colour-compatible one lifts it from near-zero to ~16–19%
      with <b>no change to the algorithm</b>:</p>
      <div class="move">
        <span class="from">0.3%</span>
        <div class="bar" role="img" aria-label="rises from 0.3 percent to about 19 percent"></div>
        <span class="to">≈19%</span>
      </div>
      <div class="move"><span class="lbl">arbitrary pair (apples / breakfast) &nbsp;→&nbsp; compatible pair (Monet day / dusk)</span></div>
    </div>
  </div>

  <hr class="rule">
  <p class="h2">Every seed · fidelity &amp; double-duty</p>
  <div class="tblwrap">
    <table>
      <thead>
        <tr>
          <th class="seed" style="text-align:left">seed</th>
          <th class="grp" colspan="3">Wall A (morning)</th>
          <th class="grp sep" colspan="3">Wall B (dusk)</th>
          <th class="grp sep" colspan="2">double-duty (colour-agreeing)</th>
        </tr>
        <tr>
          <th class="seed"></th>
          <th>rmse</th><th>ssim</th><th>edge</th>
          <th class="sep">rmse</th><th>ssim</th><th>edge</th>
          <th class="sep">Wall A</th><th>Wall B</th>
        </tr>
      </thead>
      <tbody>
{rows()}
      </tbody>
      <tfoot>
        <tr>
          <td class="seed">mean ± sd</td>
          <td>—</td><td>{mean['ssimA']:.3f}</td><td>—</td>
          <td class="sep">—</td><td>{mean['ssimB']:.3f}</td><td>—</td>
          <td class="sep good">{mean['bgA']:.1f} ± {mean['bgA_sd']:.1f}%</td>
          <td class="good">{mean['bgB']:.1f} ± {mean['bgB_sd']:.1f}%</td>
        </tr>
      </tfoot>
    </table>
  </div>
  <p class="retire">The old &ldquo;lands on subject&rdquo; figure of <s>~27%</s> is retired — it counted any
  stray shadow touching content, and ~90% of it was the wrong colour. Every number here passes a
  full-RGB colour-match test (‖arrived − wanted‖ &lt; 0.30), the same test the solver's credit now uses.</p>

  <hr class="rule">
  <p class="h2">How it was made</p>
  <div class="method">
    <div class="mcard"><p class="k">Source pair</p><p class="v">Monet, <em>San Giorgio Maggiore</em><br>morning &amp; dusk — same scene, opposite palettes</p></div>
    <div class="mcard"><p class="k">Assignment</p><p class="v mono">signed-damage<br>damage_weight 0.5<br>credit_weight 0.5</p></div>
    <div class="mcard"><p class="k">Colour-agreement credit</p><p class="v mono">match_tol 0.30<br>full-RGB, brightness incl.</p></div>
    <div class="mcard"><p class="k">Build</p><p class="v mono">274 shards · 14/14 panels<br>walls 1.8 × 1.8 m</p></div>
  </div>

  <div class="foot">
    <span>Interactive 3-D scene (panels, rays, both walls): <span class="mono">out_thickness_test/monet_final/scene.html</span></span>
    <span>Multi-seed validated · seed 3 shown (best RMSE)</span>
  </div>
</div>

<script>
(function () {{
  document.querySelectorAll('.cmp').forEach(function (c) {{
    var r = c.querySelector('input[type=range]');
    function set(v) {{ c.style.setProperty('--pos', v + '%'); }}
    r.addEventListener('input', function () {{ set(r.value); }});
    set(r.value);
  }});
}})();
</script>
"""

path = os.path.join(OUT, "deliverable.html")
with open(path, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", path, "(", len(HTML), "chars )")
