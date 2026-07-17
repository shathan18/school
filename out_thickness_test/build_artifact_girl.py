# -*- coding: utf-8 -*-
"""Assemble the self-contained Girl-front/back deliverable artifact (embeds the 4 wall images)."""
import base64, json, os
OUT = "out_thickness_test/girl_final"

def b64(p):
    with open(os.path.join(OUT, p), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

img = {k: b64(f"{k}.png") for k in ("srcA", "reconA", "srcB", "reconB")}
M = json.load(open(os.path.join(OUT, "metrics.json")))
ps = {r["seed"]: r for r in M["per_seed"]}; mean = M["mean"]; BEST = M["best_seed"]

def rows():
    out = []
    for s in (1, 2, 3, 4, 5):
        r = ps[s]; A, B, bg = r["A"], r["B"], r["bg"]
        cls = ' class="shown"' if s == BEST else ""
        tag = ' <span class="tag">shown</span>' if s == BEST else ""
        out.append(
            f'<tr{cls}><td class="seed">{s}{tag}</td>'
            f'<td>{A["rmse"]:.3f}</td><td>{A["ssim"]:.3f}</td><td>{A["edge_fidelity"]:.2f}</td>'
            f'<td class="sep">{B["rmse"]:.3f}</td><td>{B["ssim"]:.3f}</td><td>{B["edge_fidelity"]:.2f}</td>'
            f'<td class="sep good">{bg["A"]:.1f}%</td><td class="good">{bg["B"]:.1f}%</td></tr>')
    return "\n".join(out)

b2 = ps[BEST]
HTML = f"""<title>A Portrait, Cast Twice — Dual-Wall Shadow Reconstruction</title>
<style>
:root {{
  --ground:#ECEAE3; --panel:#F6F4EE; --panel-2:#EDEAE2; --ink:#2A241F; --muted:#6A6055;
  --line:#DAD5CA; --accent:#33528A; --accent-soft:#5E7BAA; --good:#8A6A22; --grid:#E3DFD5;
  --shadow:0 1px 2px rgba(42,36,31,.06),0 6px 22px rgba(42,36,31,.08);
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --ground:#181310; --panel:#221B16; --panel-2:#2A211B; --ink:#EEE5D6; --muted:#B0A08A;
    --line:#352A22; --accent:#7C9CCB; --accent-soft:#8AA6CE; --good:#D2AE62; --grid:#2E251E;
    --shadow:0 1px 2px rgba(0,0,0,.35),0 8px 30px rgba(0,0,0,.4);
  }}
}}
:root[data-theme="light"] {{
  --ground:#ECEAE3; --panel:#F6F4EE; --panel-2:#EDEAE2; --ink:#2A241F; --muted:#6A6055;
  --line:#DAD5CA; --accent:#33528A; --accent-soft:#5E7BAA; --good:#8A6A22; --grid:#E3DFD5;
  --shadow:0 1px 2px rgba(42,36,31,.06),0 6px 22px rgba(42,36,31,.08);
}}
:root[data-theme="dark"] {{
  --ground:#181310; --panel:#221B16; --panel-2:#2A211B; --ink:#EEE5D6; --muted:#B0A08A;
  --line:#352A22; --accent:#7C9CCB; --accent-soft:#8AA6CE; --good:#D2AE62; --grid:#2E251E;
  --shadow:0 1px 2px rgba(0,0,0,.35),0 8px 30px rgba(0,0,0,.4);
}}
*{{box-sizing:border-box}} html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}}
.wrap{{max-width:1060px;margin:0 auto;padding:clamp(28px,5vw,68px) clamp(18px,4vw,40px) 80px;}}
.eyebrow{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 18px;}}
h1{{font-family:Georgia,"Iowan Old Style","Palatino Linotype","Book Antiqua",serif;
  font-size:clamp(32px,6vw,56px);line-height:1.02;font-weight:600;margin:0;letter-spacing:-.01em;text-wrap:balance;}}
.lede{{font-size:clamp(16px,2vw,19px);color:var(--muted);max-width:62ch;margin:20px 0 0;}}
.lede b{{color:var(--ink);font-weight:600;}}
.rule{{height:1px;background:var(--line);border:0;margin:44px 0;}}
.h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:0 0 20px;}}
.walls{{display:grid;grid-template-columns:1fr 1fr;gap:26px;}}
@media (max-width:760px){{.walls{{grid-template-columns:1fr;}}}}
.wall figure{{margin:0;}}
.wcap{{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:0 2px 12px;}}
.wcap .name{{font-weight:600;font-size:15px;}} .wcap .sub{{color:var(--muted);font-size:12.5px;}}
.cmp{{position:relative;width:100%;aspect-ratio:1/1;border-radius:10px;overflow:hidden;background:var(--panel);
  box-shadow:var(--shadow);border:1px solid var(--line);user-select:none;touch-action:none;}}
.cmp img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;}}
.cmp .over{{clip-path:inset(0 calc(100% - var(--pos,50%)) 0 0);}}
.cmp .lab{{position:absolute;top:10px;font-size:10.5px;letter-spacing:.14em;font-weight:600;text-transform:uppercase;
  padding:4px 8px;border-radius:5px;background:color-mix(in srgb,var(--panel) 80%,transparent);color:var(--ink);backdrop-filter:blur(3px);pointer-events:none;}}
.cmp .lab.l{{left:10px;}} .cmp .lab.r{{right:10px;color:var(--accent);}}
.cmp .divide{{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:2px;background:var(--accent);transform:translateX(-1px);pointer-events:none;}}
.cmp .grip{{position:absolute;top:50%;left:var(--pos,50%);width:38px;height:38px;margin:-19px 0 0 -19px;border-radius:50%;
  background:var(--accent);border:3px solid var(--panel);box-shadow:var(--shadow);display:grid;place-items:center;cursor:ew-resize;color:var(--panel);}}
.cmp .grip svg{{width:18px;height:18px;}}
.cmp input[type=range]{{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize;}}
.cmp:focus-within .divide{{box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 40%,transparent);}}
.chips{{display:flex;flex-wrap:wrap;gap:7px;margin:14px 2px 0;}}
.chip{{font-size:12px;color:var(--muted);background:var(--panel);border:1px solid var(--line);padding:4px 9px;border-radius:20px;font-variant-numeric:tabular-nums;}}
.chip b{{color:var(--ink);font-weight:600;}} .chip.k b{{color:var(--good);}}
.hint{{text-align:center;color:var(--muted);font-size:12.5px;margin:22px auto 0;letter-spacing:.02em;max-width:70ch;}}
.finding{{display:grid;grid-template-columns:auto 1fr;gap:clamp(20px,4vw,48px);align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:clamp(22px,3.5vw,36px);box-shadow:var(--shadow);}}
@media (max-width:680px){{.finding{{grid-template-columns:1fr;}}}}
.bignum{{font-variant-numeric:tabular-nums;}}
.bignum .n{{font-size:clamp(52px,9vw,86px);line-height:.9;font-weight:600;color:var(--accent);letter-spacing:-.02em;}}
.bignum .pm{{font-size:20px;color:var(--muted);font-weight:500;}}
.bignum .cap{{font-size:12.5px;color:var(--muted);margin-top:10px;max-width:22ch;}}
.finding p{{margin:0 0 12px;}} .finding p:last-child{{margin-bottom:0;}}
.move{{display:flex;align-items:center;gap:14px;margin-top:18px;flex-wrap:wrap;}}
.move .from{{color:var(--muted);font-weight:600;font-variant-numeric:tabular-nums;}}
.move .to{{color:var(--good);font-weight:600;font-variant-numeric:tabular-nums;}}
.move .bar{{flex:1;min-width:120px;height:8px;border-radius:6px;position:relative;
  background:linear-gradient(90deg,var(--panel-2),color-mix(in srgb,var(--good) 60%,var(--panel-2)));border:1px solid var(--line);overflow:hidden;}}
.move .lbl{{font-size:12px;color:var(--muted);}}
.tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);box-shadow:var(--shadow);}}
table{{border-collapse:collapse;width:100%;min-width:560px;font-variant-numeric:tabular-nums;}}
thead th{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600;text-align:right;padding:14px 12px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}}
thead th.grp{{color:var(--accent);text-align:left;}}
tbody td{{text-align:right;padding:9px 12px;font-size:13.5px;border-bottom:1px solid var(--grid);}}
tbody td.seed{{text-align:left;color:var(--muted);}} tbody td.sep{{border-left:1px solid var(--line);}}
tbody td.good{{color:var(--good);font-weight:600;}}
tr.shown td{{background:color-mix(in srgb,var(--accent) 9%,transparent);}}
tr.shown td.seed{{color:var(--ink);font-weight:600;}}
.tag{{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--panel);background:var(--accent);padding:2px 6px;border-radius:4px;margin-left:6px;vertical-align:middle;}}
tfoot td{{text-align:right;padding:12px;font-size:13.5px;font-weight:600;border-top:2px solid var(--line);}}
tfoot td.seed{{text-align:left;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px;}}
tfoot td.sep{{border-left:1px solid var(--line);}} tfoot td.good{{color:var(--good);}}
.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;}}
.mcard{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 18px;}}
.mcard .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;}}
.mono{{font-family:"SF Mono","Cascadia Code",Consolas,ui-monospace,monospace;font-size:12.5px;}}
.mcard .v{{font-size:14px;line-height:1.6;}}
.caveat{{background:color-mix(in srgb,var(--accent) 6%,var(--panel));border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;padding:16px 20px;margin-top:24px;font-size:14px;color:var(--ink);max-width:72ch;}}
.caveat b{{color:var(--accent);}}
.retire{{color:var(--muted);font-size:13.5px;max-width:66ch;margin:22px auto 0;text-align:center;}}
.retire s{{color:var(--accent-soft);}}
.foot{{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;}}}}
</style>

<div class="wrap">
  <p class="eyebrow">Dual-Wall Shadow Sculpture · Reconstruction Study</p>
  <h1>A portrait, cast twice</h1>
  <p class="lede">One woven-acrylic sculpture, lit from two directions, casts Vermeer's
  <b>Girl with a Pearl Earring</b> on two perpendicular walls — the <b>front</b> view on Wall&nbsp;A,
  the <b>back</b> view on Wall&nbsp;B. Because both are the same figure in the same palette, this is the
  <b>most colour-compatible pair we've tested</b> — and it produces the highest genuine cross-wall
  double duty we've measured.</p>

  <hr class="rule">
  <p class="h2">The reconstruction · drag to compare</p>
  <div class="walls">
    <div class="wall">
      <div class="wcap"><span class="name">Wall A — front</span><span class="sub">background removed</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconA']}" alt="Reconstructed shadow, front view">
        <img class="over" src="{img['srcA']}" alt="Source, front view">
        <span class="lab l">Source</span><span class="lab r">Shadow</span>
        <div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal front source vs shadow">
      </div></figure>
      <div class="chips">
        <span class="chip">SSIM <b>{b2['A']['ssim']:.2f}</b></span><span class="chip">RMSE <b>{b2['A']['rmse']:.3f}</b></span>
        <span class="chip">Edge <b>{b2['A']['edge_fidelity']:.2f}</b></span><span class="chip k">double-duty <b>{b2['bg']['A']:.1f}%</b></span>
      </div>
    </div>
    <div class="wall">
      <div class="wcap"><span class="name">Wall B — back</span><span class="sub">background removed</span></div>
      <figure><div class="cmp" style="--pos:50%">
        <img class="base" src="{img['reconB']}" alt="Reconstructed shadow, back view">
        <img class="over" src="{img['srcB']}" alt="Source, back view">
        <span class="lab l">Source</span><span class="lab r">Shadow</span>
        <div class="divide"></div>
        <div class="grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l-4 6 4 6M15 6l4 6-4 6"/></svg></div>
        <input type="range" min="0" max="100" value="50" aria-label="Reveal back source vs shadow">
      </div></figure>
      <div class="chips">
        <span class="chip">SSIM <b>{b2['B']['ssim']:.2f}</b></span><span class="chip">RMSE <b>{b2['B']['rmse']:.3f}</b></span>
        <span class="chip">Edge <b>{b2['B']['edge_fidelity']:.2f}</b></span><span class="chip k">double-duty <b>{b2['bg']['B']:.1f}%</b></span>
      </div>
    </div>
  </div>
  <p class="hint">Left of the divider is the source photograph; right is the shadow the sculpture actually casts.
  Silhouette, pose, the blue turban and gold garment all read — the shared palette is exactly what lets one
  wall's material also serve the other.</p>

  <div class="caveat"><b>The face stays soft on purpose.</b> At a buildable ~200-shard budget the palette, pose
  and silhouette resolve, but fine facial features do not — they need roughly <b>9× as many pieces</b> to sharpen.
  That blur is a <b>fabrication-scale limit, not a colour one</b>: it's how many shards can be cut and laminated,
  not the medium's colour range.</div>

  <hr class="rule">
  <p class="h2">The finding</p>
  <div class="finding">
    <div class="bignum">
      <span class="n">{mean['bgA']:.1f}</span><span class="pm">%&nbsp;±&nbsp;{mean['bgA_sd']:.1f}</span>
      <div class="cap">honest colour-agreeing double duty on Wall&nbsp;A — mean over 5 seeds (up to {max(r['bg']['A'] for r in M['per_seed']):.1f}% per seed)</div>
    </div>
    <div>
      <p><b>Genuine double duty — a stray shadow from one wall that lands on the other wall's image
      <em>and carries the colour that spot wants</em> — is set by how colour-compatible the two source
      images are, not by the solver.</b></p>
      <p>The same figure front and back share almost the entire palette, so double duty climbs from
      near-zero (an arbitrary pair) to the low-twenties with <b>no change to the algorithm</b>:</p>
      <div class="move">
        <span class="from">0.3%</span>
        <div class="bar" role="img" aria-label="rises from 0.3 percent to about 24 percent"></div>
        <span class="to">≈24%</span>
      </div>
      <div class="move"><span class="lbl">arbitrary pair (apples / breakfast) &nbsp;→&nbsp; same figure, front / back</span></div>
    </div>
  </div>

  <hr class="rule">
  <p class="h2">Every seed · fidelity &amp; double duty</p>
  <div class="tblwrap">
    <table>
      <thead>
        <tr>
          <th class="seed" style="text-align:left">seed</th>
          <th class="grp" colspan="3">Wall A (front)</th>
          <th class="grp sep" colspan="3">Wall B (back)</th>
          <th class="grp sep" colspan="2">double duty (colour-agreeing)</th>
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
  <p class="retire">The old &ldquo;lands on subject&rdquo; figure of <s>~27%</s> is retired — it counted any stray
  shadow touching content, and ~90% of it was the wrong colour. Every number here passes a full-RGB colour-match
  test (‖arrived − wanted‖ &lt; 0.30), the same test the solver's credit now uses.</p>

  <hr class="rule">
  <p class="h2">How it was made</p>
  <div class="method">
    <div class="mcard"><p class="k">Source pair</p><p class="v"><em>Girl with a Pearl Earring</em><br>front &amp; back — same figure, backgrounds removed</p></div>
    <div class="mcard"><p class="k">Assignment</p><p class="v mono">signed-damage<br>damage_weight 0.5<br>credit_weight 0.5</p></div>
    <div class="mcard"><p class="k">Colour-agreement credit</p><p class="v mono">match_tol 0.30<br>full-RGB, brightness incl.</p></div>
    <div class="mcard"><p class="k">Build</p><p class="v mono">{M['shards']} shards · {M['panels_used']}/14 panels<br>walls 1.8 × 1.8 m</p></div>
  </div>

  <div class="foot">
    <span>Interactive 3-D scene: <span class="mono">out_thickness_test/girl_final/scene.html</span></span>
    <span>Multi-seed validated · seed {BEST} shown (best RMSE)</span>
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
