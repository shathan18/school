"""OPEN-ENDED SEARCH on the FLAT PORTRAITS (primary pair). Base config held fixed:
signed-damage, damage_weight=0.5, credit_weight=0.5, agree_min=0.90.
Phase: baseline reading + COARSE one-axis-at-a-time pass. HARD CAP: total shards <= 300."""
import sys, dataclasses, json, math
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

A_IMG="examples/Screenshot 2026-07-16 102515.png"   # Mona Lisa (flat)
B_IMG="examples/Screenshot 2026-07-16 102536.png"   # Pearl Earring (flat)
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets={"A":C.load_color_target(A_IMG,wr,white_thr=scene.white_threshold),
         "B":C.load_color_target(B_IMG,wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}

def cryptic(ts,table,renderer,panels,pT):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}
    carry=[]; soloedge=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]
        pt=np.ones_like(pT); pt[gi]=pT[gi]
        solo=renderer.render_color_np(pt)[w]
        d=(1.0-solo.mean(-1))>0.05
        carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
        soloedge.append(_metrics.edge_fidelity(solo,targets[w]))
    return max(carry), max(soloedge)

def evaluate(cfg, seed=3):
    ab=dict(angle_deg_range=cfg.get("angle",(5,85)))
    if cfg.get("angle_bands"): ab=dict(angle_bands=cfg["angle_bands"])
    panels,_=build_panels_greedy(scene,count=cfg["panels"],mode="deliberate",K=16,targets=targets,
                                 seed=seed,standoff=cfg.get("standoff",0.5),
                                 anchor_range=cfg.get("anchor",(0.5,2.4)),**ab)
    FS=0.135/np.sqrt(cfg["mult"])
    SP=dataclasses.replace(scene.solve,fragment_size=FS,
       fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
       fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP)
    table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed,
        damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0)
    pT=C.stack_transmit_lut(names,sc,si)
    pred=r.render_color_np(pT)
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
    comp={w:100*(((1.0-pred[w].mean(-1))>0.05)&subject[w]).sum()/subject[w].sum() for w in ("A","B")}
    mc,me=cryptic(ts,table,r,panels,pT)
    ssim=(acc["A"]["ssim"]+acc["B"]["ssim"])/2; edge=(acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    ok = (comp["A"]>=85 and comp["B"]>=85 and nA+nB<=300)
    return {"total":nA+nB,"A":nA,"B":nB,"compA":comp["A"],"compB":comp["B"],
            "ssim":ssim,"edge":edge,"A_ssim":acc["A"]["ssim"],"B_ssim":acc["B"]["ssim"],
            "A_edge":acc["A"]["edge_fidelity"],"B_edge":acc["B"]["edge_fidelity"],
            "carry":mc,"soloedge":me,"pass":ok}

BASE=dict(panels=8,angle=(5,85),standoff=0.5,anchor=(0.5,2.4),mult=1.0)
print("=== BASELINE (current-best-style) on FLAT PORTRAITS ===")
b=evaluate(BASE)
print(f"  total={b['total']} (A{b['A']}/B{b['B']}) comp {b['compA']:.0f}/{b['compB']:.0f}% | "
      f"SSIM {b['ssim']:.3f} edge {b['edge']:.3f} | CRYPTIC carry={b['carry']:.3f} soloedge={b['soloedge']:.3f} | pass={b['pass']}")

# COARSE: one axis at a time. mult kept so total<=300.
configs=[("BASE",BASE)]
for mu in (0.3,0.5,0.7):        configs.append((f"shards mult={mu}", {**BASE,"mult":mu}))
for pc in (6,10,12,14):         configs.append((f"panels={pc}",      {**BASE,"panels":pc,"mult":0.7}))
for ang in [(20,70),(35,55)]:   configs.append((f"angle={ang}",      {**BASE,"angle":ang,"mult":0.7}))
configs.append(("angle near-axis",{**BASE,"angle_bands":[(0,15),(75,90)],"mult":0.7}))
for so in (0.35,0.8):           configs.append((f"standoff={so}",    {**BASE,"standoff":so,"mult":0.7}))
configs.append(("anchor narrow",{**BASE,"anchor":(1.0,2.0),"mult":0.7}))

rows=[]
print("\n=== COARSE PASS (1 seed) ===")
print(f"{'config':20s} | {'tot':>4s} {'compA/B':>9s} | {'SSIM':>5s} {'edge':>5s} | {'carry':>5s} {'soloE':>5s} | pass")
for name,cfg in configs:
    try:
        r=evaluate(cfg); r["name"]=name; r["cfg"]=cfg; rows.append(r)
        print(f"{name:20s} | {r['total']:4d} {r['compA']:4.0f}/{r['compB']:<4.0f} | "
              f"{r['ssim']:.3f} {r['edge']:.3f} | {r['carry']:.3f} {r['soloedge']:.3f} | {'OK' if r['pass'] else 'FAIL'}")
    except Exception as e:
        print(f"{name:20s} | ERROR {e}")
json.dump(rows,open("out_thickness_test/search/coarse.json","w"),indent=1,default=str)
print("\nsaved coarse.json")
