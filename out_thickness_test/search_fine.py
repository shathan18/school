"""FINE sweep: panel count x shard count (the two axes that matter). Flat portraits.
Fixed: angle(5,85), standoff 0.5, anchor(0.5,2.4), signed dw=0.5 cw=0.5 agree=0.90."""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

A_IMG="examples/Screenshot 2026-07-16 102515.png"; B_IMG="examples/Screenshot 2026-07-16 102536.png"
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets={"A":C.load_color_target(A_IMG,wr,white_thr=scene.white_threshold),
         "B":C.load_color_target(B_IMG,wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}

def cryptic(ts,table,r,panels,pT):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}
    carry=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05
        carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(carry)

def evaluate(pc,mult,seed=3):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=16,targets=targets,
                                 seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(mult)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,
       fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
       fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP)
    table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed,
        damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0)
    pT=C.stack_transmit_lut(names,sc,si); pred=r.render_color_np(pT)
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
    comp={w:100*(((1.0-pred[w].mean(-1))>0.05)&subject[w]).sum()/subject[w].sum() for w in ("A","B")}
    ssim=(acc["A"]["ssim"]+acc["B"]["ssim"])/2; edge=(acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    carry=cryptic(ts,table,r,panels,pT)
    ok=(comp["A"]>=85 and comp["B"]>=85 and nA+nB<=300)
    return dict(panels=pc,mult=mult,total=nA+nB,compA=comp["A"],compB=comp["B"],
                ssim=ssim,edge=edge,carry=carry,pass_=ok)

rows=[]
print(f"{'panels':>6s} {'mult':>4s} {'tot':>4s} {'compA/B':>9s} | {'SSIM':>5s} {'edge':>5s} | {'carry':>5s} | pass")
for pc in (10,12,14,16):
    for mult in (0.3,0.5,0.7):
        r=evaluate(pc,mult); rows.append(r)
        print(f"{pc:6d} {mult:4.1f} {r['total']:4d} {r['compA']:4.0f}/{r['compB']:<4.0f} | "
              f"{r['ssim']:.3f} {r['edge']:.3f} | {r['carry']:.3f} | {'OK' if r['pass_'] else 'FAIL'}")
json.dump(rows,open("out_thickness_test/search/fine.json","w"),indent=1)

# rank passing configs by objective: primary (ssim+edge), then carry, then shards
ok=[r for r in rows if r["pass_"]]
ok.sort(key=lambda r:(-(r["ssim"]+r["edge"]), r["carry"], r["total"]))
print("\nTOP passing configs by objective (primary recognizability, then cryptic, then shards):")
for r in ok[:5]:
    print(f"  panels={r['panels']} mult={r['mult']} tot={r['total']}: "
          f"SSIM+edge={r['ssim']+r['edge']:.3f} carry={r['carry']:.3f}")
print("\nsaved fine.json")
