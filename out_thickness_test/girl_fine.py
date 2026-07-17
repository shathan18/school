"""RESTART on background-removed girls (subject = girl only, black bg ignored).
Base fixed: signed dw0.5 cw0.5 agree0.90. Baseline + fine sweep (panels x mult, K16)."""
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
A_IMG="examples/girl_front_nobg.png"; B_IMG="examples/girl_back_nobg.png"
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets={"A":C.load_color_target(A_IMG,wr,white_thr=scene.white_threshold),
         "B":C.load_color_target(B_IMG,wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
for w in ("A","B"): print(f"wall {w}: subject={subject[w].mean()*100:.0f}% (should be ~girl, not bg)")
def cryptic(ts,table,r,panels,pT):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; carry=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05
        carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(carry)
def evaluate(pc,mult,K,seed=3):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=K,targets=targets,seed=seed,angle_deg_range=(5,85))
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
    return dict(panels=pc,mult=mult,total=nA+nB,cA=comp["A"],cB=comp["B"],ssim=ssim,edge=edge,
                carry=cryptic(ts,table,r,panels,pT),pass_=(comp["A"]>=85 and comp["B"]>=85 and nA+nB<=300))
print("\n=== BASELINE 8p/mult1.0/K16 ===")
b=evaluate(8,1.0,16)
print(f"  tot={b['total']} comp {b['cA']:.0f}/{b['cB']:.0f}% SSIM {b['ssim']:.3f} edge {b['edge']:.3f} carry {b['carry']:.3f} pass={b['pass_']}")
rows=[]
print("\n=== FINE (panels x mult, K16, 1 seed) ===")
print(f"{'pc':>3s} {'mult':>4s} {'tot':>4s} {'cA/cB':>7s} | {'SSIM':>5s} {'edge':>5s} | {'carry':>5s} | pass")
for pc in (12,16,20):
    for mult in (0.3,0.5,0.7,1.0):
        r=evaluate(pc,mult,16); rows.append(r)
        print(f"{pc:3d} {mult:4.1f} {r['total']:4d} {r['cA']:3.0f}/{r['cB']:<3.0f} | {r['ssim']:.3f} {r['edge']:.3f} | {r['carry']:.3f} | {'OK' if r['pass_'] else 'FAIL'}")
json.dump(rows,open("out_thickness_test/search/girl_fine.json","w"),indent=1)
ok=[r for r in rows if r["pass_"]]; ok.sort(key=lambda r:(-(r["ssim"]+r["edge"]),r["carry"],r["total"]))
print("\nTOP passing:")
for r in ok[:5]: print(f"  {r['panels']}p mult={r['mult']} tot={r['total']}: prim={r['ssim']+r['edge']:.3f} carry={r['carry']:.3f} SSIM={r['ssim']:.3f}")
