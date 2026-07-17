"""STEP 2 COARSE joint sweep on OIL PAINTING (Vermeer girl, bg removed).
Axes: semantic_weight x shard_mult x panels. 1 seed. Corrected completeness gate.
Primary = FACE SSIM+edge. Confirm signed damage (dw0.5 cw0.5 agree0.90)."""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from semantic_lib import face_roi_from_target, face_metrics
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res; names=["clear"]+C.CMYK
targets={"A":C.load_color_target("examples/girl_front_nobg.png",wr,white_thr=scene.white_threshold),
         "B":C.load_color_target("examples/girl_back_nobg.png",wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
sem={};bbox={}
for w in ("A","B"): sem[w],bbox[w]=face_roi_from_target(targets[w])
def comp_blank(pred,w):
    de=np.sqrt(((pred[w]-targets[w])**2).sum(-1)); dw=np.sqrt(((1.0-targets[w])**2).sum(-1))
    return 100*((de<dw)&subject[w]).sum()/subject[w].sum()
def comp_match(pred,w,tol=0.30):
    d=np.sqrt(((pred[w]-targets[w])**2).sum(-1)); return 100*((d<tol)&subject[w]).sum()/subject[w].sum()
def evaluate(pc,mult,sw,seed=3):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=16,targets=targets,seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(mult)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=seed,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,
        semantic_masks=sem,semantic_weight=sw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si))
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
    cbA=comp_blank(pred,"A"); cbB=comp_blank(pred,"B"); cmA=comp_match(pred,"A"); cmB=comp_match(pred,"B")
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"]); fsB,feB=face_metrics(pred["B"],targets["B"],bbox["B"])
    face=(fsA+fsB+feA+feB)/2; glob=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    return dict(pc=pc,mult=mult,sw=sw,total=nA+nB,cbA=cbA,cbB=cbB,cmA=cmA,cmB=cmB,face=face,glob=glob,
                faceSSIM=(fsA+fsB)/2,gate=(cbA>=85 and cbB>=85 and nA+nB<=300))
rows=[]
print(f"{'pc':>2s} {'mul':>3s} {'sw':>3s} {'tot':>4s} {'blankA/B':>9s} {'matchA/B':>9s} | {'FACE':>5s} {'glob':>5s} | gate")
for pc in (12,16,20):
    for mult in (0.5,1.0,1.5):
        for sw in (0.0,0.5,1.0):
            r=evaluate(pc,mult,sw); rows.append(r)
            print(f"{pc:2d} {mult:3.1f} {sw:3.1f} {r['total']:4d} {r['cbA']:4.0f}/{r['cbB']:<4.0f} {r['cmA']:4.0f}/{r['cmB']:<4.0f} | {r['face']:.3f} {r['glob']:.3f} | {'OK' if r['gate'] else 'X'}")
json.dump(rows,open("out_thickness_test/search/oil_coarse.json","w"),indent=1)
ok=[r for r in rows if r["gate"]]; ok.sort(key=lambda r:-r["face"])
print("\nTOP by FACE recognizability (gate-passing):")
for r in ok[:6]: print(f"  pc={r['pc']} mult={r['mult']} sw={r['sw']} tot={r['total']}: FACE={r['face']:.3f} faceSSIM={r['faceSSIM']:.3f} glob={r['glob']:.3f} match={r['cmA']:.0f}/{r['cmB']:.0f}%")
# does semantic weight help the face? aggregate by sw
print("\nFACE score by semantic weight (mean over pc,mult):")
import statistics as st
for sw in (0.0,0.5,1.0):
    sub=[r["face"] for r in rows if r["sw"]==sw]; print(f"  sw={sw}: mean FACE={st.mean(sub):.3f}")
