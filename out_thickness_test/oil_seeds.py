"""Multi-seed: is the tiny semantic-weight face improvement real or noise? Control vs semantic
at matched pc/mult. 6 seeds. Also cryptic metric (tertiary). Oil painting (Vermeer girl)."""
import sys, dataclasses, json, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
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
def cryptic(ts,table,r,panels,pT):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; carry=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05; carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(carry)
def evaluate(pc,mult,sw,seed):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=16,targets=targets,seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(mult)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=seed,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,semantic_masks=sem,semantic_weight=sw)
    pT=C.stack_transmit_lut(names,sc,si); pred=r.render_color_np(pT)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"]); fsB,feB=face_metrics(pred["B"],targets["B"],bbox["B"])
    face=(fsA+fsB+feA+feB)/2
    acc=_metrics.evaluate_wall_accuracy(targets,pred); glob=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    return face,(fsA+fsB)/2,glob,min(comp_blank(pred,"A"),comp_blank(pred,"B")),cryptic(ts,table,r,panels,pT),bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
CANDS=[(12,0.5,0.0),(12,0.5,0.5),(12,0.5,1.0),(16,0.5,0.5),(20,0.5,0.0)]
SEEDS=[1,2,3,4,5,6]; R={}
for pc,mult,sw in CANDS:
    F=[];FS=[];G=[];CP=[];CY=[];T=[]
    print(f"\n--- pc={pc} mult={mult} sw={sw} ---")
    for s in SEEDS:
        face,fssim,glob,mc,car,tot=evaluate(pc,mult,sw,s); F.append(face);FS.append(fssim);G.append(glob);CP.append(mc);CY.append(car);T.append(tot)
        print(f"  seed {s}: FACE={face:.3f} faceSSIM={fssim:.3f} glob={glob:.3f} comp={mc:.0f}% carry={car:.3f} shards={tot} {'' if (mc>=85 and tot<=300) else '<-FAIL'}")
    passed=sum(1 for c,t in zip(CP,T) if c>=85 and t<=300)
    R[f"{pc}_{mult}_{sw}"]=dict(face=F,faceSSIM=FS,glob=G,comp=CP,carry=CY,total=T,passed=passed)
    print(f"  gate {passed}/6 | FACE {st.mean(F):.3f}±{st.pstdev(F):.3f} | faceSSIM {st.mean(FS):.3f}±{st.pstdev(FS):.3f} | carry {st.mean(CY):.3f}±{st.pstdev(CY):.3f}")
json.dump(R,open("out_thickness_test/search/oil_seeds.json","w"),indent=1)
print("\n=== is semantic real? control (12,0.5,0.0) vs semantic (12,0.5,0.5) ===")
c=R["12_0.5_0.0"]; s=R["12_0.5_0.5"]
print(f"  control FACE {st.mean(c['face']):.3f}±{st.pstdev(c['face']):.3f}  vs  semantic FACE {st.mean(s['face']):.3f}±{st.pstdev(s['face']):.3f}")
print(f"  difference {st.mean(s['face'])-st.mean(c['face']):+.3f}  (pooled stdev ~{(st.pstdev(c['face'])+st.pstdev(s['face']))/2:.3f}) -> {'REAL' if abs(st.mean(s['face'])-st.mean(c['face']))>(st.pstdev(c['face'])+st.pstdev(s['face']))/2 else 'NOISE (within stdev)'}")
