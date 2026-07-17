"""DECISIVE: uniform vs face-concentrated at MATCHED ~217 shards, 6 seeds. Is the face gain real?
Report faceSSIM/edge mean±stdev + corrected colour-match completeness (gate)."""
import sys, dataclasses, statistics as st
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
face={};bbox={}
for w in ("A","B"): face[w],bbox[w]=face_roi_from_target(targets[w])
face_bin={w:(face[w]>0.4) for w in ("A","B")}
def match30(pred,w):
    d=np.sqrt(((pred[w]-targets[w])**2).sum(-1)); return 100*((d<0.30)&subject[w]).sum()/subject[w].sum()
def evaluate(fragsize,fd,bgc,seed):
    panels,_=build_panels_greedy(scene,count=16,mode="deliberate",K=16,targets=targets,seed=seed,angle_deg_range=(5,85))
    SP=dataclasses.replace(scene.solve,fragment_size=fragsize,fragment_min_area=scene.solve.fragment_min_area*(fragsize/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(fragsize/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    kw=dict(face_masks=face_bin,face_density=fd,bg_coarsen=bgc) if fd>1 else {}
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=seed,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,**kw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si)); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"]); fsB,feB=face_metrics(pred["B"],targets["B"],bbox["B"])
    return (fsA+fsB)/2,(feA+feB)/2,fsA,min(match30(pred,"A"),match30(pred,"B")),tot
ARMS=[("UNIFORM ~217",0.133,1.0,1.0),("CONCENTRATED face40/bg12",0.26,40.0,12.0)]
SEEDS=[1,2,3,4,5,6]; R={}
for lbl,fs,fd,bgc in ARMS:
    FS=[];FE=[];FSA=[];CM=[];T=[]
    print(f"\n--- {lbl} ---")
    for s in SEEDS:
        fssim,fedge,fsa,cm,tot=evaluate(fs,fd,bgc,s); FS.append(fssim);FE.append(fedge);FSA.append(fsa);CM.append(cm);T.append(tot)
        print(f"  seed {s}: faceSSIM={fssim:.3f} faceEdge={fedge:.3f} (wallA face {fsa:.3f}) colour-match={cm:.0f}% shards={tot}")
    R[lbl]=dict(fs=FS,fe=FE,fsa=FSA,cm=CM,t=T)
    print(f"  MEAN faceSSIM {st.mean(FS):.3f}±{st.pstdev(FS):.3f} | faceEdge {st.mean(FE):.3f}±{st.pstdev(FE):.3f} | wallA-face {st.mean(FSA):.3f}±{st.pstdev(FSA):.3f} | match {st.mean(CM):.0f}% | shards~{int(st.mean(T))}")
u=R["UNIFORM ~217"]; c=R["CONCENTRATED face40/bg12"]
dS=st.mean(c["fs"])-st.mean(u["fs"]); dE=st.mean(c["fe"])-st.mean(u["fe"]); dA=st.mean(c["fsa"])-st.mean(u["fsa"])
pooled=lambda a,b:(st.pstdev(a)+st.pstdev(b))/2
print("\n=== IS FACE CONCENTRATION REAL? (matched ~217 shards) ===")
print(f"  faceSSIM  Δ={dS:+.3f} (pooled sd {pooled(u['fs'],c['fs']):.3f}) -> {'REAL' if abs(dS)>pooled(u['fs'],c['fs']) else 'noise'}")
print(f"  faceEdge  Δ={dE:+.3f} (pooled sd {pooled(u['fe'],c['fe']):.3f}) -> {'REAL' if abs(dE)>pooled(u['fe'],c['fe']) else 'noise'}")
print(f"  wallA-face Δ={dA:+.3f} (pooled sd {pooled(u['fsa'],c['fsa']):.3f}) -> {'REAL' if abs(dA)>pooled(u['fsa'],c['fsa']) else 'noise'}")
