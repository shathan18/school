"""REAL-pipeline density sweep (cross-talk PRESENT, budget DISABLED so count crosses ~220).
Fixed base layout: wide-angle (5-85 deg) seed 3, scorer unchanged. Only density varies."""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

scene = load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets = {"A": C.load_color_target("examples/apples.jpg", wr, white_thr=scene.white_threshold),
           "B": C.load_color_target("examples/breakfast.jpg", wr, white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
CEILING=220

# fixed base layout: wide/seed3 (density does NOT affect layout)
panels,_=build_panels_greedy(scene,count=14,mode="deliberate",K=16,targets=targets,
                             seed=3, angle_deg_range=(5,85))
prim_static=None

DENS=[("0.25x",0.25),("0.5x",0.5),("1x",1.0),("2x",2.0),("3x",3.0),("4x",4.0)]
base_fs=0.135
results=[]; imgs={}
for lbl,mult in DENS:
    fs=base_fs/np.sqrt(mult)
    sp=dataclasses.replace(scene.solve,fragment_size=fs,
       fragment_min_area=scene.solve.fragment_min_area*(fs/base_fs)**2,
       fragment_max_area=scene.solve.fragment_max_area*(fs/base_fs)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=sp)
    table=build_projection_table(ts); renderer=Renderer(ts,table)
    # shard_budget=0 -> disable the fabrication-ceiling coarsening so density controls count
    sc,op,fr,res,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=3,shard_budget=0)
    panel_T=C.stack_transmit_lut(names,sc,si)
    pred=renderer.render_color_np(panel_T)          # REAL render, cross-talk present
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    # subject-aware bad cross-talk (cross-talk present, measured not removed)
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}
    def rk(keep):
        pt=panel_T.copy()
        for gi,p in enumerate(panels):
            if p.name not in keep: pt[gi]=1.0
        return renderer.render_color_np(pt)
    dark=lambda im:(1.0-im.mean(axis=-1))>0.05
    ct={}
    for w in ("A","B"):
        nonprim={p.name for p in panels if prim[p.name]!=w}
        tot=dark(pred[w]).sum()
        xt=dark(rk(nonprim)[w]) if nonprim else np.zeros_like(dark(pred[w]))
        ct[w]={"bad":100*(xt&~subject[w]).sum()/max(tot,1),"good":100*(xt&subject[w]).sum()/max(tot,1)}
    nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
    r={"density":lbl,"mult":mult,"fragment_size":round(fs,4),"shards_A":nA,"shards_B":nB,
       "vs_ceiling_A":round(nA/CEILING,2),"vs_ceiling_B":round(nB/CEILING,2),
       "A_rmse":acc["A"]["rmse"],"A_ssim":acc["A"]["ssim"],"A_edge":acc["A"]["edge_fidelity"],
       "B_rmse":acc["B"]["rmse"],"B_ssim":acc["B"]["ssim"],"B_edge":acc["B"]["edge_fidelity"],
       "A_bad":ct["A"]["bad"],"B_bad":ct["B"]["bad"],"A_good":ct["A"]["good"],"B_good":ct["B"]["good"]}
    results.append(r); imgs[lbl]={"A":np.clip(pred["A"],0,1),"B":np.clip(pred["B"],0,1)}
    print(f"{lbl:5s} fs={fs:.4f} shards A={nA} B={nB} (A={nA/CEILING:.1f}x ceiling) | "
          f"A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f}/{acc['A']['edge_fidelity']:.3f} "
          f"B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}/{acc['B']['edge_fidelity']:.3f} | "
          f"A_bad={ct['A']['bad']:.1f}% B_bad={ct['B']['bad']:.1f}%")

json.dump(results,open("out_thickness_test/density_real_results.json","w"),indent=1)

# montage: target + 6 densities, both walls
fig,ax=plt.subplots(2,7,figsize=(21,6.5))
ax[0,0].imshow(np.clip(targets["A"],0,1),origin="lower",aspect="auto"); ax[0,0].set_title("target")
ax[1,0].imshow(np.clip(targets["B"],0,1),origin="lower",aspect="auto")
ax[0,0].set_ylabel("Wall A"); ax[1,0].set_ylabel("Wall B")
for a in (ax[0,0],ax[1,0]): a.set_xticks([]); a.set_yticks([])
for ci,(lbl,mult) in enumerate(DENS):
    r=results[ci]
    for ri,w in enumerate(("A","B")):
        ax[ri,ci+1].imshow(imgs[lbl][w],origin="lower",aspect="auto"); ax[ri,ci+1].set_xticks([]); ax[ri,ci+1].set_yticks([])
    ax[0,ci+1].set_title(f"{lbl}\nA:{r['shards_A']} B:{r['shards_B']} shards\n({r['vs_ceiling_A']}x ceiling)",fontsize=10)
plt.suptitle("REAL-pipeline density sweep (cross-talk PRESENT) — wide/seed3 base, budget disabled",fontsize=14)
plt.tight_layout()
import os; os.makedirs("out_thickness_test/density",exist_ok=True)
plt.savefig("out_thickness_test/density/montage.png",dpi=88,bbox_inches="tight")
print("saved out_thickness_test/density/montage.png and density_real_results.json")
