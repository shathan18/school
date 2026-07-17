"""Shared: face ROI from a loaded target, face-region metrics, semantic-aware evaluate."""
import numpy as np
from scipy import ndimage
from shadowart.targets import color as C
from shadowart import metrics as _metrics

def face_roi_from_target(t):
    """Deterministic skin-tone face ROI computed on the ALREADY-LOADED target (wall_res,
    oriented) so it aligns with shard generation. Returns soft [0,1] mask + bbox."""
    R,G,B=t[...,0],t[...,1],t[...,2]; mx=t.max(-1); mn=t.min(-1)
    flesh=(R>0.37)&(G>0.16)&(B>0.08)&((mx-mn)>0.06)&(R>G)&(R>=B)&((R-G)>0.06)
    H=t.shape[0]; flesh[:int(H*0.35),:]=False   # target is flipud (image-top->wall-top=high row); face is upper image = high rows -> keep top 65% rows i.e. rows>=0.35H
    flesh=ndimage.binary_opening(flesh,iterations=2)
    lbl,n=ndimage.label(flesh)
    if n:
        s=np.bincount(lbl.ravel()); s[0]=0; flesh=(lbl==s.argmax())
    if flesh.sum()==0:  # fallback: upper-centre box
        flesh=np.zeros_like(flesh); flesh[int(H*0.55):int(H*0.9),int(t.shape[1]*0.25):int(t.shape[1]*0.75)]=True
    flesh=ndimage.binary_dilation(flesh,iterations=6)
    soft=ndimage.gaussian_filter(flesh.astype(np.float32),sigma=max(t.shape[:2])*0.03)
    if soft.max()>0: soft/=soft.max()
    ys,xs=np.where(soft>0.4)
    bbox=(ys.min(),ys.max()+1,xs.min(),xs.max()+1) if len(ys) else (0,H,0,t.shape[1])
    return soft,bbox

def face_metrics(pred,target,bbox):
    y0,y1,x0,x1=bbox
    p=pred[y0:y1,x0:x1]; t=target[y0:y1,x0:x1]
    return _metrics.ssim(p,t), _metrics.edge_fidelity(p,t)
