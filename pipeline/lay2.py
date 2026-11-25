import numpy as np
from PIL import Image, ImageFile
from scipy import ndimage as ndi
ImageFile.LOAD_TRUNCATED_IMAGES=True
def load(n):
    return np.asarray(Image.open(f'layered/{n:03d}.jpg').convert('RGB')).astype(int)
def parts(a):
    g=a.mean(-1)
    white=(a.min(-1)>150)
    dark=~white
    bg=ndi.binary_opening(dark,np.ones((7,7)))
    lab,k=ndi.label(bg); sz=ndi.sum(bg,lab,range(1,k+1))
    bg=np.isin(lab,1+np.nonzero(sz>3000)[0])
    bg=ndi.binary_dilation(bg,iterations=3)&dark
    lines=dark&~bg
    return white,lines,bg
def face(n,nxt=None):
    a=load(n); w,l,bg=parts(a)
    lab,k=ndi.label(w)
    if nxt is None: return w,lab,k,None
    b=nxt; w2,l2,bg2=parts(b)
    l2d=ndi.binary_dilation(l2|bg2,iterations=2)
    keep=np.zeros(k+1,bool); pers=np.zeros(k+1)
    objs=ndi.find_objects(lab)
    for i,sl in enumerate(objs,1):
        if sl is None: continue
        sl2=tuple(slice(max(s.start-4,0),s.stop+4) for s in sl)
        C=lab[sl2]==i
        if C.sum()<20: continue
        ring=ndi.binary_dilation(C,iterations=3)&l[sl2]
        if ring.sum()<5: keep[i]=True; continue
        p=l2d[sl2][ring].mean(); pers[i]=p
        keep[i]=p<0.6
    return keep[lab],lab,pers,l

def isred(a): return (a[...,0]>150)&(a[...,0]-a[...,1]>60)
def face2(n,prev):
    a=load(n); w,l,bg=parts(a)
    lab,k=ndi.label(w)
    if prev is None: return w,lab
    rp=ndi.binary_dilation(isred(prev),iterations=2)
    wp,lp,bgp=parts(prev)
    blk=lp&~isred(prev)
    keep=np.zeros(k+1,bool)
    for i,sl in enumerate(ndi.find_objects(lab),1):
        if sl is None: continue
        sl2=tuple(slice(max(s.start-4,0),s.stop+4) for s in sl)
        C=lab[sl2]==i
        if C.sum()<15: continue
        ring=ndi.binary_dilation(C,iterations=2)&~C&(l[sl2]|bg[sl2])
        r=rp[sl2][ring].sum(); b=(blk[sl2][ring]&~rp[sl2][ring]).sum()
        keep[i]= r>0.6*(r+b) if (r+b)>8 else False
    return keep[lab],lab

SX=697/210; SY=1360/413
def ctmask_lay(k,thr=150):
    c=np.asarray(Image.open(f'site2/ct/{k:03d}.jpg').convert('L'))
    crop=c[48:461,264:474]
    m=Image.fromarray(((crop>thr)*255).astype(np.uint8)).resize((697,1360),Image.BILINEAR)
    return np.asarray(m)>127
def face3(n,k=None,thr=150,frac=0.5):
    a=load(n); w,l,bg=parts(a)
    lab,K=ndi.label(w)
    cm=ctmask_lay(k or n,thr)
    ins=ndi.sum(cm,lab,range(1,K+1)); sz=ndi.sum(w,lab,range(1,K+1))
    keep=np.r_[False,(ins/np.maximum(sz,1)>frac)&(sz>15)]
    return keep[lab],w,l,cm

def face4(n,k=None,thr=120,dil=10,frac=0.7,er=1):
    a=load(n); w,l,bg=parts(a)
    we=ndi.binary_erosion(w,iterations=er) if er else w
    lab,K=ndi.label(we)
    cm=ctmask_lay(k or n,thr); cmd=ndi.binary_dilation(cm,iterations=dil)
    idx=np.arange(1,K+1)
    ins=ndi.sum(cmd,lab,idx); sz=ndi.sum(we,lab,idx)
    keep=np.r_[False,(ins/np.maximum(sz,1)>frac)&(sz>15)]
    f=keep[lab]
    if er: f=ndi.binary_dilation(f,iterations=er)&w
    return f,a,w,l

def face5(n,thr=120,dil=10,frac=0.7,er=2,bth=0.45,dbg=False):
    a=load(n); w,l,bg=parts(a)
    we=ndi.binary_erosion(w,iterations=er)
    lab,K=ndi.label(we)
    cm=ctmask_lay(n,thr); cmd=ndi.binary_dilation(cm,iterations=dil)
    idx=np.arange(1,K+1)
    ins=ndi.sum(cmd,lab,idx); sz=ndi.sum(we,lab,idx)
    cand=(ins/np.maximum(sz,1)>frac)&(sz>15)
    keep=np.zeros(K+1,bool)
    if n>27:
        p=load(n-1); wp,lp,bgp=parts(p)
        redp=ndi.binary_dilation(isred(p),iterations=2)
        blk=lp&~redp
    for i,sl in enumerate(ndi.find_objects(lab),1):
        if sl is None or not cand[i-1]: continue
        if n==27: keep[i]=True; continue
        sl2=tuple(slice(max(s.start-6,0),s.stop+6) for s in sl)
        C=lab[sl2]==i
        ring=ndi.binary_dilation(C,iterations=er+2)&~ndi.binary_dilation(C,iterations=er)&l[sl2]
        if ring.sum()<6: keep[i]=True; continue
        bf=blk[sl2][ring].mean()
        keep[i]= bf<bth
    f=ndi.binary_dilation(keep[lab],iterations=er)&w
    return f,a
