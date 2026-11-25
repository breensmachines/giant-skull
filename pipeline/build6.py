import cv2,numpy as np,json,re,sys
sys.path.insert(0,'/home/claude')
from tw import T, reg
import build2 as b
O=json.load(open('/home/claude/outlines_v5.json'))
def raster(p):
    m=np.zeros((900,720),np.uint8)
    polys=[np.array(re.findall(r'(-?[\d.]+),(-?[\d.]+)',q),float).astype(np.int32) for q in p.split('Z') if q.strip()]
    polys=[q for q in polys if len(q)>2]
    # even-odd fill for holes
    for q in polys:
        t=np.zeros_like(m); cv2.fillPoly(t,[q],1); m^=t
    return m
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def shift(m,dx,dy):
    M=np.float32([[1,0,dx],[0,1,dy]]); return cv2.warpAffine(m,M,(720,900),flags=cv2.INTER_NEAREST)
def build6(i,tonly=None):
    xs=b.F[i-1]; x=int(xs)
    S=raster(O[i-1]); SR=S.copy(); SR[:,:x]=0
    t=T(i)
    # local alignment
    best=(-1,0,0); Sd=cv2.dilate(SR,disk(2))
    NOSEARCH=True
    for dx in ([] if NOSEARCH else range(-8,9,2)):
        for dy in range(-8,9,2):
            ts=shift(t,dx,dy); ts[:,:x]=0
            inter=(ts&Sd).sum(); u=(ts|SR).sum()
            sc=inter/max(u,1)
            if sc>best[0]: best=(sc,dx,dy)
    _,dx,dy=best
    for ddx in (() if NOSEARCH else (-1,0,1)):
        for ddy in (-1,0,1):
            ts=shift(t,dx+ddx,dy+ddy); ts[:,:x]=0
            sc=(ts&Sd).sum()/max((ts|SR).sum(),1)
            if sc>best[0]: best=(sc,dx+ddx,dy+ddy)
    _,dx,dy=best
    if NOSEARCH or best[0]<0.3: dx,dy=0,0      # too little overlap to trust the shift
    t=shift(t,dx,dy); t[:,:x]=0
    BW=62
    cov=t.sum()/max(SR.sum(),1)
    if tonly is None: tonly = cov<0.75
    if tonly:
        R=cv2.morphologyEx(t,cv2.MORPH_OPEN,disk(1))
        n,l,st,_=cv2.connectedComponentsWithStats(R,8)
        if n>1:
            big=st[1:,4].max()
            for q in range(1,n):
                if st[q,4]<0.08*big and st[q,4]<1500: R[l==q]=0
    else:
        td=cv2.dilate(t,disk(8))
        R=SR.copy()
        n,l,st,_=cv2.connectedComponentsWithStats(SR,8)
        for q in range(1,n):
            c=(l==q)
            if (td[c]>0).mean()<0.3: R[c]=0          # photo bits not in his tracing
        R&=td                                         # tracing is ground truth: trim photo beyond it
        band=np.zeros_like(SR); band[:,x+4:x+BW]=1
        nb=(SR & cv2.dilate(t,disk(3))) | (t & cv2.dilate(SR,disk(3)))
        R=np.where(band>0, nb, R).astype(np.uint8)
        R=cv2.morphologyEx(R,cv2.MORPH_OPEN,disk(1))
    R[:,:x]=0
    full=np.maximum(R,(b.R2L(R*255,xs)>127).astype(np.uint8))
    full[:,x-1:x+2]=np.maximum(full[:,x-1:x+2],(cv2.dilate(full,np.ones((1,9),np.uint8))[:,x-1:x+2]))
    n,l,st,_=cv2.connectedComponentsWithStats(full,8)
    full=np.isin(l,[q for q in range(1,n) if st[q,4]>=120]).astype(np.uint8)
    inv=1-full; n,l,st,_=cv2.connectedComponentsWithStats(inv,4)
    for q in range(1,n):
        if st[q,4]<40: full[l==q]=1
    return full,(dx,dy,best[0])
if __name__=='__main__':
    if sys.argv[1]=='all':
        out=[];info=[]
        for i in range(1,116):
            if i>114: out.append(O[i-1]); continue
            f,inf=build6(i); out.append(b.paths(f)); info.append(inf)
        json.dump(out,open('/home/claude/outlines_v6.json','w')); json.dump(info,open('/home/claude/v6info.json','w'),default=float); print('ok')
    else:
        tl=[]
        for i in map(int,sys.argv[1:]):
            im=cv2.imread(f'/home/claude/hq/{i:03d}.png'); f,inf=build6(i); print(i,inf)
            cs,_=cv2.findContours(f,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(255,60,0),1)
            S=raster(O[i-1]); cs,_=cv2.findContours(S,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(0,200,255),1)
            tl.append(im[60:820])
        cv2.imwrite('/tmp/b6.jpg',np.hstack(tl))
