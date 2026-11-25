import cv2,numpy as np,json,re
F=json.load(open('/home/claude/seamfit.json'))
from seg import seg as seg0
from segn import segn
NOSE=95
def seg(i):
    b=segn(i,xs=F[i-1]); bl=segn(i,xs=F[i-1],loose=True); x=int(round(F[i-1])); W=58
    b[:,max(0,x-int(W*.8)):x+W]=bl[:,max(0,x-int(W*.8)):x+W]; a=seg0(i); return (b | (a & cv2.dilate(b,np.ones((3,3),np.uint8)))).astype(np.uint8)
a,c,d=-1.0,0.00125,0.3
H,W=900,720
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
def R2L(R,xs):  # map right-side mask onto left (target left pixels)
    up=xx-xs; u=up/(a-c*up); y=yy*(1+c*u)-d*u
    w=cv2.remap(R,(u+xs).astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,borderValue=0); w[:,int(xs):]=0; return w
def L2R(L,xs):  # map left mask onto right (target right pixels)
    u=xx-xs; up=a*u/(1+c*u); y=(yy+d*u)/(1+c*u)
    w=cv2.remap(L,(up+xs).astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,borderValue=0); w[:,:int(xs)]=0; return w
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def gaps(i,xs,T=18,roi=105):
    L=cv2.cvtColor(cv2.imread(f'hq/{i:03d}.png'),cv2.COLOR_BGR2LAB)[...,0]
    bh=cv2.morphologyEx(L,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    g=(bh>T).astype(np.uint8)
    x=int(round(xs)); keep=np.zeros_like(g); keep[:,max(0,x-int(roi*0.8)):x+roi]=1; keep[:,x-6:x+7]=0
    g&=keep
    n,l,st,_=cv2.connectedComponentsWithStats(g,8)
    g=np.isin(l,[q for q in range(1,n) if st[q,4]>=12]).astype(np.uint8)
    return g*255
def build(i,tol=5,grow=10):
    xs=F[i-1]; m=seg(i)*255
    R=m.copy(); R[:,:int(xs)]=0; L=m.copy(); L[:,int(xs)+1:]=0
    Lp=L2R(L,xs)
    keepR=cv2.bitwise_and(R,cv2.dilate(Lp,disk(tol)))
    addL=cv2.bitwise_and(Lp,cv2.dilate(R,disk(grow)))
    S=cv2.bitwise_or(keepR,addL)
    S=cv2.morphologyEx(S,cv2.MORPH_CLOSE,disk(2))
    x=int(round(xs)); B=9
    band=np.maximum(m[:,x:x+B], np.fliplr(m[:,x-B:x]))
    S[:,x:x+B]=np.maximum(S[:,x:x+B],band)
    G=gaps(i,xs); G[:,:int(xs)]=0
    S=cv2.bitwise_and(S,cv2.bitwise_not(G))
    n,l,st,cen=cv2.connectedComponentsWithStats((S>127).astype(np.uint8),8)
    if n>1:
        big=st[1:,4].max()
        for q in range(1,n):
            if st[q,4]<0.03*big and st[q,4]<1500 and abs(cen[q][0]-xs)>62: S[l==q]=0
    full=cv2.bitwise_or(S,R2L(S,xs))
    # bridge seam
    full[:,int(xs)-1:int(xs)+2]=np.maximum(full[:,int(xs)-1:int(xs)+2], (cv2.dilate(full,np.ones((1,9),np.uint8))[:,int(xs)-1:int(xs)+2]>127)*255).astype(np.uint8)
    full=(full>127).astype(np.uint8)
    n,l,st,_=cv2.connectedComponentsWithStats(full,8)
    full=np.isin(l,[q for q in range(1,n) if st[q,4]>=250]).astype(np.uint8)
    inv=1-full; n,l,st,_=cv2.connectedComponentsWithStats(inv,4)
    for q in range(1,n):
        if st[q,4]<80: full[l==q]=1
    return full
def paths(mk):
    cs,_=cv2.findContours(mk,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE); parts=[]
    for cc in cs:
        if cv2.contourArea(cc)<40: continue
        ap=cv2.approxPolyDP(cc,1.0,True).reshape(-1,2)
        if len(ap)>=3: parts.append('M'+' '.join(f'{x},{y}' for x,y in ap)+'Z')
    return ''.join(parts)
if __name__=='__main__':
    import sys
    if sys.argv[1]=='all':
        out=[paths(build(i)) for i in range(1,116)]; json.dump(out,open('outlines_v3.json','w')); print('done')
    else:
        t=[]
        for i in map(int,sys.argv[1:]):
            im=cv2.imread(f'hq/{i:03d}.png'); mk=build(i)
            cs,_=cv2.findContours(mk,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(0,220,255),1)
            t.append(im[40:740,0:720])
        cv2.imwrite('/tmp/b2.jpg',cv2.resize(np.hstack(t),(720*len(t)*2//3,700*2//3)))
