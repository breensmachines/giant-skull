import cv2,numpy as np
def ctbone(n,T=150):
    g=cv2.imread(f'/home/claude/ct/{n:03d}.jpg',0)
    m=(g>=T).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n_,l,st,_=cv2.connectedComponentsWithStats(m,8)
    m=np.isin(l,[k for k in range(1,n_) if st[k,4]>=6]).astype(np.uint8)
    inv=1-m; n_,l,st,_=cv2.connectedComponentsWithStats(inv,4)
    for k in range(1,n_):
        if st[k,4]<12: m[l==k]=1
    return m
def midline(n):
    g=cv2.imread(f'/home/claude/ct/{n:03d}.jpg',0).astype(np.float32)
    best=(-1,256)
    for c in np.arange(240,276,0.5):
        M=np.float32([[-1,0,2*c],[0,1,0]]); f=cv2.warpAffine(g,M,(512,512))
        s=-np.abs(f-g)[40:470,120:400].mean()
        if s>best[0] or best[0]==-1: best=(s,c)
    return best[1]
if __name__=='__main__':
    print([ (n,midline(n)) for n in (40,60,80,100,120)])
