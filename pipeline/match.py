import cv2,numpy as np,json,itertools
from ctprep import ctbone
import build4
F=build4.b.F
MID=264.0
CT={n:cv2.dilate(ctbone(n),np.ones((3,3),np.uint8)) for n in range(27,135)}
for n in CT: CT[n][:, :int(MID)]=0
_P={}
def photo(i):
    if i not in _P:
        m=build4.clean(i).copy(); xs=int(round(F[i-1])); m[:,:xs]=0
        _P[i]=cv2.dilate(m,np.ones((5,5),np.uint8))
    return _P[i]
def warp_to_ct(i,a,b,ty):
    # photo -> ct: x_ct = MID + (x_p - seam)/a ; y_ct = (y_p - ty)/b
    s=F[i-1]
    A=np.float32([[1/a,0,MID - s/a],[0,1/b,-ty/b]])
    return cv2.warpAffine(photo(i),A,(512,512),flags=cv2.INTER_NEAREST)
def iou(x,y):
    u=(x|y).sum(); return (x&y).sum()/u if u else 0
def score(i,n,a,b,ty): return iou(warp_to_ct(i,a,b,ty),CT[n])
if __name__=='__main__':
    res={}
    for i in [20,35,50,65,80,95,108]:
        n0=int(round(27+(i-1)*107/113))
        best=(-1,None)
        for n in range(max(27,n0-14),min(134,n0+14)+1,2):
            for a in np.arange(0.9,2.3,0.1):
                for b in np.arange(0.9,2.3,0.1):
                    for ty in range(-300,301,25):
                        s=score(i,n,a,b,ty)
                        if s>best[0]: best=(s,(n,round(a,2),round(b,2),ty))
        res[i]=best; print(i,best,flush=True)
    json.dump({str(k):v for k,v in res.items()},open('coarse.json','w'))
