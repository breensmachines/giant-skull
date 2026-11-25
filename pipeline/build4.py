import cv2,numpy as np,json
import build2 as b
M=np.load("mand.npy").astype(bool); M[:600]=False
Mx=cv2.dilate(M.astype(np.uint8),np.ones((15,15),np.uint8))>0
def clean(i):
    full=b.build(i); xs=b.F[i-1]
    n,l,st,c=cv2.connectedComponentsWithStats(full,8)
    for k in range(1,n):
        x0,y0,w,h,a=st[k]
        comp=(l==k)
        inM=(comp&Mx).sum()/a
        if y0>=560 and abs(c[k][0]-xs)<25 and w<=40 and inM>0.3: full[comp]=0
    return full
if __name__=='__main__':
    out=[b.paths(clean(i)) for i in range(1,116)]
    json.dump(out,open('outlines_v5.json','w')); print('ok')
