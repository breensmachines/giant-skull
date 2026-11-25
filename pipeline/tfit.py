import json,sys,cv2,numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'/home/claude')
import build6 as B
reg=json.load(open('reg.json')); F=B.b.F
BAD={47,56,57,59}
def tmask(n):
    while n in BAD: n+=1
    f=cv2.imread(f'layface/{n:03d}.png',0)
    return (cv2.resize(f,(210,413),interpolation=cv2.INTER_AREA)>127).astype(np.uint8)
res=[]
for i in range(12,115,3):
    q=reg[i-1]; n=q['n']; s=F[i-1]
    SR=B.raster(B.O[i-1]); SR[:,:int(s)]=0
    T=tmask(n)
    def sc(p):
        a,b,ty=p; dx=0
        A=np.float32([[1/a,0,264-(s+dx)/a-264],[0,1/b,-ty/b-48]])  # to box coords (X-264, Y-48)
        w=cv2.warpAffine(SR,A,(210,413),flags=cv2.INTER_NEAREST)
        return -( (w&T).sum()/max((w|T).sum(),1) )
    best=None
    for b0 in (1.4,1.6,1.8,2.0):
        p0=[q['a'],b0,q['ty']+(q['b']-b0)*250]
        r=minimize(sc,p0,method='Nelder-Mead',options={'maxfev':300,'initial_simplex':[p0,[p0[0]+.08,p0[1],p0[2]],[p0[0],p0[1]+.08,p0[2]],[p0[0],p0[1],p0[2]+15]]})
        if best is None or r.fun<best.fun: best=r
    a,b,ty=best.x; dx=0
    res.append([i,n,-best.fun,a,b,ty,dx]); print(i,n,round(-best.fun,3),round(a,3),round(b,3),round(b/a,3),round(ty,1),round(dx,1),'| old',round(q['a'],3),round(q['b'],3),flush=True)
json.dump(res,open('tfit0.json','w'))
