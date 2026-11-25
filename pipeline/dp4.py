import json,sys,cv2,numpy as np
sys.path.insert(0,'/home/claude')
import tw, build6 as B
reg=json.load(open('reg.json')); F=B.b.F
DUP={19:17,20:18,60:59}
U=[i for i in range(1,115) if i not in DUP]
anch={12:36,22:45,110:131}
# rough linear guess for window centers
xs=[1,12,22,110,114]; ys=[27,36,45,131,134]
S={}
for i in U:
    c=int(round(np.interp(i,xs,ys)))
    x=int(F[i-1]); SR=B.raster(B.O[i-1]); SR[:,:x]=0
    for n in range(max(27,c-5),min(134,c+5)+1):
        rr=dict(reg[i-1]); rr['n']=n; tw.reg[i-1]=rr
        t=tw.T(i); S[f'{i},{n}']=float((t&SR).sum()/max((t|SR).sum(),1))
    tw.reg[i-1]=reg[i-1]
json.dump(S,open('dp4scores.json','w')); print('ok')
