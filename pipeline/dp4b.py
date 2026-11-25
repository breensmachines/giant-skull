import json,numpy as np
S0=json.load(open('dp4scores.json'))
S={}
for k,v in S0.items():
    i,n=map(int,k.split(',')); S[(i,n)]=v
mx={}
for (i,n),v in S.items(): mx[i]=max(mx.get(i,0),v)
DUP={19:17,20:18,60:59}
U=[i for i in range(1,115) if i not in DUP]
anch={1:27,12:36,22:45,110:131}
pen={0:0.06,1:0.0,2:0.06}
INF=-1e9; best={}; back={}
for k,i in enumerate(U):
    for n in range(27,135):
        if (i,n) not in S and i not in anch: continue
        if i in anch and n!=anch[i]: continue
        sc=S.get((i,n),mx.get(i,0))/max(mx.get(i,1e-6),1e-6)  # relative to frame best
        if k==0: best[(i,n)]=sc; continue
        p=U[k-1]
        cand=[(best.get((p,n-d),INF)-pen[d],n-d) for d in (0,1,2)]
        v,pn=max(cand)
        if v<=INF/2: continue
        best[(i,n)]=v+sc; back[(i,n)]=pn
last=U[-1]
n=max((v,n) for (i,n),v in best.items() if i==last and n<=134)[1]
m={last:n}
for k in range(len(U)-1,0,-1):
    n=back[(U[k],n)]; m[U[k-1]]=n
for d,s in DUP.items(): m[d]=m[s]
path=[m[i] for i in range(1,115)]
print(list(enumerate(path,1)))
json.dump(path,open('dp4path.json','w'))
