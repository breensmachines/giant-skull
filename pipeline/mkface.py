import sys; sys.path.insert(0,'/home/claude')
from lay2 import *
BAD={47,56,57,59}
for n in range(27,135):
    if n in BAD: continue
    f,a,w,l=face4(n,thr=105,dil=3,frac=0.55,er=2)
    Image.fromarray((f*255).astype(np.uint8)).save(f'layface/{n:03d}.png')
print('ok')
