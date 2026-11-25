import cv2,numpy as np,json
reg=json.load(open('/home/claude/reg.json'))
SX=697/210; SY=1360/413
BAD={47,56,57,59}
H,W=900,720
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
def lay(n):
    while n in BAD: n+=1
    return cv2.imread(f'/home/claude/layface/{n:03d}.png',0)
def T(i):
    r=reg[i-1]; f=lay(r['n'])
    X=(xx-r['s']-r.get('dx',0))/r['a']; Y=(yy-r['ty'])/r['b']
    lx=X*SX; ly=(Y-48)*SY
    w=cv2.remap(f,lx.astype(np.float32),ly.astype(np.float32),cv2.INTER_LINEAR,borderValue=0)
    w[:,:int(r['s'])]=0
    return (w>127).astype(np.uint8)
