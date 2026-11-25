import cv2, numpy as np, sys
def seg(i, dbg=False):
    im=cv2.imread(f'hq/{i:03d}.png'); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    foam=((hsv[...,2]>150)&(hsv[...,1]<60)).astype(np.uint8)
    e=cv2.Canny(cv2.GaussianBlur(g,(3,3),0),30,80)
    e=cv2.dilate(e,np.ones((2,2),np.uint8))
    reg=foam&(e==0)
    reg=cv2.morphologyEx(reg.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(reg,8)
    if n<2: return np.zeros_like(reg)
    areas=st[1:,4]; big=areas.max()
    keep=[k+1 for k,a in enumerate(areas) if a>max(800,0.12*big)]
    m=np.isin(lab,keep).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    return m
if __name__=='__main__':
    tiles=[]
    for i in map(int,sys.argv[1:]):
        im=cv2.imread(f'hq/{i:03d}.png'); m=seg(i)
        ov=im.copy(); ov[m>0]=(0.5*ov[m>0]+[0,100,200]).clip(0,255)
        cs,_=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov,cs,-1,(0,255,255),2)
        tiles.append(cv2.resize(ov,(360,450)))
    cv2.imwrite('/tmp/seg.jpg',np.hstack(tiles))
