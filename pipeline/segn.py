import cv2,numpy as np
def segn(i, drop=22, xs=None, loose=False):
    im=cv2.imread(f'hq/{i:03d}.png'); lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    L=lab[...,0].astype(np.float32); ch=np.hypot(lab[...,1].astype(float)-128,lab[...,2].astype(float)-128)
    Lb=cv2.GaussianBlur(L,(3,3),0)
    loc=cv2.dilate(Lb,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
    k=(3,3) if loose else (5,5); mu=cv2.blur(L,k); sd=np.sqrt(np.maximum(cv2.blur(L*L,k)-mu*mu,0))
    e=cv2.Canny(cv2.GaussianBlur(im,(3,3),0),18,45)
    if xs is not None:
        x=int(round(xs)); a,b=x-4,x+5
        side=np.minimum(np.roll(Lb,6,1),np.roll(Lb,-6,1))   # bright on both sides of the seam line
        Lb[:,a:b]=np.maximum(Lb[:,a:b],side[:,a:b]); sd[:,a:b]=0; e[:,a:b]=0
        ch[:,a:b]=np.minimum(ch[:,a:b],np.minimum(np.roll(ch,6,1),np.roll(ch,-6,1))[:,a:b])
    m=((Lb>=loc-drop)&(Lb>140)&(ch<16)&(sd<(14 if loose else 9))&(e==0)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    m=np.isin(l,[k for k in range(1,n) if st[k,4]>=60]).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return m
