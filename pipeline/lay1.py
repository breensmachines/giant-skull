import numpy as np
from PIL import Image
from scipy import ndimage as ndi
def red(n):
    a=np.asarray(Image.open(f'layered/{n:03d}.jpg').convert('RGB')).astype(int)
    return (a[...,0]>140)&(a[...,1]<110)&(a[...,2]<110)
def white(n):
    a=np.asarray(Image.open(f'layered/{n:03d}.jpg').convert('L'))
    return a>128
def ctedge(n,thr=None):
    c=np.asarray(Image.open(f'site2/ct/{n:03d}.jpg').convert('L')).astype(float)
    b=c>(thr or 170)
    b=ndi.binary_opening(b)
    e=b^ndi.binary_erosion(b)
    return b,e
