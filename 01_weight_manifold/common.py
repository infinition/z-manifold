import torch, torch.nn as nn, numpy as np
def forward_flat(w,x):
    i0=0; h=x
    for fin,fout in [(1,16),(16,16),(16,1)]:
        Wl=w[i0:i0+fin*fout].view(fout,fin); i0+=fin*fout
        bl=w[i0:i0+fout]; i0+=fout
        h=h@Wl.T+bl
        if fout!=1: h=torch.tanh(h)
    return h
def task_data(A,phi,n): x=torch.rand(n,1)*10-5; return x, A*torch.sin(x+phi)
XG=torch.linspace(-5,5,400).unsqueeze(1)
BASIS=torch.cat([torch.sin(XG),torch.cos(XG)],1)  # family = span{sin,cos}
PINV=torch.linalg.pinv(BASIS)
def eval_dense(w,A,phi):
    with torch.no_grad(): return ((forward_flat(w,XG)-A*torch.sin(XG+phi))**2).mean().item()
def validity(w):
    with torch.no_grad():
        y=forward_flat(w,XG); ab=PINV@y
        return ((y-BASIS@ab)**2).mean().item()
