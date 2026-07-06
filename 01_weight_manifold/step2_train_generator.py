import torch, torch.nn as nn, numpy as np, time
from common import *
torch.manual_seed(1); np.random.seed(1)
d=torch.load("weights_ds.pt"); W=d["W"]; w0=d["w0"]; NT=W.shape[0]; NPARAM=W.shape[1]
mu,sd=W.mean(0),W.std(0)+1e-6; Wn=(W-mu)/sd
ZDIM=8
enc=nn.Sequential(nn.Linear(NPARAM,256),nn.SiLU(),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,ZDIM))
dec=nn.Sequential(nn.Linear(ZDIM,64),nn.SiLU(),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,NPARAM))
opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
probe=torch.linspace(-5,5,96).unsqueeze(1)
t0=time.time()
for ep in range(4000):
    idx=torch.randint(0,NT,(32,))
    Wr=dec(enc(Wn[idx])); lw=((Wr-Wn[idx])**2).mean(); lf=0.
    for k in range(5):
        j=idx[k]
        with torch.no_grad(): yt=forward_flat(W[j],probe)
        lf=lf+((forward_flat(Wr[k]*sd+mu,probe)-yt)**2).mean()
    loss=0.2*lw+lf/5; opt.zero_grad(); loss.backward(); opt.step()
print(f"AE {time.time()-t0:.0f}s f-loss {(lf/5).item():.4f}")
torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"mu":mu,"sd":sd},"ae.pt")
