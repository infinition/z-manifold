import torch, torch.nn as nn, numpy as np, json, time
from common import *
torch.manual_seed(2); np.random.seed(2)
d=torch.load("weights_ds.pt"); W=d["W"]; w0=d["w0"]; NPARAM=W.shape[1]
a=torch.load("ae.pt"); mu,sd=a["mu"],a["sd"]; ZDIM=8
dec=nn.Sequential(nn.Linear(ZDIM,64),nn.SiLU(),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,NPARAM))
dec.load_state_dict(a["dec"])
for p in dec.parameters(): p.requires_grad_(False)
def decode(z): return dec(z)*sd+mu
def adapt_ft(wi,xs,ys,steps=150,lr=1e-2):
    w=wi.clone().requires_grad_(True); o=torch.optim.Adam([w],lr=lr)
    for _ in range(steps):
        l=((forward_flat(w,xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return w.detach(), l.item()
def adapt_z(xs,ys,z0=None,steps=150,lr=5e-2):
    z=(z0.clone() if z0 is not None else torch.zeros(1,ZDIM)).requires_grad_(True)
    o=torch.optim.Adam([z],lr=lr)
    for _ in range(steps):
        l=((forward_flat(decode(z)[0],xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return z.detach(), l.item()
R={}; t0=time.time()
# S1 few-shot
Ks=[3,5,10,20]; s1={K:{"ft":[],"z":[]} for K in Ks}
for t in range(15):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    for K in Ks:
        xs,ys=task_data(A,phi,K)
        w,_=adapt_ft(w0,xs,ys); s1[K]["ft"].append(eval_dense(w,A,phi))
        z,_=adapt_z(xs,ys);     s1[K]["z"].append(eval_dense(decode(z)[0],A,phi))
R["fewshot"]={K:{m:float(np.median(v)) for m,v in s1[K].items()} for K in Ks}
print(f"S1 {time.time()-t0:.0f}s")
# S2 forgetting: chain of 5, recover task1 with K=3
s2={"ft_last":[],"z_last":[],"ft_rec":[],"z_rec":[],"ft_val":[],"z_val":[]}
for tr in range(10):
    chain=[(np.random.uniform(0.5,3),np.random.uniform(0,np.pi)) for _ in range(5)]
    wc=w0.clone(); zc=torch.zeros(1,ZDIM)
    for (A,phi) in chain:
        xs,ys=task_data(A,phi,10)
        wc,_=adapt_ft(wc,xs,ys); zc,_=adapt_z(xs,ys,z0=zc)
    s2["ft_last"].append(eval_dense(wc,*chain[-1])); s2["z_last"].append(eval_dense(decode(zc)[0],*chain[-1]))
    s2["ft_val"].append(validity(wc)); s2["z_val"].append(validity(decode(zc)[0]))
    xs,ys=task_data(*chain[0],3)
    wr,_=adapt_ft(wc,xs,ys); s2["ft_rec"].append(eval_dense(wr,*chain[0]))
    zr,_=adapt_z(xs,ys,z0=zc); s2["z_rec"].append(eval_dense(decode(zr)[0],*chain[0]))
R["forgetting"]={k:float(np.median(v)) for k,v in s2.items()}
print(f"S2 {time.time()-t0:.0f}s")
# S3 poison 50%
s3={"ft":[],"z":[],"ft_val":[],"z_val":[]}
for t in range(12):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20)
    m=torch.rand(20,1)<0.5; ys_p=torch.where(m,torch.rand(20,1)*8-4,ys)
    w,_=adapt_ft(w0,xs,ys_p); s3["ft"].append(eval_dense(w,A,phi)); s3["ft_val"].append(validity(w))
    z,_=adapt_z(xs,ys_p);     s3["z"].append(eval_dense(decode(z)[0],A,phi)); s3["z_val"].append(validity(decode(z)[0]))
R["poison"]={k:float(np.median(v)) for k,v in s3.items()}
print(f"S3 {time.time()-t0:.0f}s")
# S4 OOD detection
lin,lo=[],[]
for t in range(12):
    xs,ys=task_data(np.random.uniform(0.5,3),np.random.uniform(0,np.pi),20); _,l=adapt_z(xs,ys); lin.append(l)
    xs,ys=task_data(np.random.uniform(4,5),np.random.uniform(0,np.pi),20); _,l=adapt_z(xs,ys); lo.append(l)
thr=(np.median(lin)+np.median(lo))/2
acc=(np.mean([l<thr for l in lin])+np.mean([l>thr for l in lo]))/2
R["ood_detect"]={"loss_in":float(np.median(lin)),"loss_ood":float(np.median(lo)),"acc":float(acc)}
print(f"S4 {time.time()-t0:.0f}s")
# S5 hybrid on OOD
def adapt_hybrid(xs,ys,steps=250,lam=1e-3):
    z,_=adapt_z(xs,ys); base=decode(z)[0].detach()
    delta=torch.zeros(NPARAM,requires_grad=True); o=torch.optim.Adam([delta],lr=5e-3)
    for _ in range(steps):
        l=((forward_flat(base+delta,xs)-ys)**2).mean()+lam*(delta**2).sum()
        o.zero_grad(); l.backward(); o.step()
    return base+delta.detach()
s5={"z":[],"ft":[],"hyb":[],"hyb_val":[]}
for t in range(10):
    A=np.random.uniform(4,5); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20)
    z,_=adapt_z(xs,ys); s5["z"].append(eval_dense(decode(z)[0],A,phi))
    w,_=adapt_ft(w0,xs,ys); s5["ft"].append(eval_dense(w,A,phi))
    wh=adapt_hybrid(xs,ys); s5["hyb"].append(eval_dense(wh,A,phi)); s5["hyb_val"].append(validity(wh))
R["hybrid_ood"]={k:float(np.median(v)) for k,v in s5.items()}
print(f"S5 {time.time()-t0:.0f}s")
print(json.dumps(R,indent=1)); json.dump(R,open("results_v3.json","w"))
