"""Benchmark v3 - suite complete: few-shot / forgetting / poisoning / OOD detect / hybrid."""
import torch, torch.nn as nn, numpy as np, json, time
torch.manual_seed(0); np.random.seed(0)

def make_net():
    return nn.Sequential(nn.Linear(1,16), nn.Tanh(), nn.Linear(16,16), nn.Tanh(), nn.Linear(16,1))
def flatten(net): return torch.cat([p.detach().flatten() for p in net.parameters()])
def set_flat(net, vec):
    i=0
    for p in net.parameters():
        n=p.numel(); p.data=vec[i:i+n].view_as(p).clone(); i+=n
def forward_flat(w,x):
    i0=0; h=x
    for fin,fout in [(1,16),(16,16),(16,1)]:
        Wl=w[i0:i0+fin*fout].view(fout,fin); i0+=fin*fout
        bl=w[i0:i0+fout]; i0+=fout
        h=h@Wl.T+bl
        if fout!=1: h=torch.tanh(h)
    return h
def task_data(A,phi,n): x=torch.rand(n,1)*10-5; return x, A*torch.sin(x+phi)

# ---- anchor + dataset (same as v2) ----
anchor=make_net(); opt=torch.optim.Adam(anchor.parameters(),lr=1e-2)
for _ in range(800):
    x,y=task_data(1.5,np.pi/2,64); l=((anchor(x)-y)**2).mean(); opt.zero_grad(); l.backward(); opt.step()
w0=flatten(anchor); NPARAM=w0.numel()
t0=time.time(); NT=400; Ws=[]
for i in range(NT):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    net=make_net(); set_flat(net,w0); o=torch.optim.Adam(net.parameters(),lr=5e-3)
    for _ in range(400):
        x,y=task_data(A,phi,64); l=((net(x)-y)**2).mean(); o.zero_grad(); l.backward(); o.step()
    Ws.append(flatten(net))
W=torch.stack(Ws); mu,sd=W.mean(0),W.std(0)+1e-6; Wn=(W-mu)/sd
print(f"dataset {time.time()-t0:.0f}s")

ZDIM=8
enc=nn.Sequential(nn.Linear(NPARAM,256),nn.SiLU(),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,ZDIM))
dec=nn.Sequential(nn.Linear(ZDIM,64),nn.SiLU(),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,NPARAM))
opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
probe=torch.linspace(-5,5,96).unsqueeze(1)
for ep in range(5000):
    idx=torch.randint(0,NT,(32,))
    Wr=dec(enc(Wn[idx])); lw=((Wr-Wn[idx])**2).mean(); lf=0.
    for k in range(6):
        j=idx[k]
        with torch.no_grad(): yt=forward_flat(W[j],probe)
        lf=lf+((forward_flat(Wr[k]*sd+mu,probe)-yt)**2).mean()
    loss=0.2*lw+lf/6; opt.zero_grad(); loss.backward(); opt.step()
print(f"AE f-loss {(lf/6).item():.4f}")

def decode(z): return dec(z)*sd+mu
def eval_dense(w,A,phi):
    x=torch.linspace(-5,5,400).unsqueeze(1)
    with torch.no_grad(): return ((forward_flat(w,x)-A*torch.sin(x+phi))**2).mean().item()
def adapt_ft(wi,xs,ys,steps=200,lr=1e-2):
    w=wi.clone().requires_grad_(True); o=torch.optim.Adam([w],lr=lr)
    for _ in range(steps):
        l=((forward_flat(w,xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return w.detach(), l.item()
def adapt_z(xs,ys,z0=None,steps=200,lr=5e-2):
    z=(z0.clone() if z0 is not None else torch.zeros(1,ZDIM)).requires_grad_(True)
    o=torch.optim.Adam([z],lr=lr)
    for _ in range(steps):
        l=((forward_flat(decode(z)[0],xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return z.detach(), l.item()

def manifold_validity(w):
    """residual to best-fit family member = is output still a valid sine?"""
    x=torch.linspace(-5,5,200).unsqueeze(1)
    with torch.no_grad(): y=forward_flat(w,x)
    best=1e9
    for A in np.linspace(0.3,5.5,40):
        for phi in np.linspace(0,np.pi,40):
            r=((y-A*torch.sin(x+phi))**2).mean().item()
            best=min(best,r)
    return best

R={}
# ============ S1: few-shot (recap) ============
Ks=[3,5,10,20]; s1={K:{"ft":[],"z":[]} for K in Ks}
for t in range(20):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    for K in Ks:
        xs,ys=task_data(A,phi,K)
        w,_=adapt_ft(w0,xs,ys); s1[K]["ft"].append(eval_dense(w,A,phi))
        z,_=adapt_z(xs,ys);     s1[K]["z"].append(eval_dense(decode(z)[0],A,phi))
R["fewshot"]={K:{m:float(np.median(v)) for m,v in s1[K].items()} for K in Ks}
print("S1 done")

# ============ S2: catastrophic forgetting ============
# chain of 5 tasks sequentially, then recover task1 with only K=3
s2={"ft_taskB":[],"z_taskB":[],"ft_recover":[],"z_recover":[],"ft_valid":[],"z_valid":[]}
for trial in range(12):
    chain=[(np.random.uniform(0.5,3),np.random.uniform(0,np.pi)) for _ in range(5)]
    wc=w0.clone(); zc=torch.zeros(1,ZDIM)
    for (A,phi) in chain:
        xs,ys=task_data(A,phi,10)
        wc,_=adapt_ft(wc,xs,ys)      # sequential FT: weights drift
        zc,_=adapt_z(xs,ys,z0=zc)    # sequential z: moves on manifold
    s2["ft_taskB"].append(eval_dense(wc,*chain[-1]))
    s2["z_taskB"].append(eval_dense(decode(zc)[0],*chain[-1]))
    # validity after the chain
    s2["ft_valid"].append(manifold_validity(wc))
    s2["z_valid"].append(manifold_validity(decode(zc)[0]))
    # recovery of task 1 with K=3 from the drifted state
    xs,ys=task_data(*chain[0],3)
    wr,_=adapt_ft(wc,xs,ys); s2["ft_recover"].append(eval_dense(wr,*chain[0]))
    zr,_=adapt_z(xs,ys,z0=zc); s2["z_recover"].append(eval_dense(decode(zr)[0],*chain[0]))
R["forgetting"]={k:float(np.median(v)) for k,v in s2.items()}
print("S2 done")

# ============ S3: poisoned data (50% random labels) ============
s3={"ft":[],"z":[],"ft_valid":[],"z_valid":[]}
for t in range(15):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20)
    m=torch.rand(20,1)<0.5
    ys_p=torch.where(m, torch.rand(20,1)*8-4, ys)  # 50% poisoned
    w,_=adapt_ft(w0,xs,ys_p); s3["ft"].append(eval_dense(w,A,phi)); s3["ft_valid"].append(manifold_validity(w))
    z,_=adapt_z(xs,ys_p);     s3["z"].append(eval_dense(decode(z)[0],A,phi)); s3["z_valid"].append(manifold_validity(decode(z)[0]))
R["poison"]={k:float(np.median(v)) for k,v in s3.items()}
print("S3 done")

# ============ S4: OOD detection via plateau loss ============
lin,lood=[],[]
for t in range(15):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20); _,l=adapt_z(xs,ys); lin.append(l)
    A=np.random.uniform(4.0,5.0); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20); _,l=adapt_z(xs,ys); lood.append(l)
thr=(np.median(lin)+np.median(lood))/2
acc=(np.mean([l<thr for l in lin])+np.mean([l>thr for l in lood]))/2
R["ood_detect"]={"loss_in":float(np.median(lin)),"loss_ood":float(np.median(lood)),"detect_acc":float(acc)}
print("S4 done")

# ============ S5: hybrid z + regularized residual on OOD ============
def adapt_hybrid(xs,ys,steps=300,lam=1e-3):
    z,_=adapt_z(xs,ys)                      # phase 1: manifold
    delta=torch.zeros(NPARAM,requires_grad=True)
    o=torch.optim.Adam([delta],lr=5e-3)
    base=decode(z)[0].detach()
    for _ in range(steps):                  # phase 2: small residual, L2-anchored
        w=base+delta
        l=((forward_flat(w,xs)-ys)**2).mean()+lam*(delta**2).sum()
        o.zero_grad(); l.backward(); o.step()
    return base+delta.detach()
s5={"z":[],"ft":[],"hybrid":[],"hybrid_valid":[]}
for t in range(12):
    A=np.random.uniform(4.0,5.0); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20)
    z,_=adapt_z(xs,ys); s5["z"].append(eval_dense(decode(z)[0],A,phi))
    w,_=adapt_ft(w0,xs,ys); s5["ft"].append(eval_dense(w,A,phi))
    wh=adapt_hybrid(xs,ys); s5["hybrid"].append(eval_dense(wh,A,phi)); s5["hybrid_valid"].append(manifold_validity(wh))
R["hybrid_ood"]={k:float(np.median(v)) for k,v in s5.items()}
print("S5 done")

print(json.dumps(R,indent=1))
json.dump(R,open("/home/claude/results_v3.json","w"))
