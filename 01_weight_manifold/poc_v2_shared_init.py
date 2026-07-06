"""PoC v2 - shared init (kills permutation symmetry), functional AE loss dominant."""
import torch, torch.nn as nn, numpy as np, json, time
torch.manual_seed(0); np.random.seed(0)

def make_net():
    return nn.Sequential(nn.Linear(1,16), nn.Tanh(),
                         nn.Linear(16,16), nn.Tanh(),
                         nn.Linear(16,1))

def flatten(net): return torch.cat([p.detach().flatten() for p in net.parameters()])
def set_flat(net, vec):
    i=0
    for p in net.parameters():
        n=p.numel(); p.data = vec[i:i+n].view_as(p).clone(); i+=n

def forward_flat(w, x):
    i0=0; h=x
    for fin,fout in [(1,16),(16,16),(16,1)]:
        Wl=w[i0:i0+fin*fout].view(fout,fin); i0+=fin*fout
        bl=w[i0:i0+fout]; i0+=fout
        h=h@Wl.T+bl
        if fout!=1: h=torch.tanh(h)
    return h

def task_data(A,phi,n): 
    x=torch.rand(n,1)*10-5; return x, A*torch.sin(x+phi)

# shared anchor init: pretrain one net on a mid-range task, all task nets start there
anchor = make_net()
opt = torch.optim.Adam(anchor.parameters(), lr=1e-2)
for _ in range(800):
    x,y = task_data(1.5, np.pi/2, 64)
    l=((anchor(x)-y)**2).mean(); opt.zero_grad(); l.backward(); opt.step()
w0 = flatten(anchor)
NPARAM=w0.numel()

t0=time.time(); NT=400; Ws=[]
for i in range(NT):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    net=make_net(); set_flat(net,w0)
    opt=torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(400):
        x,y=task_data(A,phi,64)
        l=((net(x)-y)**2).mean(); opt.zero_grad(); l.backward(); opt.step()
    Ws.append(flatten(net))
W=torch.stack(Ws)
print(f"dataset {W.shape} {time.time()-t0:.0f}s | spread: {W.std(0).mean():.3f} (v1 was much higher)")
mu,sd = W.mean(0), W.std(0)+1e-6
Wn=(W-mu)/sd

ZDIM=8
enc=nn.Sequential(nn.Linear(NPARAM,256),nn.SiLU(),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,ZDIM))
dec=nn.Sequential(nn.Linear(ZDIM,64),nn.SiLU(),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,NPARAM))
opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
probe=torch.linspace(-5,5,96).unsqueeze(1)
for ep in range(5000):
    idx=torch.randint(0,NT,(32,))
    Wr = dec(enc(Wn[idx]))
    loss_w=((Wr-Wn[idx])**2).mean()
    loss_f=0.
    for k in range(6):
        j=idx[k]
        with torch.no_grad(): y_true=forward_flat(W[j],probe)
        loss_f=loss_f+((forward_flat(Wr[k]*sd+mu,probe)-y_true)**2).mean()
    loss=0.2*loss_w+loss_f/6
    opt.zero_grad(); loss.backward(); opt.step()
print(f"AE: w-loss {loss_w.item():.4f}, f-loss {(loss_f/6).item():.4f}")

def decode(z): return dec(z)*sd+mu
def eval_dense(w,A,phi):
    x=torch.linspace(-5,5,400).unsqueeze(1)
    with torch.no_grad(): return ((forward_flat(w,x)-A*torch.sin(x+phi))**2).mean().item()

def adapt_ft(winit,xs,ys,steps=200,lr=1e-2):
    w=winit.clone().requires_grad_(True); o=torch.optim.Adam([w],lr=lr)
    for _ in range(steps):
        l=((forward_flat(w,xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return w.detach()

def adapt_z(xs,ys,steps=200,lr=5e-2):
    z=torch.zeros(1,ZDIM,requires_grad=True); o=torch.optim.Adam([z],lr=lr)
    for _ in range(steps):
        l=((forward_flat(decode(z)[0],xs)-ys)**2).mean(); o.zero_grad(); l.backward(); o.step()
    return decode(z.detach())[0]

NTEST=25; Ks=[3,5,10,20]
res={K:{"ft_anchor":[], "z":[]} for K in Ks}
for t in range(NTEST):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    for K in Ks:
        xs,ys=task_data(A,phi,K)
        res[K]["ft_anchor"].append(eval_dense(adapt_ft(w0,xs,ys),A,phi))
        res[K]["z"].append(eval_dense(adapt_z(xs,ys),A,phi))

# OOD amplitude
ood={"z":[],"ft_anchor":[]}
for t in range(10):
    A=np.random.uniform(4,5); phi=np.random.uniform(0,np.pi)
    xs,ys=task_data(A,phi,20)
    ood["z"].append(eval_dense(adapt_z(xs,ys),A,phi))
    ood["ft_anchor"].append(eval_dense(adapt_ft(w0,xs,ys),A,phi))

out={"K":{K:{m:float(np.median(v)) for m,v in res[K].items()} for K in Ks},
     "ood":{k:float(np.median(v)) for k,v in ood.items()}}
print(json.dumps(out,indent=1))
json.dump(out,open("/home/claude/results_v2.json","w"))
