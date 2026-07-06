"""
PoC: Latent Weight-Space Adaptation ("champ de poids implicite")
================================================================
1. Task family: y = A*sin(x + phi), A in [0.5,3.0], phi in [0,pi]
2. Train 400 MLPs (1-16-16-1) on 400 tasks -> weight dataset
3. Train generator G: z (dim 8) -> weights (AE over weight space)
4. Benchmark on 25 UNSEEN tasks, few-shot K in {5,10,20}:
   a) FT-full : fine-tune all weights from random init
   b) FT-mean : fine-tune all weights from mean-weight init
   c) Z-ADAPT : optimize z only (8 params), decoder frozen
5. Metrics: dense-grid test MSE, params updated, OOD expressivity limit
"""
import torch, torch.nn as nn, numpy as np, json, time
torch.manual_seed(0); np.random.seed(0)

def make_net():
    return nn.Sequential(nn.Linear(1,16), nn.Tanh(),
                         nn.Linear(16,16), nn.Tanh(),
                         nn.Linear(16,1))

NPARAM = sum(p.numel() for p in make_net().parameters())
print(f"params per net: {NPARAM}")

def flatten(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])

def set_flat(net, vec):
    i = 0
    for p in net.parameters():
        n = p.numel()
        p.data = vec[i:i+n].view_as(p).clone()
        i += n

def task_data(A, phi, n, xmin=-5, xmax=5):
    x = torch.rand(n,1)*(xmax-xmin)+xmin
    return x, A*torch.sin(x+phi)

def train_task(A, phi, steps=800, lr=1e-2):
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        x,y = task_data(A,phi,64)
        loss = ((net(x)-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, loss.item()

# ---------- 1) build weight dataset ----------
t0=time.time()
NT = 400
Ws, tasks = [], []
for i in range(NT):
    A = np.random.uniform(0.5,3.0); phi = np.random.uniform(0,np.pi)
    net,_ = train_task(A,phi)
    Ws.append(flatten(net)); tasks.append((A,phi))
W = torch.stack(Ws)
print(f"weight dataset: {W.shape}, {time.time()-t0:.0f}s")
mu, sd = W.mean(0), W.std(0)+1e-6
Wn = (W-mu)/sd

# ---------- 2) generator (AE over weight space) ----------
ZDIM = 8
enc = nn.Sequential(nn.Linear(NPARAM,256), nn.SiLU(), nn.Linear(256,64), nn.SiLU(), nn.Linear(64,ZDIM))
dec = nn.Sequential(nn.Linear(ZDIM,64), nn.SiLU(), nn.Linear(64,256), nn.SiLU(), nn.Linear(256,NPARAM))
opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3)
probe_x = torch.linspace(-5,5,128).unsqueeze(1)  # functional loss probe
tmp = make_net()
for ep in range(4000):
    idx = torch.randint(0,NT,(64,))
    z = enc(Wn[idx]); Wr = dec(z)
    loss_w = ((Wr-Wn[idx])**2).mean()
    # functional reconstruction loss on a few samples (crucial: weights are not the metric that matters, functions are)
    loss_f = 0.0
    for j in idx[:8]:
        set_flat(tmp, W[j]); y_true = tmp(probe_x).detach()
        wr = dec(enc(Wn[j:j+1]))[0]*sd+mu
        # functional forward with reconstructed weights (manual, differentiable wrt dec)
        i0=0; h=probe_x
        for lin in [(0,1,16),(1,16,16),(2,16,1)]:
            _,fin,fout = lin
            Wl = wr[i0:i0+fin*fout].view(fout,fin); i0+=fin*fout
            bl = wr[i0:i0+fout]; i0+=fout
            h = h@Wl.T + bl
            if fout!=1: h = torch.tanh(h)
        loss_f = loss_f + ((h-y_true)**2).mean()
    loss = loss_w + 0.05*loss_f/8
    opt.zero_grad(); loss.backward(); opt.step()
print(f"AE trained, w-loss {loss_w.item():.4f}")

def decode(z):  # z -> flat weights (differentiable)
    return dec(z)*sd+mu

def forward_flat(wflat, x):  # differentiable functional forward
    i0=0; h=x
    for fin,fout in [(1,16),(16,16),(16,1)]:
        Wl = wflat[i0:i0+fin*fout].view(fout,fin); i0+=fin*fout
        bl = wflat[i0:i0+fout]; i0+=fout
        h = h@Wl.T + bl
        if fout!=1: h = torch.tanh(h)
    return h

# ---------- 3) adaptation benchmark ----------
def eval_dense(wflat, A, phi):
    x = torch.linspace(-5,5,400).unsqueeze(1)
    with torch.no_grad():
        return ((forward_flat(wflat,x)-A*torch.sin(x+phi))**2).mean().item()

def adapt_ft(w0, xs, ys, steps=200, lr=1e-2):
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = ((forward_flat(w,xs)-ys)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach()

def adapt_z(xs, ys, steps=200, lr=5e-2):
    z = torch.zeros(1,ZDIM, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        w = decode(z)[0]
        loss = ((forward_flat(w,xs)-ys)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return decode(z.detach())[0]

NTEST = 25
results = {k:{"ft_rand":[], "ft_mean":[], "z":[]} for k in [5,10,20]}
w_mean = W.mean(0)
t0=time.time()
for t in range(NTEST):
    A = np.random.uniform(0.5,3.0); phi = np.random.uniform(0,np.pi)
    for K in [5,10,20]:
        xs,ys = task_data(A,phi,K)
        w_r = adapt_ft(flatten(make_net()), xs, ys)
        w_m = adapt_ft(w_mean, xs, ys)
        w_z = adapt_z(xs, ys)
        results[K]["ft_rand"].append(eval_dense(w_r,A,phi))
        results[K]["ft_mean"].append(eval_dense(w_m,A,phi))
        results[K]["z"].append(eval_dense(w_z,A,phi))
print(f"benchmark done {time.time()-t0:.0f}s")

# ---------- 4) OOD expressivity test (limit of the manifold) ----------
ood = {"z":[], "ft_mean":[]}
for t in range(10):
    A = np.random.uniform(4.0,5.0); phi = np.random.uniform(0,np.pi)  # amplitude out of training range
    xs,ys = task_data(A,phi,20)
    ood["z"].append(eval_dense(adapt_z(xs,ys),A,phi))
    ood["ft_mean"].append(eval_dense(adapt_ft(w_mean,xs,ys),A,phi))

out = {"K": {}, "ood": {k: float(np.median(v)) for k,v in ood.items()}}
for K in [5,10,20]:
    out["K"][K] = {m: float(np.median(v)) for m,v in results[K].items()}
print(json.dumps(out, indent=1))
json.dump(out, open("/home/claude/results.json","w"))
