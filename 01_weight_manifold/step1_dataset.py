import torch, torch.nn as nn, numpy as np, time
torch.manual_seed(0); np.random.seed(0)
def make_net(): return nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,16),nn.Tanh(),nn.Linear(16,1))
def flatten(net): return torch.cat([p.detach().flatten() for p in net.parameters()])
def set_flat(net,vec):
    i=0
    for p in net.parameters():
        n=p.numel(); p.data=vec[i:i+n].view_as(p).clone(); i+=n
def task_data(A,phi,n): x=torch.rand(n,1)*10-5; return x, A*torch.sin(x+phi)
anchor=make_net(); opt=torch.optim.Adam(anchor.parameters(),lr=1e-2)
for _ in range(800):
    x,y=task_data(1.5,np.pi/2,64); l=((anchor(x)-y)**2).mean(); opt.zero_grad(); l.backward(); opt.step()
w0=flatten(anchor)
t0=time.time(); NT=300; Ws=[]
for i in range(NT):
    A=np.random.uniform(0.5,3.0); phi=np.random.uniform(0,np.pi)
    net=make_net(); set_flat(net,w0); o=torch.optim.Adam(net.parameters(),lr=5e-3)
    for _ in range(350):
        x,y=task_data(A,phi,64); l=((net(x)-y)**2).mean(); o.zero_grad(); l.backward(); o.step()
    Ws.append(flatten(net))
W=torch.stack(Ws)
torch.save({"W":W,"w0":w0},"weights_ds.pt")
print(f"done {time.time()-t0:.0f}s")
