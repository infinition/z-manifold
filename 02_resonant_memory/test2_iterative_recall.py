"""Iterative resonant recall: successive interference cancellation.
Passes over all keys; confident decodes are subtracted from the trace."""
import numpy as np, json
rng=np.random.default_rng(1)
DC=256; DR=512; VOCAB=1000; TRIALS=5

def cvec(n,d): return np.exp(1j*rng.uniform(0,2*np.pi,(n,d)))
def rvec(n,d): return rng.choice([-1.,1.],(n,d))

def iter_recall_cplx(N, passes=5):
    K=cvec(N,DC); V=cvec(VOCAB,DC); idx=rng.integers(0,VOCAB,N)
    M=(K*V[idx]).sum(0); dec=np.full(N,-1)
    for p in range(passes):
        # confidence-ranked pass
        cand=[]
        for i in range(N):
            if dec[i]>=0: continue
            v=M*np.conj(K[i]); sim=np.real(V@np.conj(v))/DC
            j=sim.argmax(); s=np.sort(sim)
            margin=s[-1]-s[-2]
            cand.append((margin,i,j))
        cand.sort(reverse=True)
        for margin,i,j in cand[:max(1,len(cand)//3)]:  # subtract top-third most confident
            dec[i]=j; M=M-K[i]*V[j]
    return (dec==idx).mean()

def iter_recall_real(N, passes=5):
    K=rvec(N,DR); V=rvec(VOCAB,DR); idx=rng.integers(0,VOCAB,N)
    M=(K*V[idx]).sum(0).astype(float); dec=np.full(N,-1)
    for p in range(passes):
        cand=[]
        for i in range(N):
            if dec[i]>=0: continue
            v=M*K[i]; sim=(V@v)/DR
            j=sim.argmax(); s=np.sort(sim)
            cand.append((s[-1]-s[-2],i,j))
        cand.sort(reverse=True)
        for margin,i,j in cand[:max(1,len(cand)//3)]:
            dec[i]=j; M=M-K[i]*V[j]
    return (dec==idx).mean()

out={"N":[], "cplx_iter":[], "real_iter":[]}
for N in [50,100,150,200,300,400]:
    ac=np.mean([iter_recall_cplx(N) for _ in range(TRIALS)])
    ar=np.mean([iter_recall_real(N) for _ in range(TRIALS)])
    out["N"].append(N); out["cplx_iter"].append(round(float(ac),3)); out["real_iter"].append(round(float(ar),3))
    print(N, ac, ar, flush=True)
json.dump(out,open("qmem2_results.json","w"))
