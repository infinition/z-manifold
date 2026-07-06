"""Superposition memory: complex phasors vs real bipolar, equal real-param budget.
Binding: elementwise product (FHRR vs MAP). Recall: unbind + nearest-neighbor cleanup.
Test A: capacity (accuracy vs N items stored in ONE fixed-size trace)
Test B: robustness (corrupt fraction of trace) with/without redundant encoding R=3
"""
import numpy as np, json
rng=np.random.default_rng(0)
DC=256          # complex dims
DR=512          # real dims  -> same real-parameter budget
VOCAB=1000; TRIALS=5

def cvec(n,d): return np.exp(1j*rng.uniform(0,2*np.pi,(n,d)))          # unit phasors
def rvec(n,d): return rng.choice([-1.,1.],(n,d))                        # bipolar

def run_capacity(Ns):
    acc={"cplx":[], "real":[]}
    for N in Ns:
        ac=ar=0
        for t in range(TRIALS):
            # complex
            K,V=cvec(N,DC),cvec(VOCAB,DC); idx=rng.integers(0,VOCAB,N)
            M=(K*V[idx]).sum(0)                        # superposed trace
            probe=rng.integers(0,N,20)
            for p in probe:
                v=M*np.conj(K[p])                      # unbind (unitary)
                sim=np.real(V@np.conj(v))              # cleanup
                ac+= (sim.argmax()==idx[p])
            # real
            K,V=rvec(N,DR),rvec(VOCAB,DR); idx=rng.integers(0,VOCAB,N)
            M=(K*V[idx]).sum(0)
            for p in probe:
                v=M*K[p]                               # unbind (self-inverse)
                sim=V@v
                ar+= (sim.argmax()==idx[p])
        acc["cplx"].append(ac/(TRIALS*20)); acc["real"].append(ar/(TRIALS*20))
    return acc

Ns=[10,25,50,75,100,150,200,300,400]
cap=run_capacity(Ns)

# Test B: corruption robustness at N=100, complex, R=1 vs R=3 redundant traces folded in one field of same size
def run_robust(fracs,N=100,R=3):
    out={"R1":[], "R3":[]}
    for f in fracs:
        a1=a3=0
        for t in range(TRIALS):
            V=cvec(VOCAB,DC); idx=rng.integers(0,VOCAB,N)
            K1=cvec(N,DC)
            M1=(K1*V[idx]).sum(0)
            Ks=[cvec(N,DC) for _ in range(R)]
            M3=sum((Ks[r]*V[idx]).sum(0) for r in range(R))   # 3 redundant encodings, SAME field size
            for M,mode in [(M1,1),(M3,3)]:
                Mc=M.copy(); m=rng.random(DC)<f; Mc[m]=0       # erase fraction f of components
                probe=rng.integers(0,N,20)
                good=0
                for p in probe:
                    if mode==1: v=Mc*np.conj(K1[p]); sim=np.real(V@np.conj(v))
                    else:
                        sim=0
                        for r in range(R): sim=sim+np.real(V@np.conj(Mc*np.conj(Ks[r][p])))  # vote
                    good+= (sim.argmax()==idx[p])
                if mode==1: a1+=good
                else: a3+=good
        out["R1"].append(a1/(TRIALS*20)); out["R3"].append(a3/(TRIALS*20))
    return out

fracs=[0.0,0.2,0.4,0.6,0.8]
rob=run_robust(fracs)
res={"Ns":Ns,"capacity":cap,"fracs":fracs,"robust":rob}
print(json.dumps(res,indent=1)); json.dump(res,open("qmem_results.json","w"))
