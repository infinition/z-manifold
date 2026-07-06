"""Telecharge les 196 adaptateurs LoRA flan-t5-large de LoraHub (fichiers adapter_* seulement)."""
import os, sys, time
from huggingface_hub import list_models, snapshot_download

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")
os.makedirs(DEST, exist_ok=True)

repos = sorted(m.id for m in list_models(author="lorahub", limit=None) if "flan_t5_large" in m.id)
print(f"{len(repos)} repos a telecharger vers {DEST}", flush=True)

ok, fail = 0, []
t0 = time.time()
for i, rid in enumerate(repos):
    name = rid.split("/", 1)[1]
    tgt = os.path.join(DEST, name)
    if os.path.exists(os.path.join(tgt, "adapter_model.bin")):
        ok += 1
        continue
    try:
        snapshot_download(rid, allow_patterns=["adapter_*"], local_dir=tgt)
        ok += 1
    except Exception as e:
        fail.append((rid, str(e)[:120]))
    if (i + 1) % 20 == 0:
        print(f"{i+1}/{len(repos)} ok={ok} fail={len(fail)} {time.time()-t0:.0f}s", flush=True)

print(f"FIN ok={ok} fail={len(fail)} {time.time()-t0:.0f}s", flush=True)
for rid, e in fail:
    print("ECHEC", rid, e, flush=True)
