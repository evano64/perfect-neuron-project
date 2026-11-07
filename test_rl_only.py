import sys,os,time
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import torch
from transformers import AutoTokenizer
from model_def import OscModel

DEVICE='cuda'
tok=AutoTokenizer.from_pretrained('camembert-base')

path='results_rl/rl_latest.pt'
print("Loading RL model...",flush=True)
m=OscModel().to(DEVICE)
sd=torch.load(path,map_location=DEVICE)
if 'model' in sd: sd=sd['model']
m.load_state_dict(sd,strict=False); del sd; m.eval()
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB",flush=True)

ids=tok.encode("Il etait une fois",return_tensors='pt').to(DEVICE)
print(f"Input: {ids.shape}",flush=True)

t0=time.time()
with torch.no_grad():
    out=m(ids)
print(f"Forward: {time.time()-t0:.2f}s shape={out.shape}",flush=True)

t0=time.time()
with torch.no_grad():
    for i in range(10):
        inp=ids[:,-256:]
        logits=m(inp)[:,-1,:]/0.8
        k_val=min(50,logits.size(-1))
        v,_=torch.topk(logits,k_val)
        logits[logits<v[0,-1]]=-float('inf')
        nxt=torch.multinomial(torch.softmax(logits,-1),1)
        ids=torch.cat([ids,nxt],1)
print(f"10 tokens: {time.time()-t0:.2f}s",flush=True)
print(f"Text: {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)}",flush=True)
print("OK",flush=True)
