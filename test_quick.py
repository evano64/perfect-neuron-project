import sys,os,time
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import torch
from transformers import AutoTokenizer
from model_def import OscModel

DEVICE='cuda'
tok=AutoTokenizer.from_pretrained('camembert-base')

for name,path in [('RL','results_rl/rl_latest.pt'),('LeCarnet','results_lecarnet/best.pt')]:
    torch.cuda.empty_cache()
    print(f"\n--- {name} ---",flush=True)
    m=OscModel().to(DEVICE)
    sd=torch.load(path,map_location=DEVICE)
    if 'model' in sd: sd=sd['model']
    m.load_state_dict(sd,strict=False); del sd
    m.eval()
    
    ids=tok.encode("Il était une fois",return_tensors='pt').to(DEVICE)
    print(f"  Input shape: {ids.shape}",flush=True)
    t0=time.time()
    with torch.no_grad():
        out=m(ids)
    print(f"  Forward: {time.time()-t0:.2f}s, output shape: {out.shape}",flush=True)
    
    # Generate 10 tokens
    t0=time.time()
    with torch.no_grad():
        for i in range(10):
            inp=ids[:,-256:]
            logits=m(inp)[:,-1,:]/0.8
            v,_=torch.topk(logits,50)
            logits[logits<v[-1:]]=-float('inf')
            nxt=torch.multinomial(torch.softmax(logits,-1),1)
            ids=torch.cat([ids,nxt],1)
    print(f"  10 tokens: {time.time()-t0:.2f}s",flush=True)
    print(f"  Generated: {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)}",flush=True)
    
    del m; torch.cuda.empty_cache()

print("\nDONE",flush=True)
