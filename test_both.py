import sys,os,time
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import torch
from transformers import AutoTokenizer
from model_def import OscModel

DEVICE='cuda'
tok=AutoTokenizer.from_pretrained('camembert-base')

with open('quick_test.log','w',encoding='utf-8') as f:
    def log(m):f.write(m+'\n');f.flush()
    
    path='results_rl/rl_latest.pt'
    log("Loading RL model...")
    m=OscModel().to(DEVICE)
    sd=torch.load(path,map_location=DEVICE)
    if 'model' in sd: sd=sd['model']
    m.load_state_dict(sd,strict=False); del sd; m.eval()
    log(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    ids=tok.encode("Il etait une fois",return_tensors='pt').to(DEVICE)
    log(f"Input: {ids.shape}")
    
    t0=time.time()
    with torch.no_grad(): out=m(ids)
    log(f"Forward: {time.time()-t0:.2f}s")
    
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
    log(f"10 tokens: {time.time()-t0:.2f}s")
    log(f"Text: {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)}")
    
    del m; torch.cuda.empty_cache()
    
    # Now LeCarnet
    path='results_lecarnet/best.pt'
    log("\nLoading LeCarnet model...")
    m=OscModel().to(DEVICE)
    sd=torch.load(path,map_location=DEVICE)
    if 'model' in sd: sd=sd['model']
    m.load_state_dict(sd,strict=False); del sd; m.eval()
    log(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    ids=tok.encode("Il etait une fois",return_tensors='pt').to(DEVICE)
    t0=time.time()
    with torch.no_grad(): out=m(ids)
    log(f"Forward: {time.time()-t0:.2f}s")
    
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
    log(f"10 tokens: {time.time()-t0:.2f}s")
    log(f"Text: {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)}")
    
    log("\nDONE")
