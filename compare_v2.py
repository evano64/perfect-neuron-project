import sys,os,json,math,time
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"

import torch
from transformers import AutoTokenizer
from model_def import OscModel

DEVICE='cuda'; BLOCK=256; TEMP=0.8; TOP_K=50; GEN_LEN=60
tok=AutoTokenizer.from_pretrained('camembert-base')

PROMPTS=[
    "La signification de la vie est","Il était une fois","Le chat est assis sur",
    "Bonjour tout le monde","L'intelligence artificielle","La science est importante car",
    "Dans un futur lointain","Les enfants jouent dans le jardin",
    "La musique est un art qui","Mon meilleur souvenir est",
]

@torch.no_grad()
def generate(model,prompt):
    model.eval()
    ids=tok.encode(prompt,return_tensors='pt').to(DEVICE)
    plen=ids.shape[1]
    for _ in range(GEN_LEN):
        inp=ids[:,-BLOCK:]
        logits=model(inp)[:,-1,:]/TEMP
        if TOP_K>0:
            k_val=min(TOP_K,logits.size(-1))
            v,_=torch.topk(logits,k_val)
            logits[logits<v[0,-1]]=-float('inf')
        nxt=torch.multinomial(torch.softmax(logits,-1),1)
        ids=torch.cat([ids,nxt],1)
        if nxt.item()==6: break
    return tok.decode(ids[0,plen:].cpu().tolist(),skip_special_tokens=True)

def metrics(texts):
    from collections import Counter
    all_t=[]; bigrams=[]
    for t in texts:
        ids=tok.encode(t,add_special_tokens=False)
        all_t.extend(ids)
        for i in range(len(ids)-1): bigrams.append((ids[i],ids[i+1]))
    if not all_t: return {}
    tc=Counter(all_t); bc=Counter(bigrams)
    total=len(all_t)
    entropy=-sum(c/total*math.log2(c/total) for c in tc.values() if c>0)
    return {
        'unique%': round(len(tc)/total*100,1),
        'bigrams%': round(len(bc)/max(len(bigrams),1)*100,1),
        'entropy': round(entropy,2),
        'avg_len': round(total/len(texts),1),
    }

def load_model(path):
    m=OscModel().to(DEVICE)
    sd=torch.load(path,map_location=DEVICE)
    if 'model' in sd: sd=sd['model']
    m.load_state_dict(sd,strict=False)
    del sd
    m.eval()
    return m

print("="*60)
print("COMPARISON: LeCarnet vs RL (10 prompts)")
print("="*60)

all_results={}
for name,path in [('RL','results_rl/rl_latest.pt'),('LeCarnet','results_lecarnet/best.pt')]:
    print(f"\n--- {name} ---")
    print(f"Loading {path}...",flush=True)
    torch.cuda.empty_cache()
    model=load_model(path)
    mem=torch.cuda.memory_allocated()/1e9
    print(f"  GPU: {mem:.1f} GB",flush=True)

    gens=[]; times=[]
    for i,p in enumerate(PROMPTS):
        t0=time.time()
        g=generate(model,p)
        dt=time.time()-t0
        gens.append(g); times.append(dt)
        print(f"  [{i+1}/10] {dt:.1f}s | {p[:30]}... -> {g[:60]}...",flush=True)

    m=metrics(gens)
    avg_t=sum(times)/len(times)
    m['avg_gen_s']=round(avg_t,2)
    all_results[name]={'metrics':m,'generations':[{'prompt':p,'gen':g,'time':t} for p,g,t in zip(PROMPTS,gens,times)]}
    print(f"  Metrics: {m}",flush=True)

    # free GPU
    del model
    torch.cuda.empty_cache()

# Compare
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Metric':<15} {'RL':>10} {'LeCarnet':>10}")
print("-"*35)
for k in all_results['RL']['metrics']:
    v1=all_results['RL']['metrics'][k]
    v2=all_results['LeCarnet']['metrics'][k]
    print(f"{k:<15} {v1:>10} {v2:>10}")

print("\n--- Examples ---")
for i in range(5):
    p=PROMPTS[i]
    g1=all_results['RL']['generations'][i]['gen'][:100]
    g2=all_results['LeCarnet']['generations'][i]['gen'][:100]
    print(f"\n[{i+1}] {p}")
    print(f"  RL: {g1}")
    print(f"  LC: {g2}")

with open('results_comparison.json','w',encoding='utf-8') as f:
    json.dump(all_results,f,ensure_ascii=False,indent=2)
print("\nSaved to results_comparison.json")
