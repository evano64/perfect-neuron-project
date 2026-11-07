import sys,os,json,math,time,random
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
os.environ["PYTORCH_HIP_ALLOC_CONF"]="expandable_segments:True"

import torch
from transformers import AutoTokenizer
from model_def import OscModel

# ── Config ──
DEVICE='cuda'; BLOCK=256; TEMP=0.8; TOP_K=50; GEN_LEN=80
tok=AutoTokenizer.from_pretrained('camembert-base')

FIXED_PROMPTS=[
    "La signification de la vie est","Il était une fois","Le chat est assis sur",
    "Bonjour tout le monde","L'intelligence artificielle","La science est importante car",
    "Dans un futur lointain","Le président de la république a déclaré",
    "Les enfants jouent dans le jardin","La musique est un art qui",
    "Le climat change et les conséquences","Mon meilleur souvenir est",
    "La cuisine française est reconnue","L'éducation est fondamentale pour",
    "Le sport nous apprend","La technologie transforme notre société",
    "Les oiseaux chantent le matin","La philosophie cherche à comprendre",
    "Un bon livre doit","L'amitié est importante parce que",
    "La ville de Paris est connue","Le problème de l'eau potable",
    "Les animaux marins vivent dans","La peinture impressionniste a changé",
    "Le système solaire contient","Les voyages permettent de",
    "La liberté d'expression est","Le chocolat est fabriqué à partir",
    "Les maths sont utiles dans","La pluie tombe lorsque",
    "Le football est le sport le plus","L'océan couvre pourcentage de la Terre",
    "La musique classique est composée par","Les abeilles sont importantes pour",
    "Le bois est un matériau","La cuisine japonaise utilise beaucoup de",
    "Le temps passe vite quand","Les étoiles sont visibles la nuit",
    "Le jardinage nécessite de la patience","La gravité maintient les objets",
    "Les livres nous permettent de","Le pain est un aliment de base",
    "La mer Méditerranée est située","Le théâtre est un art vivant",
    "Les pommes poussent sur les arbres","La géographie étudie les territoires",
    "Le café est consommé dans le monde entier","La nuit tombait sur la ville",
    "Le silence est parfois nécessaire pour","La dentelle est fabriquée à la main",
]

@torch.no_grad()
def generate(model,prompt,max_new=GEN_LEN,temp=TEMP,top_k=TOP_K):
    model.eval()
    ids=tok.encode(prompt,return_tensors='pt').to(DEVICE)
    prompt_len=ids.shape[1]
    for _ in range(max_new):
        inp=ids[:,-BLOCK:]
        pos=torch.arange(inp.shape[1],device=DEVICE).unsqueeze(0)
        logits=model(inp)[:,-1,:]
        logits=logits/temp
        if top_k>0:
            v,_=torch.topk(logits,min(top_k,logits.size(-1)))
            logits[logits<v[-1:]]=-float('inf')
        probs=torch.softmax(logits,dim=-1)
        nxt=torch.multinomial(probs,1)
        ids=torch.cat([ids,nxt],dim=1)
        if nxt.item()==6:  # EOS
            break
    gen_ids=ids[0,prompt_len:].cpu().tolist()
    return tok.decode(gen_ids,skip_special_tokens=True)

def compute_metrics(texts):
    all_tokens=[]; bigrams=[]; total_len=0
    for t in texts:
        ids=tok.encode(t,add_special_tokens=False)
        all_tokens.extend(ids)
        total_len+=len(ids)
        for i in range(len(ids)-1):
            bigrams.append((ids[i],ids[i+1]))
    if not all_tokens:
        return {'unique':0,'bigrams':0,'entropy':0,'avg_len':0}
    from collections import Counter
    tc=Counter(all_tokens)
    unique_pct=len(tc)/len(all_tokens)*100
    bc=Counter(bigrams)
    bigrams_pct=len(bc)/max(len(bigrams),1)*100
    # entropy
    total=len(all_tokens)
    entropy=0
    for c in tc.values():
        p=c/total
        if p>0: entropy-=p*math.log2(p)
    return {'unique':round(unique_pct,1),'bigrams':round(bigrams_pct,1),'entropy':round(entropy,2),'avg_len':round(total_len/len(texts),1)}

def load_model(ckpt_path):
    m=OscModel().to(DEVICE)
    sd=torch.load(ckpt_path,map_location=DEVICE)
    if 'model' in sd:
        sd=sd['model']
    m.load_state_dict(sd,strict=False)
    m.eval()
    return m

print("="*70)
print("COMPARISON: LeCarnet (best.pt) vs RL (rl_latest.pt)")
print("="*70)

# Load models
print("\nLoading LeCarnet model...")
lec=load_model('results_lecarnet/best.pt')
print(f"  Loaded: {sum(p.numel() for p in lec.parameters())/1e6:.1f}M params")

print("Loading RL model...")
rl=load_model('results_rl/rl_latest.pt')
print(f"  Loaded: {sum(p.numel() for p in rl.parameters())/1e6:.1f}M params")

results={'lecarnet':[],'rl':[]}

subset=FIXED_PROMPTS[:10]  # 10 prompts only for speed
print(f"\nGenerating {len(subset)} prompts x 2 models...")
for i,prompt in enumerate(subset):
    t0=time.time()
    g_lec=generate(lec,prompt)
    t_lec=time.time()-t0
    t0=time.time()
    g_rl=generate(rl,prompt)
    t_rl=time.time()-t0
    results['lecarnet'].append({'prompt':prompt,'gen':g_lec,'time':t_lec})
    results['rl'].append({'prompt':prompt,'gen':g_rl,'time':t_rl})
    if (i+1)%10==0:
        print(f"  [{i+1}/{len(FIXED_PROMPTS)}] done")

# Compute metrics
for model_name in ['lecarnet','rl']:
    gens=[r['gen'] for r in results[model_name]]
    m=compute_metrics(gens)
    avg_time=sum(r['time'] for r in results[model_name])/len(results[model_name])
    results[model_name+'_metrics']=m
    results[model_name+'_avg_time']=avg_time

print("\n" + "="*70)
print("METRICS COMPARISON")
print("="*70)
print(f"{'Metric':<20} {'LeCarnet':>12} {'RL':>12} {'Delta':>12}")
print("-"*56)
for k in ['unique','bigrams','entropy','avg_len']:
    v1=results['lecarnet_metrics'][k]
    v2=results['rl_metrics'][k]
    d=v1-v2
    sign='+' if d>0 else ''
    print(f"{k:<20} {v1:>12} {v2:>12} {sign}{d:>11}")
print(f"{'avg_gen_time(s)':<20} {results['lecarnet_avg_time']:>12.2f} {results['rl_avg_time']:>12.2f}")

# Show examples
print("\n" + "="*70)
print("GENERATION EXAMPLES (first 10)")
print("="*70)
for i in range(min(10,len(subset))):
    p=subset[i]
    g1=results['lecarnet'][i]['gen'][:120]
    g2=results['rl'][i]['gen'][:120]
    print(f"\n[{i+1}] P: {p}")
    print(f"  LC: {g1}")
    print(f"  RL: {g2}")

# Save full results
with open('results_comparison_lecarnet_rl.json','w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
print(f"\nFull results saved to results_comparison_lecarnet_rl.json")
