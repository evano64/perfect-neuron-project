"""
RL Generate — Génère 200 prompts × 8 complétions pour notation par juge externe
Charge le checkpoint, génère des complétions, sauvegarde en JSONL
"""
import os,json,random,math,time,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULT_DIR='results_5b'
RL_DATA='rl_data'
os.makedirs(RL_DATA,exist_ok=True)

device=torch.device('cuda')

# ── Model (même archi que train_fr_100m_5b.py) ──────────────────────
class KuramotoOsc(nn.Module):
    def __init__(self,d_model,n_osc,chunk=32):
        super().__init__()
        self.n_osc,self.chunk=n_osc,chunk
        self.omega=nn.Parameter(torch.linspace(0.0,math.pi,n_osc))
        self.decay_logit=nn.Parameter(torch.full((n_osc,),3.0));self.decay_logit.requires_grad=False
        self.B_re=nn.Parameter(torch.randn(n_osc,d_model)/d_model**0.5)
        self.B_im=nn.Parameter(torch.randn(n_osc,d_model)/d_model**0.5)
        self.K_raw=nn.Parameter(torch.randn(n_osc,n_osc)/n_osc**0.5*0.1)
        self.C=nn.Parameter(torch.randn(d_model,n_osc*3)/(n_osc*3)**0.5)
    def _K(self):return self.K_raw-self.K_raw.t()
    def forward(self,x,h_carry=None):
        Bsz,T,_=x.shape
        decay=torch.sigmoid(self.decay_logit);a=torch.polar(decay.float(),self.omega.float());K=self._K()
        u=(torch.einsum('btd,od->bto',x.float(),self.B_re.float())+1j*torch.einsum('btd,od->bto',x.float(),self.B_im.float()))
        hc=h_carry if h_carry is not None else torch.zeros(Bsz,self.n_osc,dtype=torch.cfloat,device=x.device)
        la=torch.log(a);o=[]
        for s in range(0,T,self.chunk):
            e=min(s+self.chunk,T);L=e-s;uc=u[:,s:e,:];ti=torch.arange(L,device=x.device,dtype=torch.float32)
            ap=torch.exp(la[None]*ti[:,None]);an=torch.exp(-la[None]*ti[:,None])
            Hc=ap[None]*torch.cumsum(an[None]*uc,dim=1)
            Hc=Hc+0.1*torch.einsum('ij,btj->bti',K.to(Hc.dtype),Hc)
            Hc=Hc+torch.exp(la[None]*(ti[:,None]+1))[None]*hc[:,None,:]
            o.append(Hc);hc=Hc[:,-1,:]
        H=torch.cat(o,dim=1)
        feats=torch.cat([H.real,H.imag,H.abs()],dim=-1)
        return (feats@self.C.t().float()).to(x.dtype),hc

class OscModel(nn.Module):
    def __init__(self,vocab=32005,d_model=1152,n_osc=1536,n_layers=4,n_heads=16,max_len=256,dropout=0.1):
        super().__init__()
        self.d_model=d_model;self.n_heads=n_heads;self.head_dim=d_model//n_heads
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.dropout=nn.Dropout(dropout)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'osc':KuramotoOsc(d_model,n_osc),
                'norm1':nn.LayerNorm(d_model),'norm2':nn.LayerNorm(d_model),
                'qkv':nn.Linear(d_model,3*d_model,bias=False),'proj':nn.Linear(d_model,d_model,bias=False),
                'norm3':nn.LayerNorm(d_model),'dropout':nn.Dropout(dropout),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos);x=self.dropout(x)
        for blk in self.layers:
            o,hc=blk['osc'](blk['norm1'](x),None);x=x+blk['dropout'](o)
            h=blk['norm3'](x);B2,T2,D=h.shape
            qkv=blk['qkv'](h).reshape(B2,T2,3,self.n_heads,self.head_dim);q,k,v=qkv.unbind(2)
            a=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),dropout_p=0.0,is_causal=True)
            x=x+blk['proj'](blk['dropout'](a.permute(0,2,1,3).reshape(B2,T2,D)))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

# ── Prompts ──────────────────────────────────────────────────────────
FIXED_PROMPTS=[
    "La signification de la vie est",
    "Il était une fois",
    "Le chat est assis sur",
    "Bonjour tout le monde",
    "L'intelligence artificielle",
    "La science est importante car",
    "Dans un futur lointain",
    "Le président de la république a déclaré",
    "Les enfants jouent dans le jardin",
    "La musique est un art qui",
    "Le climat change et les conséquences",
    "Mon meilleur souvenir est",
    "La cuisine française est reconnue",
    "L'éducation est fondamentale pour",
    "Le sport nous apprend",
    "La technologie transforme notre société",
    "Les oiseaux chantent le matin",
    "La philosophie cherche à comprendre",
    "Un bon livre doit",
    "L'amitié est importante parce que",
    "La ville de Paris est connue",
    "Le problème de l'eau potable",
    "Les animaux marins vivent dans",
    "La peinture impressionniste a changé",
    "Le système solaire contient",
    "Les voyages permettent de",
    "La liberté d'expression est",
    "Le chocolat est fabriqué à partir",
    "Les maths sont utiles dans",
    "La pluie tombe lorsque",
    "Le football est le sport le plus",
    "L'océan couvre pourcentage de la Terre",
    "La musique classique est composée par",
    "Les abeilles sont importantes pour",
    "Le bois est un matériau",
    "La cuisine japonaise utilise beaucoup de",
    "Le temps passe vite quand",
    "Les étoiles sont visibles la nuit",
    "La dentelle est fabriquée à la main",
    "Le jardinage nécessite de la patience",
    "La gravité maintient les objets",
    "Les livres nous permettent de",
    "Le pain est un aliment de base",
    "La mer Méditerranée est située",
    "Le théâtre est un art vivant",
    "Les pommes poussent sur les arbres",
    "La géographie étudie les territoires",
    "Le café est consommé dans le monde entier",
    "La nuit tombait sur la ville",
    "Le silence est parfois nécessaire pour",
]

def sample_random_prompts(mm,n=150,block=128):
    """Échantillonne des prompts aléatoires du corpus (premiers tokens d'un bloc)"""
    prompts=[]
    max_start=len(mm)-block-1
    for _ in range(n):
        start=random.randint(0,max_start)
        chunk=mm[start:start+block].astype(np.int64)
        prompts.append(chunk.tolist())
    return prompts

# ── Main ─────────────────────────────────────────────────────────────
if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('camembert-base')

    # Load model
    model=OscModel().to(device).to(torch.bfloat16)
    ckpt=torch.load(f'{RESULT_DIR}/best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Model loaded: step {ckpt["step"]}, loss {ckpt["loss"]:.4f}')

    # Load dataset for random prompts
    mm=np.load('finefrench_5b_tokens.npy',mmap_mode='r')
    print(f'Dataset: {len(mm):,} tokens')

    # Generate random prompt token IDs
    random_ids=sample_random_prompts(mm,n=150)

    all_prompts=[]
    # 50 fixed prompts (tokenized)
    for p in FIXED_PROMPTS:
        ids=tok.encode(p,return_tensors='pt')[0].tolist()
        all_prompts.append({'type':'fixed','text':p,'ids':ids})
    # 150 random prompts
    for i,ids in enumerate(random_ids):
        text=tok.decode(ids[:20],skip_special_tokens=True)
        all_prompts.append({'type':'random','text':text,'ids':ids})

    print(f'Total prompts: {len(all_prompts)}')

    # Generation params
    import sys
    N_PROMPTS=int(sys.argv[1]) if len(sys.argv)>1 else 100
    K=4
    MAX_NEW=64
    PROMPT_MAX=128
    TEMP=0.8
    TOP_P=0.95

    # Trim to N prompts
    all_prompts=all_prompts[:N_PROMPTS]

    out_path=f'{RL_DATA}/generations.jsonl'
    t_start=time.time()
    with open(out_path,'w',encoding='utf-8') as f:
        for pi,prompt_info in enumerate(all_prompts):
            ids=prompt_info['ids'][:PROMPT_MAX]
            prompt_tensor=torch.tensor([ids],device=device,dtype=torch.long)

            completions=[]
            for k in range(K):
                with torch.no_grad():
                    out=prompt_tensor.clone()
                    for step in range(MAX_NEW):
                        with torch.amp.autocast('cuda',enabled=False):
                            logits=model(out).float()
                        nxt=logits[:,-1,:]/TEMP
                        sorted_logits,sorted_idx=torch.sort(nxt,descending=True)
                        cumprobs=torch.cumsum(F.softmax(sorted_logits,dim=-1),dim=-1)
                        mask=cumprobs-TOP_P>0
                        sorted_logits[mask]=-1e9
                        probs=F.softmax(sorted_logits,dim=-1)
                        idx=torch.multinomial(probs,1)
                        next_token=sorted_idx.gather(-1,idx)
                        out=torch.cat([out,next_token],dim=1)
                        if next_token.item()==2:
                            break
                    gen_ids=out[0,len(ids):].tolist()
                    gen_text=tok.decode(gen_ids,skip_special_tokens=True)
                    completions.append(gen_text)

            entry={
                'id':pi,
                'prompt':prompt_info['text'],
                'prompt_ids_len':len(ids),
                'completions':completions,
            }
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            f.flush()

            elapsed=time.time()-t_start
            eta=elapsed/(pi+1)*N_PROMPTS
            avg_unique=0
            display=prompt_info['text'][:35]
            try:print(f'  [{pi+1:3d}/{N_PROMPTS}] "{display}" | {elapsed:.0f}s elapsed | ETA {eta:.0f}s',flush=True)
            except:print(f'  [{pi+1:3d}/{N_PROMPTS}] (unicode) | {elapsed:.0f}s elapsed | ETA {eta:.0f}s',flush=True)

    elapsed=time.time()-t_start
    print(f'\nDone: {out_path} ({elapsed:.0f}s)')
    print(f'Total: {N_PROMPTS} prompts x {K} completions = {N_PROMPTS*K} generations')
