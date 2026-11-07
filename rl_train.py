"""
RL Train — RAFT (best-of-N) + ancre KL + mix pretraining
Entraîne sur les données notées par le juge externe
"""
import os,json,random,math,time,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULT_DIR='results_rl'
RL_DATA='rl_data'
PRETRAIN_DATA='finefrench_5b_tokens.npy'
os.makedirs(RESULT_DIR,exist_ok=True)
LF=open(f'{RESULT_DIR}/rl_train.log','a',encoding='utf-8')
def log(m):LF.write(m+'\n');LF.flush();print(m,flush=True)

device=torch.device('cuda')

# ── Model (même archi) ──────────────────────────────────────────────
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

# ── Load scores + generations ─────────────────────────────────────────
def load_scored_data():
    """Charge scores.jsonl et generations.jsonl, fusionne par id"""
    scores_map={}
    with open(f'{RL_DATA}/scores.jsonl','r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                s=json.loads(line)
                scores_map[s['id']]=s

    gens_map={}
    with open(f'{RL_DATA}/generations.jsonl','r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                g=json.loads(line)
                gens_map[g['id']]=g

    # Fusionner: garder uniquement les ids présents dans les deux
    merged=[]
    for sid,scored in scores_map.items():
        if sid in gens_map:
            gen=gens_map[sid]
            merged.append({
                'id':sid,
                'prompt':scored['prompt'],
                'scored_completions':scored['completions'],  # dicts avec grammar,local_coh etc
                'gen_completions':gen['completions'],         # strings (le texte brut)
            })
    return merged

# ── Pretraining batch ────────────────────────────────────────────────
def pretrain_batch(mm,B,block,device):
    """Random batch du corpus prétraining (standard CE)"""
    nt=len(mm)-block-1
    ix=[random.randint(0,nt) for _ in range(B)]
    xb=torch.stack([torch.from_numpy(mm[i:i+block].astype(np.int64).copy()) for i in ix]).to(device)
    yb=torch.stack([torch.from_numpy(mm[i+1:i+block+1].astype(np.int64).copy()) for i in ix]).to(device)
    return xb,yb

# ── KL computation ───────────────────────────────────────────────────
def compute_kl(model,ref_model,tokens,prompt_lens,device):
    """
    Compute KL divergence entre modèle courant et référence
    sur les tokens de complétion uniquement.
    """
    with torch.no_grad():
        ref_logits=ref_model(tokens).float()
    model_logits=model(tokens).float()
    # Pour chaque séquence, masquer les tokens du prompt
    B,T,V=model_logits.shape
    kl_total=0.0
    n_tokens=0
    for b in range(B):
        pl=prompt_lens[b]
        # KL sur les tokens après le prompt
        ref_logprobs=F.log_softmax(ref_logits[b,pl:],dim=-1)
        model_logprobs=F.log_softmax(model_logits[b,pl:],dim=-1)
        kl=F.kl_div(ref_logprobs,model_logprobs.exp(),reduction='sum')
        kl_total+=kl.item()
        n_tokens+=T-pl
    return kl_total/max(n_tokens,1)

# ── Main ─────────────────────────────────────────────────────────────
if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('camembert-base')

    # Load scored + generated data (merged)
    scored_data=load_scored_data()
    log(f'Merged entries: {len(scored_data)}')
    if len(scored_data)==0:
        log('No scored data found! Run rl_generate.py then rl_judge.py first.')
        exit(1)

    # Stats
    all_scores=[c['total'] for s in scored_data for c in s['scored_completions']]
    log(f'Score stats: avg={sum(all_scores)/len(all_scores):.1f} min={min(all_scores)} max={max(all_scores)}')

    # Load current model
    model=OscModel().to(device).to(torch.bfloat16)
    current_ckpt=f'{RESULT_DIR}/rl_latest.pt'
    base_ckpt='results_5b/best.pt'
    if os.path.exists(current_ckpt):
        ckpt=torch.load(current_ckpt,map_location='cuda',weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_step=ckpt['step']+1
        log(f'Resumed RL from step {ckpt["step"]}, loss={ckpt["loss"]:.4f}')
    else:
        ckpt=torch.load(base_ckpt,map_location='cuda',weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_step=1
        log(f'Loaded base model: step {ckpt["step"]}, loss {ckpt["loss"]:.4f}')

    # Reference model (frozen, for KL)
    ref_model=OscModel().to(device).to(torch.bfloat16)
    ref_ckpt=torch.load(base_ckpt,map_location='cuda',weights_only=False)
    ref_model.load_state_dict(ref_ckpt['model'])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad=False
    log('Reference model loaded (frozen)')

    # Load pretraining data
    mm=np.load(PRETRAIN_DATA,mmap_mode='r')
    log(f'Pretrain data: {len(mm):,} tokens')

    # Hyperparams
    B_RAFT=16          # prompts par batch RL
    B_PRETRAIN=16      # batch pretraining
    BLOCK=256
    LR=1e-5
    BETA_KL=0.01       # KL penalty weight
    N_EPOCHS=3
    N_STEPS=200        # steps per epoch
    N_TOTAL=start_step-1+N_EPOCHS*N_STEPS

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)

    log(f'RL Config: B_raft={B_RAFT} B_pt={B_PRETRAIN} lr={LR} beta_kl={BETA_KL} epochs={N_EPOCHS} steps/epoch={N_STEPS}')

    losses=[];kl_vals=[];raft_losses=[];start_time=time.time()

    for global_step in range(start_step,N_TOTAL+1):
        model.train()
        t0=time.time()
        opt.zero_grad(set_to_none=True)

        # ── RAFT batch ────────────────────────────────────────────
        batch_entries=random.sample(scored_data,min(B_RAFT,len(scored_data)))
        raft_tokens=[];raft_targets=[];raft_weights=[];raft_prompt_lens=[]

        for entry in batch_entries:
            scored_comps=entry['scored_completions']
            gen_comps=entry['gen_completions']
            sorted_comps=sorted(scored_comps,key=lambda x:x['total'],reverse=True)
            n_keep=max(1,len(sorted_comps)//4)
            top_comps=sorted_comps[:n_keep]
            scores_t=torch.tensor([c['total'] for c in top_comps],dtype=torch.float32)
            weights=F.softmax(scores_t,dim=0)

            for i,comp in enumerate(top_comps):
                orig_idx=comp['index']
                comp_text=gen_comps[orig_idx]  # string from generations.jsonl

                full_text=entry['prompt']+' '+comp_text
                ids=tok.encode(full_text,return_tensors='pt')[0][:BLOCK]
                if len(ids)<10:
                    continue

                prompt_len=len(tok.encode(entry['prompt'],return_tensors='pt')[0][:BLOCK//2])
                raft_tokens.append(ids)
                raft_targets.append(torch.cat([ids[1:],torch.tensor([0])]))
                raft_weights.append(weights[i].item())
                raft_prompt_lens.append(min(prompt_len,BLOCK//2))

        if not raft_tokens:
            if global_step==start_step:
                log(f'DEBUG: scored_comps[0]={scored_data[0]["scored_completions"][0]}')
                log(f'DEBUG: gen_comps type={type(scored_data[0]["gen_completions"][0])} len={len(scored_data[0]["gen_completions"])}')
                e=scored_data[0]
                for sc in e['scored_completions'][:2]:
                    idx=sc['index']
                    ct=e['gen_completions'][idx]
                    log(f'DEBUG: idx={idx} type={type(ct)} val={repr(ct)[:60]}')
            log(f'Step {global_step}: No valid raft pairs, skipping')
            continue

        # Pad to same length
        max_len=max(t.size(0) for t in raft_tokens)
        raft_tokens_padded=torch.zeros(len(raft_tokens),max_len,dtype=torch.long,device=device)
        raft_targets_padded=torch.zeros(len(raft_tokens),max_len,dtype=torch.long,device=device)
        raft_mask=torch.zeros(len(raft_tokens),max_len,dtype=torch.bool,device=device)

        for i,(ids,tgt) in enumerate(zip(raft_tokens,raft_targets)):
            L=ids.size(0)
            raft_tokens_padded[i,:L]=ids
            raft_targets_padded[i,:L]=tgt
            raft_mask[i,:L]=True

        # Forward pass
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            logits=model(raft_tokens_padded).float()
            # CE sur tous les tokens (le mask gère le padding)
            logprobs=F.log_softmax(logits,dim=-1)
            token_logprobs=torch.gather(logprobs,2,raft_targets_padded.unsqueeze(2)).squeeze(2)
            # RAFT loss: moyenne pondérée par les poids
            token_losses=-token_logprobs
            # Masquer le padding
            masked_losses=token_losses*raft_mask.float()
            # Poids par complétion (broadcast)
            weights_t=torch.tensor(raft_weights,device=device).unsqueeze(1)
            weighted_loss=(masked_losses*weights_t).sum()/raft_mask.float().sum()
            raft_loss=weighted_loss

        # KL against reference
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            kl=compute_kl(model,ref_model,raft_tokens_padded,raft_prompt_lens,device)

        # Total loss
        loss=raft_loss+BETA_KL*kl

        # ── Pretrain batch (50/50 mix) ────────────────────────────
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            xb,yb=pretrain_batch(mm,B_PRETRAIN,BLOCK,device)
            pt_logits=model(xb).float().reshape(-1,32005)
            pt_targets=yb.reshape(-1)
            ok=pt_targets<pt_logits.shape[-1]
            pt_loss=F.cross_entropy(pt_logits[ok],pt_targets[ok],reduction='mean',label_smoothing=0.1)

        # Combined loss: 50% RAFT + 50% pretraining
        total_loss=0.5*loss+0.5*pt_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        elapsed=time.time()-t0
        losses.append(total_loss.item())
        raft_losses.append(raft_loss.item())
        kl_vals.append(kl)

        if global_step%10==0:
            avg_loss=sum(losses[-10:])/len(losses[-10:])
            avg_raft=sum(raft_losses[-10:])/len(raft_losses[-10:])
            avg_kl=sum(kl_vals[-10:])/len(kl_vals[-10:])
            log(f'Step {global_step:4d} | loss={avg_loss:.4f} raft={avg_raft:.4f} pt={pt_loss.item():.4f} kl={avg_kl:.4f} | {elapsed:.1f}s')

        if global_step%50==0:
            # Save checkpoint
            torch.save({
                'step':global_step,
                'model':model.state_dict(),
                'loss':sum(losses[-50:])/len(losses[-50:]),
                'optimizer':opt.state_dict(),
            },f'{RESULT_DIR}/rl_latest.pt')
            log(f'  Checkpoint saved (step {global_step})')

            # Quick eval: score summary
            model.eval()
            with torch.no_grad():
                # Générer un échantillon pour vérifier la cohérence
                test_prompts=["La signification de la vie est","Il était une fois","Le chat est assis sur"]
                for tp in test_prompts:
                    ids=tok.encode(tp,return_tensors='pt').to(device)
                    gen=[]
                    for _ in range(64):
                        with torch.amp.autocast('cuda',enabled=False):
                            nxt=model(ids).float()
                        probs=torch.softmax(nxt[:,-1,:]/0.8,dim=-1)
                        v,_=torch.topk(probs,50);probs[probs<v[:,-1:]]=0;probs/=probs.sum(-1,keepdim=True)
                        ntx=torch.multinomial(probs,1)
                        ids=torch.cat([ids,ntx],dim=1)
                        gen.append(ntx.item())
                    unique=len(set(gen))/len(gen)*100
                    log(f'    "{tp}" -> {tok.decode(ids[0])[:80]}... (unique={unique:.0f}%)')

            json.dump({
                'step':global_step,'loss':avg_loss,'raft':avg_raft,'kl':avg_kl,
                'n_scored':len(scored_data),'beta_kl':BETA_KL,
            },open(f'{RESULT_DIR}/rl_status.json','w'))

    log(f'\n=== RL DONE === Best loss: {min(losses):.4f}')
    torch.save({
        'step':N_TOTAL,'model':model.state_dict(),
        'loss':min(losses),'optimizer':opt.state_dict(),
    },f'{RESULT_DIR}/rl_final.pt')
    LF.close()
