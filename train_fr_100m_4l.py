"""
Oscillator LLM 100M 4L - French (Lucie dataset, CamemBERT tokenizer)
d=1152, 4 layers, 16 heads, 1536 osc/couche, vocab=32005
bf16 throughout, ~103M params, short gradient path
"""
import sys,os,json,time,random,math,copy,traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULT_DIR='results_fr100m_4l'
STATUS_FILE='training_status_fr100m_4l.json'
os.makedirs(RESULT_DIR,exist_ok=True)
LF=open(f'{RESULT_DIR}/train.log','a',encoding='utf-8')
def log(m):LF.write(m+'\n');LF.flush();print(m,flush=True)

device=torch.device('cuda')

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
        u=(torch.einsum('btd,od->bto',x.float(),self.B_re.float())+1j*torch.einsum('btd,od->bto',x.float(),self.B_im.float()))  # FIX audit: .float() pour gen hors autocast
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
        return (feats@self.C.t().float()).to(x.dtype),hc  # FIX audit: .float() pour gen hors autocast

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
        self._init()
    def _init(self):
        for n,p in self.named_parameters():
            if any(k in n for k in['omega','decay_logit','K_raw']):continue
            if p.dim()>1:nn.init.xavier_uniform_(p,gain=0.7)
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos);x=self.dropout(x)
        for blk in self.layers:
            o,hc=blk['osc'](blk['norm1'](x),None);x=x+blk['dropout'](o)
            h=blk['norm3'](x);B2,T2,D=h.shape
            qkv=blk['qkv'](h).reshape(B2,T2,3,self.n_heads,self.head_dim);q,k,v=qkv.unbind(2)
            a=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),dropout_p=0.1 if self.training else 0.0,is_causal=True)  # FIX audit: pas de dropout en eval/gen
            x=x+blk['proj'](blk['dropout'](a.permute(0,2,1,3).reshape(B2,T2,D)))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

log('=== Oscillator FR 100M 4L ===')
model=OscModel().to(device).to(torch.bfloat16)
np_=sum(p.numel()for p in model.parameters())
no=sum(p.numel()for n,p in model.named_parameters()if any(k in n for k in['omega','decay_logit','K_raw','B_re','B_im']))
log(f'Params: {np_:,} (osc={no:,}={no/np_*100:.1f}%)')
log(f'VRAM: {torch.cuda.memory_allocated()/1e6:.0f}MB allocated')

mm=np.load('lucie_fr_2b.npy',mmap_mode='r')
log(f'Dataset: {len(mm):,} tokens, dtype={mm.dtype}, range=[{mm.min()},{mm.max()}]')

B=32;N=500000;BLOCK=256;LR_MAX=3e-3;WARMUP=1000;GRAD_CLIP=0.5
HELDOUT=10_000_000  # FIX audit: 10M derniers tokens reserves a l'eval (jamais vus en train)
opt=torch.optim.AdamW(model.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.999))
nt=mm.shape[0]-BLOCK-1-HELDOUT;tt=0.0;ts=0;start_step=1;nan_streak=0
ev_lo=mm.shape[0]-HELDOUT;ev_hi=mm.shape[0]-BLOCK-1
log(f'Train window: [0,{nt:,}) | Eval window: [{ev_lo:,},{ev_hi:,})')

def cosine_lr(step,total,warmup,lr_max,lr_min=1e-5):
    if step<warmup:return lr_max*(step+1)/max(warmup,1)
    prog=(step-warmup)/max(total-warmup,1)
    return lr_min+0.5*(lr_max-lr_min)*(1+math.cos(math.pi*prog))

opt.param_groups[0]['lr']=cosine_lr(0,N,WARMUP,LR_MAX)

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('camembert-base')

RESUME_CKPT=f'{RESULT_DIR}/ckpt_latest.pt'
if os.path.exists(RESUME_CKPT):
    ckpt=torch.load(RESUME_CKPT,map_location='cuda',weights_only=False)
    model.load_state_dict(ckpt['model'],strict=False)
    opt.load_state_dict(ckpt['optimizer'])
    start_step=ckpt['step']+1
    ts=ckpt.get('tokens_seen',ckpt['step']*B*BLOCK)  # FIX audit: compteur tokens persistant (fallback: estimation par step)
    log(f'Resumed from step {ckpt["step"]}, loss={ckpt["loss"]:.4f}, tokens_seen={ts/1e9:.2f}B')

model.train();t_start=time.time()
best_eval=999;best_step=0;eval_losses=[];losses=[]

for step in range(start_step,N+1):
    lr=cosine_lr(step-1,N,WARMUP,LR_MAX)
    opt.param_groups[0]['lr']=lr
    t0=time.time();opt.zero_grad(set_to_none=True)
    skip=False;cur_loss=None
    try:
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            ix=[random.randint(0,nt)for _ in range(B)]
            xb=torch.stack([torch.from_numpy(mm[i:i+BLOCK].astype(np.int64).copy())for i in ix]).to(device)
            yb=torch.stack([torch.from_numpy(mm[i+1:i+BLOCK+1].astype(np.int64).copy())for i in ix]).to(device)
            lg=model(xb).float().reshape(-1,32005);yf=yb.reshape(-1);ok=yf<lg.shape[-1]
            loss=F.cross_entropy(lg[ok],yf[ok],reduction='mean',label_smoothing=0.1);ts+=B*BLOCK;cur_loss=loss.item()
        if torch.isnan(loss)or torch.isinf(loss):
            nan_streak+=1;skip=True;log(f'  NaN/Inf loss at step {step} (streak={nan_streak})')
            if nan_streak>=100:log('100 consecutive NaNs, stopping.');break
        else:nan_streak=max(0,nan_streak-1)
        if not skip:
            loss.backward()
            has_nan_grad=False
            for p in model.parameters():
                if p.grad is not None and(torch.isnan(p.grad).any()or torch.isinf(p.grad).any()):
                    has_nan_grad=True;break
            if has_nan_grad:
                opt.zero_grad(set_to_none=True);nan_streak+=1;skip=True
                log(f'  NaN/Inf grad at step {step} (streak={nan_streak})')
                if nan_streak>=100:log('100 consecutive NaN grads, stopping.');break
            else:nan_streak=max(0,nan_streak-1)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            log(f'  OOM at step {step}');torch.cuda.empty_cache();skip=True
        else:log(f'  CRASH at step {step}: {e}');skip=True
    except Exception as e:
        log(f'  UNEXPECTED ERROR at step {step}: {e}');skip=True
    tt+=time.time()-t0
    if cur_loss is not None:losses.append(cur_loss)

    if step%100==0 and losses:
        elapsed=time.time()-t_start;tok_s=ts/elapsed;avg_loss=sum(losses[-100:])/len(losses[-100:])
        log(f'Step {step:6d} loss={avg_loss:.4f} lr={lr:.6f} {step/tt:.1f}st/s {elapsed:.0f}s {ts/1e9:.2f}Btok')
        json.dump({
            'step':step,'total_steps':N,'tokens_seen':int(ts),'loss':avg_loss,'lr':lr,
            'tok_s':int(tok_s),'elapsed_s':round(tt,1),
            'cur_mem_gb':round(torch.cuda.memory_allocated()/1e9,2),
            'batch_size':B,'n_params':np_,'variant':'Osc-FR-100M-4L',
            'best_eval':best_eval,'best_step':best_step
        },open(STATUS_FILE,'w'))

    if step%500==0:
        try:
            model.eval();el_=[];el_h=[]
            with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(20):
                    ei=[random.randint(ev_lo,ev_hi)for _ in range(16)]  # FIX audit: held-out
                    ex=torch.stack([torch.from_numpy(mm[i:i+BLOCK+1].astype(np.int64).copy())for i in ei]).to(device)
                    logits=model(ex[:,:-1]).float()
                    fl=logits.reshape(-1,32005);ft=ex[:,1:].reshape(-1)
                    el=F.cross_entropy(fl,ft,label_smoothing=0.1);el_.append(el.item())
                    el_h.append(F.cross_entropy(fl,ft).item())  # FIX audit: vraie NLL sans smoothing
                    del ex,logits
            evl=sum(el_)/len(el_);nll=sum(el_h)/len(el_h);eval_losses.append((step,evl,nll))
            log(f'  EVAL{step}: smooth={evl:.4f} nll_hard={nll:.4f} ppl={math.exp(min(nll,20)):.1f}')
            if evl<best_eval:
                best_eval=evl;best_step=step
                torch.save({'step':step,'model':model.state_dict(),'loss':evl,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/best.pt')
                log(f'  -> NEW BEST')
        except Exception as e:
            log(f'  EVAL FAILED: {e}')
        try:
            torch.save({'step':step,'model':model.state_dict(),'loss':avg_loss if losses else 999,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/ckpt_latest.pt')
            log(f'  Checkpoint saved at step {step}')
        except Exception as e:log(f'  CKPT FAILED: {e}')
        try:
            gen_metrics={'unique':[],'bigram':[],'trigram':[],'entropy':[]}
            with torch.no_grad():
                for p_ in["La signification de la vie est","Il était une fois","Le chat est assis sur","Bonjour tout le monde","L'intelligence artificielle"]:
                    out_ids=tok.encode(p_,return_tensors='pt').to(device)
                    gen_toks=[];logits_all=[]
                    for _ in range(120):
                        with torch.amp.autocast('cuda',enabled=False):
                            nxt=model(out_ids).float()
                        logits_all.append(nxt[:,-1,:])
                        probs=torch.softmax(nxt[:,-1,:]/0.8,dim=-1)
                        v,_=torch.topk(probs,50);probs[probs<v[:,-1:]]=0;probs/=probs.sum(-1,keepdim=True)
                        ntx=torch.multinomial(probs,1);out_ids=torch.cat([out_ids,ntx],dim=1)
                        gen_toks.append(ntx.item())
                    unique=len(set(gen_toks))/len(gen_toks)*100
                    bg=[(gen_toks[i],gen_toks[i+1])for i in range(len(gen_toks)-1)]
                    bigram=(1-len(set(bg))/max(len(bg),1))*100
                    tg=[(gen_toks[i],gen_toks[i+1],gen_toks[i+2])for i in range(len(gen_toks)-2)]
                    trigram=(1-len(set(tg))/max(len(tg),1))*100
                    ent=torch.stack([F.softmax(lg,dim=-1)for lg in logits_all])
                    entropy=-torch.sum(ent*torch.log(ent+1e-10),dim=-1).mean().item()
                    gen_metrics['unique'].append(unique);gen_metrics['bigram'].append(bigram)
                    gen_metrics['trigram'].append(trigram);gen_metrics['entropy'].append(entropy)
                    log(f'  >> "{p_}" -> {tok.decode(out_ids[0])}')
                    log(f'     uniq={unique:.0f}% bigram={bigram:.0f}% trigram={trigram:.0f}% entropy={entropy:.2f}')
            avg_u=sum(gen_metrics['unique'])/len(gen_metrics['unique'])
            avg_bg=sum(gen_metrics['bigram'])/len(gen_metrics['bigram'])
            avg_tg=sum(gen_metrics['trigram'])/len(gen_metrics['trigram'])
            avg_ent=sum(gen_metrics['entropy'])/len(gen_metrics['entropy'])
            log(f'  GEN AVG: unique={avg_u:.1f}% bigram={avg_bg:.1f}% trigram={avg_tg:.1f}% entropy={avg_ent:.2f}')
            json.dump({'step':step,'unique':avg_u,'bigram':avg_bg,'trigram':avg_tg,'entropy':avg_ent},open(f'{RESULT_DIR}/gen_metrics.json','w'))
        except Exception as e:log(f'  Gen: {e}')
        model.train()

    if step%50000==0 and step>0:
        try:
            torch.save({'step':step,'model':model.state_dict(),'loss':avg_loss if losses else 999,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/step_{step}.pt')
            log(f'  CHECKPOINT saved at step {step}')
        except Exception as e:log(f'  CKPT50K FAILED: {e}')

log(f'\n=== DONE === Best eval: {best_eval:.4f} at step {best_step}')
try:
    torch.save({'step':N,'model':model.state_dict(),'loss':avg_loss if losses else 999,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/final.pt')
except:pass
LF.close()
