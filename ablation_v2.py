"""
ABLACTION PROPRE v2: Oscillateur vs SwiGLU sur LeCarnet
Séquentiel: un modèle à la fois, pas de contention GPU.
"""
import warnings
warnings.filterwarnings('ignore')
import sys,os,json,time,random,math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL']='1'

RESULT_DIR='results_ablation_v2'
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
            a=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),dropout_p=0.1 if self.training else 0.0,is_causal=True)
            x=x+blk['proj'](blk['dropout'](a.permute(0,2,1,3).reshape(B2,T2,D)))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))

class BaselineModel(nn.Module):
    def __init__(self,vocab=32005,d_model=1152,n_layers=4,n_heads=16,d_ff=3072,max_len=256,dropout=0.1):
        super().__init__()
        self.d_model=d_model;self.n_heads=n_heads;self.head_dim=d_model//n_heads
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.dropout=nn.Dropout(dropout)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mlp':SwiGLU(d_model,d_ff,dropout),
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
            x=x+blk['dropout'](blk['mlp'](blk['norm1'](x)))
            h=blk['norm3'](x);B2,T2,D=h.shape
            qkv=blk['qkv'](h).reshape(B2,T2,3,self.n_heads,self.head_dim);q,k,v=qkv.unbind(2)
            a=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),dropout_p=0.1 if self.training else 0.0,is_causal=True)
            x=x+blk['proj'](blk['dropout'](a.permute(0,2,1,3).reshape(B2,T2,D)))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('camembert-base')
mm=np.load('lecarnet_tokens.npy',mmap_mode='r')

B=32;N=5000;BLOCK=256;LR_MAX=3e-3;WARMUP=1000;GRAD_CLIP=0.5
HELDOUT=2_000_000
nt=mm.shape[0]-BLOCK-1-HELDOUT
ev_lo=mm.shape[0]-HELDOUT;ev_hi=mm.shape[0]-BLOCK-1

def cosine_lr(step,lr_max):
    if step<WARMUP:return lr_max*(step+1)/max(WARMUP,1)
    prog=(step-WARMUP)/max(N-WARMUP,1)
    return 1e-5+0.5*(lr_max-1e-5)*(1+math.cos(math.pi*prog))

def train_model(model,name,lr_max=3e-3):
    log(f'\n{"="*50}')
    log(f'TRAINING: {name} (lr_max={lr_max})')
    np_=sum(p.numel()for p in model.parameters())
    log(f'Params: {np_:,}')
    model=model.to(device).to(torch.bfloat16)

    opt=torch.optim.AdamW(model.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.999))
    ts=0;tt=0.0;best_eval=999
    model.train();t_start=time.time();losses=[]

    for step in range(1,N+1):
        lr=cosine_lr(step-1,lr_max)
        opt.param_groups[0]['lr']=lr
        t0=time.time();opt.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                ix=[random.randint(0,nt)for _ in range(B)]
                xb=torch.stack([torch.from_numpy(mm[i:i+BLOCK].astype(np.int64).copy())for i in ix]).to(device)
                yb=torch.stack([torch.from_numpy(mm[i+1:i+BLOCK+1].astype(np.int64).copy())for i in ix]).to(device)
                lg=model(xb).float().reshape(-1,32005);yf=yb.reshape(-1);ok=yf<lg.shape[-1]
                loss=F.cross_entropy(lg[ok],yf[ok],reduction='mean',label_smoothing=0.1);ts+=B*BLOCK
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP);opt.step()
        except RuntimeError as e:
            if 'out of memory' in str(e):torch.cuda.empty_cache()
        except:pass
        tt+=time.time()-t0
        if not torch.isnan(loss):losses.append(loss.item())

        if step%100==0 and losses:
            elapsed=time.time()-t_start;avg=sum(losses[-100:])/len(losses[-100:])
            log(f'  Step {step:5d} loss={avg:.4f} lr={lr:.6f} {step/tt:.1f}st/s {elapsed:.0f}s')

        if step%500==0:
            model.eval();el_=[]
            with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(20):
                    ei=[random.randint(ev_lo,ev_hi)for _ in range(16)]
                    ex=torch.stack([torch.from_numpy(mm[i:i+BLOCK+1].astype(np.int64).copy())for i in ei]).to(device)
                    logits=model(ex[:,:-1]).float()
                    fl=logits.reshape(-1,32005);ft=ex[:,1:].reshape(-1)
                    el_.append(F.cross_entropy(fl,ft,label_smoothing=0.1).item())
                    del ex,logits
            evl=sum(el_)/len(el_)
            log(f'  EVAL{step}: loss={evl:.4f} ppl={math.exp(min(evl,20)):.1f}')
            if evl<best_eval:
                best_eval=evl
                torch.save({'step':step,'model':model.state_dict(),'loss':evl},f'{RESULT_DIR}/{name}_best.pt')
                log(f'  -> NEW BEST')
            # Gen
            try:
                with torch.no_grad():
                    for p_ in["Il etait une fois","Le chat est assis sur"]:
                        ids=tok.encode(p_,return_tensors='pt').to(device)
                        for _ in range(40):
                            inp=ids[:,-BLOCK:]
                            lg=model(inp)[:,-1,:]/0.8;v,_=torch.topk(lg,50);lg[lg<v[0,-1]]=-float('inf')
                            nxt=torch.multinomial(torch.softmax(lg,-1),1);ids=torch.cat([ids,nxt],1)
                        log(f'  >> "{p_}" -> {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)[:100]}')
            except:pass
            model.train()

    log(f'  DONE: best_eval={best_eval:.4f}')
    return best_eval

log('=== ABLATION v2: Oscillateur vs SwiGLU (sequencial) ===')
log(f'Dataset: {len(mm):,} tokens | Steps: {N} | Batch: {B}')
log('NOTE: Osc LR=1e-3 (stabilite), Baseline LR=3e-3 (avantage baseline)')

# Train baseline first (clean GPU, LR=3e-3)
base=BaselineModel()
base_eval=train_model(base,'baseline',lr_max=3e-3)
del base;torch.cuda.empty_cache()

# Then oscillator (clean GPU, LR=1e-3 pour stabilite)
osc=OscModel()
osc_eval=train_model(osc,'oscillator',lr_max=1e-3)
del osc;torch.cuda.empty_cache()

log(f'\n{"="*50}')
log(f'FINAL: Baseline={base_eval:.4f} | Oscillator={osc_eval:.4f}')
log(f'GAP: {base_eval-osc_eval:+.4f}')
if osc_eval<base_eval:log('OSCILLATOR WINS')
else:log('BASELINE WINS')
LF.close()
