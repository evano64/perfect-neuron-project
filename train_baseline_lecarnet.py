"""
Baseline SwiGLU Transformer 100M 4L - LeCarnet (same as oscillator LeCarnet)
ABLATION PROPRE: meme archi sauf le bloc KuramotoOsc -> SwiGLU
Pour prouver que l'oscillateur apporte vraiment quelque chose.
"""
import warnings
warnings.filterwarnings('ignore')
import sys,os,json,time,random,math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL']='1'

RESULT_DIR='results_baseline_lecarnet'
os.makedirs(RESULT_DIR,exist_ok=True)
LF=open(f'{RESULT_DIR}/train.log','a',encoding='utf-8')
def log(m):LF.write(m+'\n');LF.flush();print(m,flush=True)

device=torch.device('cuda')

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))

class BaselineSwiGLU(nn.Module):
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

log('=== Baseline SwiGLU 100M 4L - LeCarnet ===')
model=BaselineSwiGLU().to(device).to(torch.bfloat16)
np_=sum(p.numel()for p in model.parameters())
log(f'Params: {np_:,}')

mm=np.load('lecarnet_tokens.npy',mmap_mode='r')
log(f'Dataset: {len(mm):,} tokens, dtype={mm.dtype}, range=[{mm.min()},{mm.max()}]')

B=32;N=5000;BLOCK=256;LR_MAX=3e-3;WARMUP=1000;GRAD_CLIP=0.5
HELDOUT=2_000_000
opt=torch.optim.AdamW(model.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.999))
nt=mm.shape[0]-BLOCK-1-HELDOUT;ts=0;nan_streak=0
ev_lo=mm.shape[0]-HELDOUT;ev_hi=mm.shape[0]-BLOCK-1
log(f'Train: [0,{nt:,}) | Eval: [{ev_lo:,},{ev_hi:,})')

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('camembert-base')

def cosine_lr(step,total,warmup,lr_max,lr_min=1e-5):
    if step<warmup:return lr_max*(step+1)/max(warmup,1)
    prog=(step-warmup)/max(total-warmup,1)
    return lr_min+0.5*(lr_max-lr_min)*(1+math.cos(math.pi*prog))

start_step=1

best_eval=999;best_step=0;losses=[];tt=0.0
model.train();t_start=time.time()

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
            nan_streak+=1;skip=True
            if nan_streak>=100:break
        else:nan_streak=max(0,nan_streak-1)
        if not skip:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step()
    except RuntimeError as e:
        if 'out of memory' in str(e):torch.cuda.empty_cache();skip=True
        else:skip=True
    except:skip=True
    tt+=time.time()-t0
    if cur_loss is not None:losses.append(cur_loss)

    if step%100==0 and losses:
        elapsed=time.time()-t_start;avg_loss=sum(losses[-100:])/len(losses[-100:])
        log(f'Step {step:5d} loss={avg_loss:.4f} lr={lr:.6f} {step/tt:.1f}st/s {elapsed:.0f}s {ts/1e6:.1f}Mtok')

    if step%500==0:
        try:
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
                best_eval=evl;best_step=step
                torch.save({'step':step,'model':model.state_dict(),'loss':evl,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/best.pt')
                log(f'  -> NEW BEST')
        except:pass
        try:
            torch.save({'step':step,'model':model.state_dict(),'loss':avg_loss if losses else 999,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/ckpt_latest.pt')
        except:pass
        try:
            with torch.no_grad():
                for p_ in["Il etait une fois","Le chat est assis sur","La signification de la vie est"]:
                    ids=tok.encode(p_,return_tensors='pt').to(device)
                    for _ in range(60):
                        inp=ids[:,-BLOCK:]
                        logits=model(inp)[:,-1,:]/0.8
                        v,_=torch.topk(logits,50)
                        logits[logits<v[0,-1]]=-float('inf')
                        nxt=torch.multinomial(torch.softmax(logits,-1),1)
                        ids=torch.cat([ids,nxt],1)
                    log(f'  >> "{p_}" -> {tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)[:120]}')
        except:pass
        model.train()

log(f'\n=== DONE === Best eval: {best_eval:.4f} at step {best_step}')
try:
    torch.save({'step':N,'model':model.state_dict(),'loss':avg_loss if losses else 999,'optimizer':opt.state_dict(),'tokens_seen':int(ts)},f'{RESULT_DIR}/final.pt')
except:pass
LF.close()
