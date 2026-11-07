"""
ABLACTION PROPRE: Oscillateur vs SwiGLU sur LeCarnet
Les deux depuis zéro, mêmes hyperparamètres, même dataset.
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

RESULT_DIR='results_ablation_lecarnet'
os.makedirs(RESULT_DIR,exist_ok=True)
LF=open(f'{RESULT_DIR}/train.log','a',encoding='utf-8')
def log(m):LF.write(m+'\n');LF.flush();print(m,flush=True)

device=torch.device('cuda')

# ── Oscillator block ──
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

# ── SwiGLU block ──
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))

# ── Oscillator Model ──
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

# ── Baseline SwiGLU Model (même structure, SwiGLU remplace Osc) ──
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

log('=== ABLATION LeCarnet: Oscillateur vs SwiGLU ===')
log('Même dataset, mêmes hyperparams, même archi sauf le bloc mélangeur')

osc=OscModel().to(device).to(torch.bfloat16)
base=BaselineModel().to(device).to(torch.bfloat16)
osc_np=sum(p.numel()for p in osc.parameters())
base_np=sum(p.numel()for p in base.parameters())
log(f'Oscillateur: {osc_np:,} params')
log(f'Baseline:    {base_np:,} params')

mm=np.load('lecarnet_tokens.npy',mmap_mode='r')
log(f'Dataset: {len(mm):,} tokens')

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('camembert-base')

# ── Hyperparams identiques ──
B=32;N=5000;BLOCK=256;LR_MAX=3e-3;WARMUP=1000;GRAD_CLIP=0.5
HELDOUT=2_000_000
nt=mm.shape[0]-BLOCK-1-HELDOUT
ev_lo=mm.shape[0]-HELDOUT;ev_hi=mm.shape[0]-BLOCK-1
log(f'Train: [0,{nt:,}) | Eval: [{ev_lo:,},{ev_hi:,})')

opt_osc=torch.optim.AdamW(osc.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.999))
opt_base=torch.optim.AdamW(base.parameters(),lr=0,weight_decay=0.01,betas=(0.9,0.999))

def cosine_lr(step,total,warmup,lr_max,lr_min=1e-5):
    if step<warmup:return lr_max*(step+1)/max(warmup,1)
    prog=(step-warmup)/max(total-warmup,1)
    return lr_min+0.5*(lr_max-lr_min)*(1+math.cos(math.pi*prog))

ts_osc=0;ts_base=0;tt=0.0
osc.train();base.train();t_start=time.time()
best_osc=999;base_best=999;step_base=0;step_osc=0

results={'osc':[],'base':[]}

for step in range(1,N+1):
    lr=cosine_lr(step-1,N,WARMUP,LR_MAX)

    # ── Train Oscillator ──
    opt_osc.param_groups[0]['lr']=lr
    try:
        t0=time.time()
        opt_osc.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            ix=[random.randint(0,nt)for _ in range(B)]
            xb=torch.stack([torch.from_numpy(mm[i:i+BLOCK].astype(np.int64).copy())for i in ix]).to(device)
            yb=torch.stack([torch.from_numpy(mm[i+1:i+BLOCK+1].astype(np.int64).copy())for i in ix]).to(device)
            lg=osc(xb).float().reshape(-1,32005);yf=yb.reshape(-1);ok=yf<lg.shape[-1]
            loss_o=F.cross_entropy(lg[ok],yf[ok],reduction='mean',label_smoothing=0.1);ts_osc+=B*BLOCK
        loss_o.backward();torch.nn.utils.clip_grad_norm_(osc.parameters(),GRAD_CLIP);opt_osc.step()
    except:pass

    # ── Train Baseline ──
    opt_base.param_groups[0]['lr']=lr
    try:
        opt_base.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            ix2=[random.randint(0,nt)for _ in range(B)]
            xb2=torch.stack([torch.from_numpy(mm[i:i+BLOCK].astype(np.int64).copy())for i in ix2]).to(device)
            yb2=torch.stack([torch.from_numpy(mm[i+1:i+BLOCK+1].astype(np.int64).copy())for i in ix2]).to(device)
            lg2=base(xb2).float().reshape(-1,32005);yf2=yb2.reshape(-1);ok2=yf2<lg2.shape[-1]
            loss_b=F.cross_entropy(lg2[ok2],yf2[ok2],reduction='mean',label_smoothing=0.1);ts_base+=B*BLOCK
        loss_b.backward();torch.nn.utils.clip_grad_norm_(base.parameters(),GRAD_CLIP);opt_base.step()
    except:pass

    tt+=time.time()-t0

    if step%100==0:
        elapsed=time.time()-t_start
        log(f'Step {step:5d} | OSC loss={loss_o.item():.4f} | BASE loss={loss_b.item():.4f} | lr={lr:.6f} | {elapsed:.0f}s')

    if step%500==0:
        # Eval both
        osc.eval();base.eval()
        el_o=[];el_b=[]
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(20):
                ei=[random.randint(ev_lo,ev_hi)for _ in range(16)]
                ex=torch.stack([torch.from_numpy(mm[i:i+BLOCK+1].astype(np.int64).copy())for i in ei]).to(device)
                lo=osc(ex[:,:-1]).float();fl=lo.reshape(-1,32005);ft=ex[:,1:].reshape(-1)
                el_o.append(F.cross_entropy(fl,ft,label_smoothing=0.1).item())
                lb=base(ex[:,:-1]).float();fl2=lb.reshape(-1,32005)
                el_b.append(F.cross_entropy(fl2,ft,label_smoothing=0.1).item())
                del ex,lo,lb
        ev_o=sum(el_o)/len(el_o);ev_b=sum(el_b)/len(el_b)
        results['osc'].append({'step':step,'loss':ev_o})
        results['base'].append({'step':step,'loss':ev_b})
        best_osc=min(best_osc,ev_o);base_best=min(base_best,ev_b)

        log(f'  EVAL{step}: OSC={ev_o:.4f} ({math.exp(min(ev_o,20)):.1f}ppl) | BASE={ev_b:.4f} ({math.exp(min(ev_b,20)):.1f}ppl) | GAP={ev_b-ev_o:+.4f}')
        if ev_o<best_osc or ev_b<base_best:
            if ev_o<=ev_b: log(f'  -> OSC WINS')
            else: log(f'  -> BASE WINS')

        # Gen samples
        try:
            with torch.no_grad():
                for p_ in["Il etait une fois","Le chat est assis sur"]:
                    ids=tok.encode(p_,return_tensors='pt').to(device)
                    ids2=ids.clone()
                    for _ in range(40):
                        inp=ids[:,-BLOCK:]
                        lg=osc(inp)[:,-1,:]/0.8;v,_=torch.topk(lg,50);lg[lg<v[0,-1]]=-float('inf')
                        nxt=torch.multinomial(torch.softmax(lg,-1),1);ids=torch.cat([ids,nxt],1)
                        inp2=ids2[:,-BLOCK:]
                        lg2=base(inp2)[:,-1,:]/0.8;v2,_=torch.topk(lg2,50);lg2[lg2<v2[0,-1]]=-float('inf')
                        nxt2=torch.multinomial(torch.softmax(lg2,-1),1);ids2=torch.cat([ids2,nxt2],1)
                    og=tok.decode(ids[0].cpu().tolist(),skip_special_tokens=True)[:100]
                    bg=tok.decode(ids2[0].cpu().tolist(),skip_special_tokens=True)[:100]
                    log(f'  OSC: "{p_}" -> {og}')
                    log(f'  BAS: "{p_}" -> {bg}')
        except:pass
        osc.train();base.train()

        json.dump(results,open(f'{RESULT_DIR}/ablation.json','w'))

        # Save bests
        if ev_o<=best_osc:
            torch.save({'step':step,'model':osc.state_dict(),'loss':ev_o},f'{RESULT_DIR}/osc_best.pt')
        if ev_b<=base_best:
            torch.save({'step':step,'model':base.state_dict(),'loss':ev_b},f'{RESULT_DIR}/base_best.pt')

log(f'\n=== DONE ===')
log(f'Osc best: {best_osc:.4f} | Baseline best: {base_best:.4f}')
log(f'Final gap: {base_best-best_osc:+.4f}')
LF.close()
