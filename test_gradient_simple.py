"""
TEST DE GRADIENT SIMPLE: vérifie si les oscillateurs peuvent apprendre.
"""
import sys,os,math
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device('cuda')

class LayerScale(nn.Module):
    def __init__(self,d,init=1e-4):
        super().__init__()
        self.gamma=nn.Parameter(torch.full((d,),init))
    def forward(self,x):
        return x*self.gamma

class KuramotoOsc(nn.Module):
    def __init__(self,d_model,n_osc,chunk=32):
        super().__init__()
        self.n_osc,self.chunk=n_osc,chunk
        self.omega=nn.Parameter(torch.linspace(0.0,math.pi,n_osc))
        self.decay_logit=nn.Parameter(torch.full((n_osc,),0.5))
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

class OscOnlyModel(nn.Module):
    def __init__(self,vocab=32005,d_model=800,n_osc=1024,n_layers=8,max_len=256,dropout_p=0.1):
        super().__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.dropout=nn.Dropout(dropout_p)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'osc':KuramotoOsc(d_model,n_osc),
                'norm1':nn.LayerNorm(d_model),'norm2':nn.LayerNorm(d_model),
                'drop':nn.Dropout(dropout_p),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
        self.ls_osc=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos);x=self.dropout(x)
        for i,blk in enumerate(self.layers):
            o,hc=blk['osc'](blk['norm1'](x),None)
            x=x+self.ls_osc[i](blk['drop'](o))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

print("="*60)
print("TEST GRADIENT OSC-ONLY")
print("="*60)

model=OscOnlyModel().to(device).to(torch.bfloat16)
model.train()

for T in [10, 32, 64, 128, 256]:
    print(f"\n--- T={T} ---")
    try:
        tokens=torch.randint(100,30000,(1,T)).to(device)
        targets=torch.randint(100,30000,(1,T)).to(device)
        
        model.zero_grad()
        
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            logits=model(tokens)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        
        loss.backward()
        
        # Gradient norms
        grad_norms={}
        for name,p in model.named_parameters():
            if p.grad is not None:
                grad_norms[name]=p.grad.float().norm().item()
        
        # Oscillator gradients
        print(f"  Loss: {loss.item():.4f}")
        for i in range(len(model.layers)):
            for n in ['omega','decay_logit','B_re','B_im','K_raw','C']:
                key=f'layers.{i}.osc.{n}'
                if key in grad_norms:
                    print(f"    {key}: {grad_norms[key]:.6f}")
        
        # Check for NaN
        has_nan=any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
        print(f"  NaN gradients: {has_nan}")
        
        del logits,loss
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n\nTest terminé !")
