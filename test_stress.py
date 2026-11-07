"""
TEST DE STRESS: Séquences de plus en plus longues.
Mesure si l'oscillator explose, se stabilise, ou dégénère.
"""
import sys,os,math
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
device=torch.device('cuda')
torch.cuda.reset_peak_memory_stats()

# ============================================================
# MODÈLES (copiés directement)
# ============================================================
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

class OscMLPModel(nn.Module):
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
                'fc1':nn.Linear(d_model,4*d_model,bias=False),
                'fc2':nn.Linear(4*d_model,d_model,bias=False),
                'drop':nn.Dropout(dropout_p),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
        self.ls_osc=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
        self.ls_mlp=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos);x=self.dropout(x)
        for i,blk in enumerate(self.layers):
            o,hc=blk['osc'](blk['norm1'](x),None)
            x=x+self.ls_osc[i](blk['drop'](o))
            m=blk['drop'](F.silu(blk['fc1'](blk['norm2'](x))))
            x=x+self.ls_mlp[i](blk['fc2'](m))
        return self.head(self.norm_f(x))

# ============================================================
# TEST DE STRESS
# ============================================================
def stress_test(model,label,seq_lengths):
    """
    Pour chaque longueur de séquence:
    1. Feed un batch aléatoire
    2. Mesure: normes des sorties, gradient, NaN/Inf
    3. Vérifie si l'oscillator "survit"
    """
    model.eval()
    results={}
    
    for T in seq_lengths:
        print(f"\n  --- Séquence T={T} ---")
        try:
            # Batch aléatoire
            tokens=torch.randint(100,30000,(1,T)).to(device)
            tokens[0,0]=3  # BOS
            
            # Forward pass
            with torch.no_grad():
                logits=model(tokens)
                
            # Mesures
            out_norm=logits.float().norm().item()
            out_mean=logits.float().mean().item()
            out_std=logits.float().std().item()
            out_max=logits.float().abs().max().item()
            out_min=logits.float().abs().min().item()
            has_nan=torch.isnan(logits).any().item()
            has_inf=torch.isinf(logits).any().item()
            
            # Analyser chaque couche oscillateur
            layer_stats=[]
            x=model.embed(tokens)+model.pos(torch.arange(T,device=device).unsqueeze(0))
            x=model.dropout(x)
            
            for i,blk in enumerate(model.layers):
                o,hc=blk['osc'](blk['norm1'](x),None)
                
                # Normes
                o_norm=o.float().norm().item()
                hc_norm=hc.abs().norm().item() if hc is not None else 0
                o_nan=torch.isnan(o).any().item()
                
                # Decay effectif
                decay=torch.sigmoid(blk['osc'].decay_logit)
                decay_mean=decay.mean().item()
                decay_min=decay.min().item()
                decay_max=decay.max().item()
                
                # Fréquences
                omega=blk['osc'].omega.data
                freq_std=omega.std().item()
                
                layer_stats.append({
                    'o_norm':o_norm,
                    'hc_norm':hc_norm,
                    'o_nan':o_nan,
                    'decay_mean':decay_mean,
                    'decay_min':decay_min,
                    'decay_max':decay_max,
                    'freq_std':freq_std,
                })
                
                # Residual
                x=x+blk['drop'](o)
                if hasattr(blk,'fc1'):
                    m=blk['drop'](F.silu(blk['fc1'](blk['norm2'](x))))
                    x=x+blk['fc2'](m)
            
            results[T]={
                'out_norm':out_norm,
                'out_mean':out_mean,
                'out_std':out_std,
                'out_max':out_max,
                'has_nan':has_nan,
                'has_inf':has_inf,
                'layers':layer_stats,
            }
            
            status="OK" if not has_nan and not has_inf else "CRASH"
            print(f"    Status: {status}")
            print(f"    Logits: norm={out_norm:.2f}, mean={out_mean:.2f}, std={out_std:.2f}, max={out_max:.2f}")
            print(f"    NaN={has_nan}, Inf={has_inf}")
            for j,s in enumerate(layer_stats):
                print(f"    Layer {j}: osc_out={s['o_norm']:.2f}, carry={s['hc_norm']:.2f}, "
                      f"decay=[{s['decay_min']:.3f}-{s['decay_max']:.3f}], freq_std={s['freq_std']:.3f}")
                
        except RuntimeError as e:
            print(f"    CRASH: {e}")
            results[T]={'crash':str(e)}
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results[T]={'error':str(e)}
    
    return results

# ============================================================
# TEST DE GRADIENT (peut-il apprendre sur séquences longues ?)
# ============================================================
def gradient_test(model,label,seq_lengths):
    """
    Mesure les gradients pour voir si le modèle peut apprendre
    sur des séquences de plus en plus longues.
    """
    model.train()
    results={}
    
    for T in seq_lengths:
        print(f"\n  --- Gradient T={T} ---")
        try:
            tokens=torch.randint(100,30000,(1,T)).to(device)
            targets=torch.randint(100,30000,(1,T)).to(device)
            
            model.zero_grad()
            logits=model(tokens)
            loss_val=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
            loss_val.backward()
            
            # Gradient norms par couche
            grad_norms={}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    grad_norms[name]=p.grad.float().norm().item()
            
            # Gradient de l'oscillator spécifiquement
            osc_grads=[grad_norms.get(f'layers.{i}.osc.{n}',0) 
                       for i in range(len(model.layers)) 
                       for n in ['omega','decay_logit','B_re','B_im','K_raw','C']]
            
            total_osc_grad=sum(osc_grads)
            max_osc_grad=max(osc_grads) if osc_grads else 0
            
            # Nettoyer
            del logits,loss_val
            torch.cuda.empty_cache()
            
            results[T]={
                'loss':loss_val.item(),
                'total_osc_grad':total_osc_grad,
                'max_osc_grad':max_osc_grad,
                'has_nan':any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None),
            }
            
            print(f"    Loss: {loss_val.item():.4f}")
            print(f"    Osc grad total: {total_osc_grad:.6f}, max: {max_osc_grad:.6f}")
            print(f"    NaN gradients: {results[T]['has_nan']}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results[T]={'error':str(e)}
    
    model.eval()
    return results

# ============================================================
# MAIN
# ============================================================
if __name__=='__main__':
    print("="*70)
    print("TEST DE STRESS: CAPACITÉ DES OSCILLATEURS")
    print("="*70)
    
    # Modèles
    models={
        'OSC-ONLY': OscOnlyModel().to(device).to(torch.bfloat16),
        'OSC+MLP': OscMLPModel().to(device).to(torch.bfloat16),
    }
    
    # Longueurs de test
    seq_lengths=[10, 32, 64, 128, 256]
    
    all_results={}
    for label,model in models.items():
        print(f"\n{'='*70}")
        print(f"MODÈLE: {label}")
        print(f"{'='*70}")
        
        print("\n--- TEST 1: FORWARD PASS (stress) ---")
        fwd=stress_test(model,label,seq_lengths)
        
        print("\n--- TEST 2: GRADIENT (apprentissage) ---")
        grad=gradient_test(model,label,seq_lengths)
        
        all_results[label]={'forward':fwd,'gradient':grad}
        
        del model
        torch.cuda.empty_cache()
    
    # ============================================================
    # RÉSUMÉ
    # ============================================================
    print("\n" + "="*70)
    print("RÉSUMÉ: SURVIE DES OSCILLATEURS")
    print("="*70)
    
    print("\nForward pass (norme des logits):")
    print(f"{'T':>6}", end="")
    for label in all_results:
        print(f" | {label:>15}", end="")
    print()
    for T in seq_lengths:
        print(f"{T:>6}", end="")
        for label in all_results:
            r=all_results[label]['forward'].get(T,{})
            if 'crash' in r or 'error' in r:
                print(f" | {'CRASH':>15}", end="")
            elif 'out_norm' in r:
                print(f" | {r['out_norm']:>15.2f}", end="")
            else:
                print(f" | {'?':>15}", end="")
        print()
    
    print("\nGradient oscillateur (peut-il apprendre ?):")
    print(f"{'T':>6}", end="")
    for label in all_results:
        print(f" | {label:>15}", end="")
    print()
    for T in seq_lengths:
        print(f"{T:>6}", end="")
        for label in all_results:
            r=all_results[label]['gradient'].get(T,{})
            if 'error' in r:
                print(f" | {'ERROR':>15}", end="")
            elif 'total_osc_grad' in r:
                print(f" | {r['total_osc_grad']:>15.6f}", end="")
            else:
                print(f" | {'?':>15}", end="")
        print()
    
    print("\n" + "="*70)
    print("INTERPRÉTATION:")
    print("- Si norme explose → oscillator instable")
    print("- Si grad → 0 → ne peut plus apprendre")
    print("- Si crash → architecture ne supporte pas la longueur")
    print("="*70)
