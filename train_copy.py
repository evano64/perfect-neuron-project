"""
COPY TASK: Entraîner les oscillateurs à recracher ce qu'on leur donne.
Progression: 1 token → 2 tokens → 3... jusqu'à ce qu'ils échouent.
"""
import sys,os,math
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device=torch.device('cuda')

# ============================================================
# MODÈLES
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
    def __init__(self,vocab=32005,d_model=512,n_osc=512,n_layers=4,max_len=512,dropout_p=0.0):
        super().__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'osc':KuramotoOsc(d_model,n_osc),
                'norm1':nn.LayerNorm(d_model),'norm2':nn.LayerNorm(d_model),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
        self.ls_osc=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos)
        for i,blk in enumerate(self.layers):
            o,hc=blk['osc'](blk['norm1'](x),None)
            x=x+self.ls_osc[i](o)
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

class OscMLPModel(nn.Module):
    def __init__(self,vocab=32005,d_model=512,n_osc=512,n_layers=4,max_len=512,dropout_p=0.0):
        super().__init__()
        self.d_model=d_model
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'osc':KuramotoOsc(d_model,n_osc),
                'norm1':nn.LayerNorm(d_model),'norm2':nn.LayerNorm(d_model),
                'fc1':nn.Linear(d_model,4*d_model,bias=False),
                'fc2':nn.Linear(4*d_model,d_model,bias=False),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
        self.ls_osc=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
        self.ls_mlp=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos)
        for i,blk in enumerate(self.layers):
            o,hc=blk['osc'](blk['norm1'](x),None)
            x=x+self.ls_osc[i](o)
            m=F.silu(blk['fc1'](blk['norm2'](x)))
            x=x+self.ls_mlp[i](blk['fc2'](m))
        return self.head(self.norm_f(x))

# ============================================================
# GÉNÉRATION DES DONNÉES COPY TASK
# ============================================================
def generate_copy_batch(B, copy_len, vocab_size=32005, device='cuda'):
    """
    Génère un batch pour la tâche copy:
    [BOS] [tokens à copier × copy_len] [SEP] [tokens à copier × copy_len]
    
    Le modèle doit prédire les tokens à copier après le SEP.
    """
    BOS, SEP = 1, 2  # tokens spéciaux
    
    # Tokens aléatoires à copier (de 100 à vocab_size-1 pour éviter les tokens spéciaux)
    copy_tokens=torch.randint(100,vocab_size,(B,copy_len))
    
    # Séquence: [BOS][copy][SEP][copy]
    sep=torch.full((B,1),SEP,dtype=torch.long)
    bos=torch.full((B,1),BOS,dtype=torch.long)
    
    seq=torch.cat([bos,copy_tokens,sep,copy_tokens],dim=1).to(device)
    
    # Targets: on ne prédit que la 2ème moitié (après le SEP)
    # Le modèle reçoit toute la séquence, mais la loss n'est que sur la partie copy
    targets=seq[:,1:]  # shift by 1
    
    return seq, targets, copy_len

# ============================================================
# ENTRAÎNEMENT COPY TASK
# ============================================================
def train_copy_task(model, label, max_copy_len=200, steps_per_len=2000, 
                    eval_every=200, batch_size=32, lr=3e-4):
    """
    Entraîne le modèle sur la copy task avec des longueurs croissantes.
    """
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    
    results={}
    
    print(f"\n{'='*70}")
    print(f"COPY TASK: {label}")
    print(f"Max copy len: {max_copy_len}, Steps per len: {steps_per_len}")
    print(f"{'='*70}")
    
    for copy_len in range(1, max_copy_len+1):
        # Ajuster batch size pour la mémoire
        if copy_len > 100:
            bs=16
        elif copy_len > 50:
            bs=24
        else:
            bs=batch_size
        
        losses=[]
        correct_total=0
        total_tokens=0
        
        for step in range(steps_per_len):
            seq,targets,cl=generate_copy_batch(bs,copy_len,device=device)
            
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                logits=model(seq)
                # Loss uniquement sur la 2ème moitié (la copie)
                # La 2ème moitié commence après BOS + copy_len + SEP = copy_len + 2
                start_idx=cl+2  # position où commence la copie à prédire
                logits_copy=logits[:,start_idx-1:-1]  # shift pour prédiction
                targets_copy=targets[:,start_idx-1:]
                
                loss=F.cross_entropy(
                    logits_copy.reshape(-1,logits.size(-1)),
                    targets_copy.reshape(-1)
                )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Accuracy
            with torch.no_grad():
                preds=logits_copy.argmax(dim=-1)
                correct=(preds==targets_copy).sum().item()
                correct_total+=correct
                total_tokens+=targets_copy.numel()
        
        avg_loss=np.mean(losses[-100:])  # dernière fenêtre
        accuracy=correct_total/total_tokens if total_tokens>0 else 0
        
        results[copy_len]={'loss':avg_loss,'accuracy':accuracy}
        
        # Afficher tous les 10 ou si important
        if copy_len<=10 or copy_len%10==0 or copy_len==max_copy_len:
            print(f"  Copy len {copy_len:4d} → loss={avg_loss:.4f}, accuracy={accuracy:.2%}")
        
        # Si accuracy basse après beaucoup d'steps, on peut arrêter
        if copy_len>5 and accuracy<0.1:
            print(f"  ⚠️  Échoue à copy_len={copy_len} (accuracy={accuracy:.2%})")
            # Continuer quand même pour voir jusqu'où ça va
    
    return results

# ============================================================
# ÉVALUATION (sans entraînement)
# ============================================================
def eval_copy_task(model, label, copy_lengths=[1,2,3,5,10,20,50,100,150,200]):
    """
    Évalue la capacité de copy sans entraînement.
    """
    model.eval()
    results={}
    
    print(f"\n{'='*70}")
    print(f"ÉVALUATION COPY: {label}")
    print(f"{'='*70}")
    
    for cl in copy_lengths:
        seq,targets,_=generate_copy_batch(32,cl,device=device)
        
        with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
            logits=model(seq)
            start_idx=cl+2
            logits_copy=logits[:,start_idx-1:-1]
            targets_copy=targets[:,start_idx-1:]
            
            preds=logits_copy.argmax(dim=-1)
            correct=(preds==targets_copy).float().mean().item()
            
            # Top-5 accuracy
            top5=torch.topk(logits_copy,5,dim=-1).indices
            top5_correct=(top5==targets_copy.unsqueeze(-1)).any(dim=-1).float().mean().item()
        
        results[cl]={'accuracy':correct,'top5':top5_correct}
        print(f"  Copy len {cl:4d} → top1={correct:.2%}, top5={top5_correct:.2%}")
    
    return results

# ============================================================
# MAIN
# ============================================================
if __name__=='__main__':
    print("="*70)
    print("COPY TASK: TEST DE MÉMOIRE DES OSCILLATEURS")
    print("="*70)
    
    # Créer les modèles
    models={
        'OSC-ONLY': OscOnlyModel().to(device).to(torch.bfloat16),
        'OSC+MLP': OscMLPModel().to(device).to(torch.bfloat16),
    }
    
    # Paramètres
    all_results={}
    
    for label,model in models.items():
        print(f"\n{'='*70}")
        print(f"MODÈLE: {label}")
        np_=sum(p.numel() for p in model.parameters())
        print(f"Params: {np_:,}")
        print(f"{'='*70}")
        
        # Évaluer d'abord (avant entraînement)
        print("\n--- ÉVALUATION AVANT ENTRAÎNEMENT ---")
        eval_before=eval_copy_task(model,label,[1,2,3,5,10,20])
        
        # Entraîner sur copy task
        print("\n--- ENTRAÎNEMENT ---")
        train_results=train_copy_task(
            model,label,
            max_copy_len=50,      # jusqu'à 50 tokens à copier
            steps_per_len=500,    # 500 steps par longueur
            eval_every=100,
            batch_size=32,
            lr=5e-4
        )
        
        # Évaluer après entraînement
        print("\n--- ÉVALUATION APRÈS ENTRAÎNEMENT ---")
        eval_after=eval_copy_task(model,label,[1,2,3,5,10,20,30,40,50])
        
        all_results[label]={'train':train_results,'eval_before':eval_before,'eval_after':eval_after}
        
        # Sauvegarder
        os.makedirs('results_copy',exist_ok=True)
        torch.save({'model':model.state_dict(),'results':all_results[label]},
                   f'results_copy/{label.lower().replace("+","_")}.pt')
        
        del model
        torch.cuda.empty_cache()
    
    # ============================================================
    # COMPARAISON
    # ============================================================
    print("\n" + "="*70)
    print("COMPARAISON: COPY TASK")
    print("="*70)
    
    print("\nAccuracy après entraînement:")
    print(f"{'Copy len':>10}", end="")
    for label in all_results:
        print(f" | {label:>15}", end="")
    print()
    for cl in [1,2,3,5,10,20,30,40,50]:
        print(f"{cl:>10}", end="")
        for label in all_results:
            val=all_results[label]['eval_after'].get(cl,{}).get('accuracy',0)
            print(f" | {val:>15.2%}", end="")
        print()
    
    print("\n" + "="*70)
    print("INTERPRÉTATION:")
    print("- Si accuracy=100% → le modèle copie parfaitement")
    print("- Si accuracy chute après un certain len → mémoire limitée")
    print("- Compare OSC-ONLY vs OSC+MLP → le MLP aide-t-il la mémoire ?")
    print("="*70)
