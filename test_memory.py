"""
Test mémoire des oscillateurs : capacité de rétention sur séquences longues.
Teste OSC-only et OSC+MLP avec distances croissantes key→query.
"""
import sys,os,math
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device('cuda')

# ============================================================
# DÉFINITION DES MODÈLES (copiés pour éviter l'import des scripts d'entraînement)
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

class OscModel8L(nn.Module):
    def __init__(self,vocab=32005,d_model=800,n_osc=1024,n_layers=8,n_heads=8,max_len=256,dropout_p=0.1):
        super().__init__()
        self.d_model=d_model;self.n_heads=n_heads;self.head_dim=d_model//n_heads
        self.embed=nn.Embedding(vocab,d_model);self.pos=nn.Embedding(max_len,d_model)
        self.dropout=nn.Dropout(dropout_p)
        self.layers=nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'osc':KuramotoOsc(d_model,n_osc),
                'norm1':nn.LayerNorm(d_model),'norm2':nn.LayerNorm(d_model),
                'qkv':nn.Linear(d_model,3*d_model,bias=False),'proj':nn.Linear(d_model,d_model,bias=False),
                'norm3':nn.LayerNorm(d_model),'drop':nn.Dropout(dropout_p),
            }))
        self.norm_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False);self.head.weight=self.embed.weight
        self.ls_osc=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
        self.ls_attn=nn.ModuleList([LayerScale(d_model,1e-4) for _ in range(n_layers)])
    def forward(self,tokens):
        B,T=tokens.shape;pos=torch.arange(T,device=tokens.device).unsqueeze(0)
        x=self.embed(tokens)+self.pos(pos);x=self.dropout(x)
        for i,blk in enumerate(self.layers):
            o,hc=blk['osc'](blk['norm1'](x),None);x=x+self.ls_osc[i](blk['drop'](o))
            h=blk['norm3'](x);B2,T2,D=h.shape
            qkv=blk['qkv'](h).reshape(B2,T2,3,self.n_heads,self.head_dim);q,k,v=qkv.unbind(2)
            a=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),dropout_p=0.0,is_causal=True)
            x=x+self.ls_attn[i](blk['proj'](blk['drop'](a.permute(0,2,1,3).reshape(B2,T2,D))))
            x=blk['norm2'](x)
        return self.head(self.norm_f(x))

from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('camembert-base')

def load_model(cls,path,label):
    print(f"\n{'='*60}")
    print(f"Chargement: {label}")
    m=cls().to(device).to(torch.bfloat16)
    if os.path.exists(path):
        ckpt=torch.load(path,map_location=device)
        if 'model' in ckpt:
            m.load_state_dict(ckpt['model'],strict=False)
            print(f"  Checkpoint chargé: {path}")
        else:
            m.load_state_dict(ckpt,strict=False)
            print(f"  Poids chargés (raw dict)")
    else:
        print(f"  Pas de checkpoint → poids aléatoires (test architectural)")
    m.eval()
    return m

# ============================================================
# TEST 1: CAPACITÉ DE RÉTENTION (key→query distance)
# ============================================================
def test_retention(model,label,distances,n_trials=50):
    """
    Pour chaque distance d:
    - Insère un "token clé" à position 0
    - Après d tokens de bruit, demande de prédire le token suivant
    - Mesure la probabilité du token clé
    """
    model.eval()
    results={}
    vocab_size=32005

    for d in distances:
        correct=0
        key_probs=[]
        hidden_norms=[]

        for _ in range(n_trials):
            # Séquence: [BOS] [KEY] [bruit×d] [QUERY]
            key_token=torch.randint(100,30000,(1,))  # token aléatoire
            noise=torch.randint(0,100,(1,d))          # bruit

            # Query: le token après le bruit devrait "rappeler" le key
            seq=torch.cat([torch.tensor([[3]]),key_token.unsqueeze(0),noise],dim=1)  # 3=BOS

            with torch.no_grad():
                logits=model(seq.to(device))
                # Logits à la dernière position
                last_logits=logits[0,-1,:]

                # Probabilité du key_token
                key_prob=F.softmax(last_logits,dim=-1)[key_token.item()].item()
                key_probs.append(key_prob)

                # Hidden state norm
                if hasattr(model,'layers'):
                    # Hook pour capturer les hidden states
                    pass

        avg_key_prob=np.mean(key_probs)
        results[d]=avg_key_prob
        print(f"  Distance {d:4d} → P(key)={avg_key_prob:.6f} ({avg_key_prob*100:.3f}%)")

    return results

# ============================================================
# TEST 2: CAPACITÉ DE MÉMOIRE (répétition du même token)
# ============================================================
def test_repetition(model,label,sizes,n_trials=30):
    """
    Teste si le modèle peut "compter" combien de fois un token apparaît.
    Séquence: [BOS] [X] × N → prédit N
    """
    model.eval()
    results={}
    target_token=1234  # token fixe

    for n in sizes:
        correct=0
        for _ in range(n_trials):
            # Créer séquence avec N répétitions du target
            seq=torch.cat([torch.tensor([[3]]),
                          torch.full((1,n),target_token,dtype=torch.long)],dim=1)

            with torch.no_grad():
                logits=model(seq.to(device))
                pred=logits[0,-1,:].argmax().item()

                # Vérifier si le modèle prédit quelque chose de sensé
                # (pas forcément N, mais mesurons la distribution)
                probs=F.softmax(logits[0,-1,:],dim=-1)
                top5=torch.topk(probs,5)
                top5_tokens=top5.indices.cpu().tolist()
                top5_probs=top5.values.cpu().tolist()

        # Mesurer l'entropie de la sortie (plus basse = plus confiant)
        with torch.no_grad():
            logits=model(seq.to(device))
            probs=F.softmax(logits[0,-1,:],dim=-1)
            entropy=-(probs*torch.log(probs+1e-10)).sum().item()
            avg_prob=probs.mean().item()

        results[n]={'entropy':entropy,'avg_prob':avg_prob}
        print(f"  Taille {n:4d} → entropy={entropy:.3f}, P_moy={avg_prob:.6f}")

    return results

# ============================================================
# TEST 3: ÉTAT DES OSCILLATEURS (normes, decay effectif)
# ============================================================
def test_oscillator_states(model,label,distances):
    """
    Analyse les états internes des oscillateurs pour différentes longueurs.
    """
    model.eval()
    results={}

    for d in distances:
        seq=torch.randint(100,30000,(1,d+1))
        seq[0,0]=3  # BOS

        with torch.no_grad():
            # Passer couche par couche
            x=model.embed(seq.to(device))+model.pos(torch.arange(d+1,device=device).unsqueeze(0))
            x=model.dropout(x)

            layer_states=[]
            for i,blk in enumerate(model.layers):
                if hasattr(blk,'osc'):
                    o,hc=blk['osc'](blk['norm1'](x),None)
                    # Mesurer norme de sortie
                    out_norm=o.float().norm().item()
                    # Mesurer norme du hidden carry
                    carry_norm=hc.abs().norm().item() if hc is not None else 0
                    layer_states.append({
                        'out_norm':out_norm,
                        'carry_norm':carry_norm,
                        'decay_effective':torch.sigmoid(blk['osc'].decay_logit).mean().item()
                    })
                    # Residual
                    x=x+blk['drop'](o)
                    if hasattr(blk,'fc1'):  # OSC+MLP
                        m=blk['drop'](F.silu(blk['fc1'](blk['norm2'](x))))
                        x=x+blk['fc2'](m)

        results[d]=layer_states
        print(f"\n  Distance {d}:")
        for i,s in enumerate(layer_states):
            print(f"    Layer {i}: out_norm={s['out_norm']:.2f}, carry_norm={s['carry_norm']:.2f}, decay={s['decay_effective']:.4f}")

    return results

# ============================================================
# TEST 4: COPY TASK (la tâche de mémoire canonique)
# ============================================================
def test_copy_task(model,label,seq_lengths=[10,20,50,100,200],n_trials=20):
    """
    Tâche copy: [BOS] [A][B][C] [bruit×N] → prédit [A][B][C]
    Mesure la capacité à recopier une séquence après du bruit.
    """
    model.eval()
    results={}

    for seq_len in seq_lengths:
        copy_len=min(5,seq_len//3)  # Longueur de la séquence à copier
        correct=0

        for _ in range(n_trials):
            # Tokens à copier
            copy_tokens=torch.randint(100,30000,(1,copy_len))
            # Bruit
            noise_len=seq_len-copy_len
            noise=torch.randint(0,100,(1,noise_len))

            # Séquence complète
            seq=torch.cat([torch.tensor([[3]]),copy_tokens,noise],dim=1)

            with torch.no_grad():
                logits=model(seq.to(device))

                # Pour chaque token à copier, vérifier si le modèle le prédit
                # On regarde les logits après le bruit
                # Mais ici on a un seul token de sortie → on regarde si le premier token
                # de la séquence copiée est dans le top-k
                last_logits=logits[0,-1,:]
                probs=F.softmax(last_logits,dim=-1)

                # Le premier token à copier devrait être prédit
                top10=torch.topk(probs,10)
                if copy_tokens[0,0].item() in top10.indices.cpu().tolist():
                    correct+=1

        accuracy=correct/n_trials
        results[seq_len]=accuracy
        print(f"  Seq len {seq_len:4d} (copy {copy_len}) → accuracy={accuracy:.2%}")

    return results

# ============================================================
# MAIN
# ============================================================
if __name__=='__main__':
    print("="*60)
    print("TEST MÉMOIRE DES OSCILLATEURS")
    print("="*60)

    # Charger les modèles
    models={
        '8L ATT+OSC': (OscModel8L,'results_depth_8l/best.pt'),
        'OSC-ONLY': (OscOnlyModel,'results_osc_only/best.pt'),
        'OSC+MLP': (OscMLPModel,'results_osc_mlp/best.pt'),
    }

    all_results={}
    for label,(cls,path) in models.items():
        model=load_model(cls,path,label)

        print(f"\n{'='*60}")
        print(f"TESTS POUR: {label}")
        print(f"{'='*60}")

        # Test 1: Rétention
        print(f"\n--- Test 1: Rétention (key→query) ---")
        distances=[10,20,50,100,150,200,250]
        ret=test_retention(model,label,distances)

        # Test 2: Répétition
        print(f"\n--- Test 2: Répétition ---")
        sizes=[5,10,20,50,100]
        rep=test_repetition(model,label,sizes)

        # Test 3: États oscillateurs
        print(f"\n--- Test 3: États oscillateurs ---")
        osc=test_oscillator_states(model,label,distances[:4])

        # Test 4: Copy task
        print(f"\n--- Test 4: Copy task ---")
        copy=test_copy_task(model,label,[10,20,50,100,200])

        all_results[label]={'retention':ret,'repetition':rep,'oscillator':osc,'copy':copy}

        # Nettoyer GPU
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # RÉSUMÉ COMPARATIF
    # ============================================================
    print("\n" + "="*60)
    print("RÉSUMÉ COMPARATIF")
    print("="*60)

    print("\n--- Rétention (P(key) à distance d) ---")
    print(f"{'Distance':>10}", end="")
    for label in all_results:
        print(f" | {label:>12}", end="")
    print()
    for d in [10,20,50,100,200]:
        print(f"{d:>10}", end="")
        for label in all_results:
            val=all_results[label]['retention'].get(d,0)
            print(f" | {val:>12.6f}", end="")
        print()

    print("\n--- Copy Task (accuracy) ---")
    print(f"{'Seq len':>10}", end="")
    for label in all_results:
        print(f" | {label:>12}", end="")
    print()
    for sl in [10,20,50,100,200]:
        print(f"{sl:>10}", end="")
        for label in all_results:
            val=all_results[label]['copy'].get(sl,0)
            print(f" | {val:>12.2%}", end="")
        print()

    print("\n--- Decay effectif moyen par couche (distance=100) ---")
    for label in all_results:
        print(f"\n{label}:")
        if 100 in all_results[label]['oscillator']:
            for i,s in enumerate(all_results[label]['oscillator'][100]):
                print(f"  Layer {i}: decay={s['decay_effective']:.4f}")

    print("\n" + "="*60)
    print("INTERPRÉTATION:")
    print("Si P(key) chute avec la distance → pas de mémoire longue")
    print("Si copy accuracy chute → les oscillateurs ne copient pas")
    print("Si decay → 0 vite → l'oscillator oublie tout")
    print("="*60)
