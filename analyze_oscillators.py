import sys, os, math, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F

# ===== LOAD MODEL =====
CKPT = 'results_depth_8l/best.pt'
print(f"Loading {CKPT} ...")
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

# Build model
from model_def import OscModel
model = OscModel(vocab=32005, d_model=800, n_osc=1024, n_layers=8, n_heads=16, max_len=256)
model.load_state_dict(sd, strict=False)
model.eval()

print(f"Model loaded. Step: {ckpt.get('step', '?')}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print()

# ===== ANALYZE OSCILLATOR PARAMETERS =====
print("=" * 70)
print("ANALYSE DES PARAMÈTRES DES OSCILLATEURS")
print("=" * 70)

for i, layer in enumerate(model.layers):
    osc = layer['osc']
    print(f"\n--- Couche {i} ---")

    # 1. Fréquences (omega)
    omega = osc.omega.detach().numpy()
    print(f"  Fréquences (omega):")
    print(f"    Min: {omega.min():.4f}  Max: {omega.max():.4f}  Mean: {omega.mean():.4f}")
    print(f"    Std: {omega.std():.4f}")
    print(f"    Unique values: {len(np.unique(np.round(omega, 3)))}/{len(omega)}")
    # Check if frequencies are diverse or collapsed
    hist, edges = np.histogram(omega, bins=20)
    print(f"    Distribution: {hist.tolist()}")

    # 2. Decay (logit → sigmoid)
    decay_logit = osc.decay_logit.detach().numpy()
    decay = 1 / (1 + np.exp(-decay_logit))  # sigmoid
    print(f"  Decay (logit):")
    print(f"    Min: {decay_logit.min():.4f}  Max: {decay_logit.max():.4f}  Mean: {decay_logit.mean():.4f}")
    print(f"  Decay (sigmoid):")
    print(f"    Min: {decay.min():.4f}  Max: {decay.max():.4f}  Mean: {decay.mean():.4f}")
    print(f"    <0.5 (fast forget): {(decay < 0.5).sum()}/{len(decay)}")
    print(f"    >0.5 (slow forget): {(decay > 0.5).sum()}/{len(decay)}")
    print(f"    >0.9 (very slow): {(decay > 0.9).sum()}/{len(decay)}")

    # 3. Coupling matrix K
    K = osc._K().detach().numpy()
    print(f"  Couplage (K):")
    print(f"    Norm: {np.linalg.norm(K):.4f}")
    print(f"    Max abs: {np.abs(K).max():.4f}")
    print(f"    Mean abs: {np.abs(K).mean():.4f}")
    # Spectral analysis of K
    eigvals = np.linalg.eigvals(K)
    print(f"    Eigenvalues: real max={eigvals.real.max():.4f}, imag max={eigvals.imag.max():.4f}")
    print(f"    Rank: {np.linalg.matrix_rank(K, tol=1e-3)}/{K.shape[0]}")

    # 4. Input projection B
    B_re = osc.B_re.detach().numpy()
    B_im = osc.B_im.detach().numpy()
    print(f"  Projection B:")
    print(f"    B_re norm: {np.linalg.norm(B_re):.4f}")
    print(f"    B_im norm: {np.linalg.norm(B_im):.4f}")
    print(f"    Ratio B_im/B_re: {np.linalg.norm(B_im)/np.linalg.norm(B_re):.4f}")

    # 5. Output projection C
    C = osc.C.detach().numpy()
    print(f"  Projection C:")
    print(f"    Norm: {np.linalg.norm(C):.4f}")
    print(f"    Shape: {C.shape}")

print()

# ===== ANALYZE DYNAMICS ON REAL TEXT =====
print("=" * 70)
print("ANALYSE DES DYNAMIQUES SUR DU TEXTE RÉEL")
print("=" * 70)

# Tokenizer (simple)
# Simple character-level tokenizer for analysis
def simple_tokenize(text):
    return torch.tensor([[ord(c) % 32005 for c in text]])

prompts = [
    "Bonjour tout le monde",
    "Le chat est assis sur",
    "Il était une fois",
    "L'intelligence artificielle",
    "La signification de la vie est",
]

for prompt in prompts:
    print(f"\n>>> {prompt}")
    ids = simple_tokenize(prompt)
    B, T = ids.shape

    # Run model and capture oscillator states
    with torch.no_grad():
        x = model.embed(ids) + model.pos(torch.arange(T).unsqueeze(0))
        x = model.dropout(x)

        for i, blk in enumerate(model.layers):
            # Oscillator
            o, hc = blk['osc'](blk['norm1'](x), None)

            # Analyze the oscillator output
            o_np = o[0].numpy()  # shape (T, d_model)
            print(f"  Couche {i}:")
            print(f"    Output norm per position: {[f'{np.linalg.norm(o_np[t]):.2f}' for t in range(min(T, 8))]}")
            print(f"    Output mean: {o_np.mean():.4f}  std: {o_np.std():.4f}")

            # Check if oscillator is doing something different from input
            x_norm = np.linalg.norm(x[0].numpy(), axis=1)
            o_norm = np.linalg.norm(o_np, axis=1)
            print(f"    Input norm: {[f'{x_norm[t]:.2f}' for t in range(min(T, 8))]}")
            print(f"    Oscillator norm: {[f'{o_norm[t]:.2f}' for t in range(min(T, 8))]}")

            # Update x
            x = x + model.ls_osc[i](blk['drop'](o))

            # Attention
            h = blk['norm3'](x)
            B2, T2, D = h.shape
            qkv = blk['qkv'](h).reshape(B2, T2, 3, model.n_heads, model.head_dim)
            q, k, v = qkv.unbind(2)
            a = F.scaled_dot_product_attention(
                q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                dropout_p=0.0, is_causal=True
            )
            x = x + model.ls_attn[i](blk['proj'](blk['drop'](a.permute(0,2,1,3).reshape(B2, T2, D))))
            x = blk['norm2'](x)

print()

# ===== FREQUENCY ANALYSIS =====
print("=" * 70)
print("ANALYSE FRÉQUENTIELLE DES OSCILLATEURS")
print("=" * 70)

for i, layer in enumerate(model.layers):
    osc = layer['osc']
    omega = osc.omega.detach().numpy()
    decay = torch.sigmoid(osc.decay_logit).detach().numpy()

    # Group by frequency bands
    low = omega < math.pi / 3
    mid = (omega >= math.pi / 3) & (omega < 2 * math.pi / 3)
    high = omega >= 2 * math.pi / 3

    print(f"\nCouche {i}:")
    print(f"  Basse fréquence (<π/3): {low.sum()} osc, decay moyen={decay[low].mean():.4f}")
    print(f"  Moyenne fréquence (π/3-2π/3): {mid.sum()} osc, decay moyen={decay[mid].mean():.4f}")
    print(f"  Haute fréquence (>2π/3): {high.sum()} osc, decay moyen={decay[high].mean():.4f}")

    # Correlation between frequency and decay
    corr = np.corrcoef(omega, decay)[0, 1]
    print(f"  Corrélation fréquence-decay: {corr:.4f}")

print()

# ===== COUPLING ANALYSIS =====
print("=" * 70)
print("ANALYSE DU COUPLAGE (K)")
print("=" * 70)

for i, layer in enumerate(model.layers):
    osc = layer['osc']
    K = osc._K().detach().numpy()
    omega = osc.omega.detach().numpy()

    # Find strongest couplings
    K_abs = np.abs(K)
    np.fill_diagonal(K_abs, 0)

    # Top 10 couplings
    flat_idx = np.argsort(K_abs.ravel())[-10:]
    rows, cols = np.unravel_index(flat_idx, K.shape)

    print(f"\nCouche {i}:")
    print(f"  Top 10 couplages (fort → faible):")
    for r, c in zip(rows[::-1], cols[::-1]):
        if r < c:  # upper triangle only
            print(f"    osc[{r}] (ω={omega[r]:.3f}) ↔ osc[{c}] (ω={omega[c]:.3f}): K={K[r,c]:.4f}")

    # Average coupling by frequency difference
    freq_diff = np.abs(omega[:, None] - omega[None, :])
    mask = np.triu(np.ones_like(freq_diff, dtype=bool), k=1)
    avg_coupling_by_freqdiff = []
    for d in np.linspace(0, math.pi, 10):
        mask_d = mask & (np.abs(freq_diff - d) < 0.3)
        if mask_d.sum() > 0:
            avg_coupling_by_freqdiff.append((d, np.abs(K[mask_d]).mean()))
    if avg_coupling_by_freqdiff:
        print(f"  Couplage moyen par différence de fréquence:")
        for d, c in avg_coupling_by_freqdiff:
            print(f"    Δω={d:.2f}: |K|={c:.6f}")

print()

# ===== CRITICAL: STATE ACCUMULATION ANALYSIS =====
print("=" * 70)
print("ANALYSE CRITIQUE : ACCUMULATION D'ÉTAT (le decay est-il appris ?)")
print("=" * 70)

for i, layer in enumerate(model.layers):
    osc = layer['osc']
    decay_logit = osc.decay_logit.detach().numpy()
    decay = 1 / (1 + np.exp(-decay_logit))

    print(f"\nCouche {i}:")
    print(f"  decay_logit: MIN={decay_logit.min():.6f} MAX={decay_logit.max():.6f} MEAN={decay_logit.mean():.6f}")
    print(f"  decay (sigmoid): MIN={decay.min():.6f} MAX={decay.max():.6f} MEAN={decay.mean():.6f}")
    print(f"  → TOUTS les oscillateurs ont decay={decay.mean():.4f} (jamais de oubli!)")

    # What does decay=0.9526 mean?
    half_life = -math.log(2) / math.log(decay.mean())
    print(f"  → Demi-vie de l'état: {half_life:.1f} tokens")
    print(f"  → Après 256 tokens, il reste {decay.mean()**256*100:.1f}% de l'état initial")

print()

# ===== STATE GROWTH ANALYSIS =====
print("=" * 70)
print("ANALYSE : CROISSANCE DE L'ÉTAT AU FIL DES TOKENS")
print("=" * 70)

# Run on a longer text and track oscillator state magnitude
text = "Le chat noir est assis sur le canapé. Il fait beau dehors. Les oiseaux chantent dans les arbres."
ids = simple_tokenize(text)
B, T = ids.shape

with torch.no_grad():
    x = model.embed(ids) + model.pos(torch.arange(T).unsqueeze(0))
    x = model.dropout(x)

    for i, blk in enumerate(model.layers):
        # Run oscillator step by step to track state
        osc = blk['osc']
        norm1 = blk['norm1']
        h_carry = None

        x_norm = norm1(x)
        decay = torch.sigmoid(osc.decay_logit)
        omega = osc.omega
        a = torch.polar(decay.float(), omega.float())
        K = osc._K()
        B_re = osc.B_re
        B_im = osc.B_im

        Bsz, T_step, _ = x_norm.shape
        u = (torch.einsum('btd,od->bto', x_norm.float(), B_re.float()) +
             1j * torch.einsum('btd,od->bto', x_norm.float(), B_im.float()))
        hc = torch.zeros(Bsz, osc.n_osc, dtype=torch.cfloat)
        la = torch.log(a)

        print(f"\nCouche {i} - Évolution de l'état:")
        state_norms = []
        for t in range(T_step):
            ut = u[:, t:t+1, :]
            ap = la.unsqueeze(0)
            hc = torch.exp(ap) * hc + ut
            hc = hc + 0.1 * torch.einsum('ij,bj->bi', K.to(hc.dtype), hc)
            state_norms.append(hc[0].abs().mean().item())

        print(f"  |z| par token: ", end="")
        for t, s in enumerate(state_norms):
            print(f"t{t}={s:.2f}", end=" ")
        print()
        print(f"  Croissance: t0={state_norms[0]:.2f} → t{T_step-1}={state_norms[-1]:.2f} (×{state_norms[-1]/state_norms[0]:.1f})")

        # Update x for next layer
        feats = torch.cat([hc.real, hc.imag, hc.abs()], dim=-1)
        o = (feats @ osc.C.t().float()).to(x.dtype)
        x = x + model.ls_osc[i](blk['drop'](o))

        # Attention
        h = blk['norm3'](x)
        B2, T2, D = h.shape
        qkv = blk['qkv'](h).reshape(B2, T2, 3, model.n_heads, model.head_dim)
        q, k, v = qkv.unbind(2)
        a = F.scaled_dot_product_attention(
            q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
            dropout_p=0.0, is_causal=True
        )
        x = x + model.ls_attn[i](blk['proj'](blk['drop'](a.permute(0,2,1,3).reshape(B2, T2, D))))
        x = blk['norm2'](x)

print()
print("DONE")
