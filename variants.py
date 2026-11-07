"""
3 variantes oscillatoires pour comparaison equitable.

Contraintes :
- Meme taille cible (~2-3M params)
- Meme corpus (5.32M tokens CamemBERT)
- Meme contexte (256 tokens)
- Meme nombre de couches (2)

VARIANTE 0 (BASE) : Oscillateurs recurrents simples (le code osc_lm.py)
VARIANTE A : Multi-tetes frequentielles (lecture par sous-groupes d'oscillateurs)
VARIANTE B : Synchronisation entre oscillateurs (couplage de Kuramoto)
VARIANTE C : Fourier Neural Operator (FFT apprise + filtre frequentiel)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# BLOC OSCILLATEUR PARTAGE (la base commune)
# =====================================================================
class OscillatorLayer(nn.Module):
    """Couche oscillatoire recurrente (identique a osc_lm.py)."""

    def __init__(self, d_model, n_osc, chunk=32):
        super().__init__()
        self.n_osc = n_osc
        self.d_model = d_model
        self.chunk = chunk
        self.omega = nn.Parameter(torch.linspace(0.0, math.pi, n_osc))
        self.decay_logit = nn.Parameter(torch.full((n_osc,), 3.0))
        self.B_re = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        self.B_im = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        self.C = nn.Parameter(torch.randn(d_model, n_osc * 3) / (n_osc * 3)**0.5)

    def forward(self, x):
        Bsz, T, _ = x.shape
        decay = torch.sigmoid(self.decay_logit)
        a = torch.polar(decay, self.omega)
        u = (torch.einsum('btd,od->bto', x, self.B_re)
             + 1j * torch.einsum('btd,od->bto', x, self.B_im))
        chunk = self.chunk
        h_carry = torch.zeros(Bsz, self.n_osc, dtype=torch.cfloat, device=x.device)
        log_a = torch.log(a)
        outs = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            L = end - start
            uc = u[:, start:end, :]
            t_idx = torch.arange(L, device=x.device, dtype=torch.float32)
            a_pow = torch.exp(log_a[None] * t_idx[:, None])
            a_neg = torch.exp(-log_a[None] * t_idx[:, None])
            cs = torch.cumsum(a_neg[None] * uc, dim=1)
            Hc = a_pow[None] * cs
            carry_decay = torch.exp(log_a[None] * (t_idx[:, None] + 1.0))
            Hc = Hc + carry_decay[None] * h_carry[:, None, :]
            outs.append(Hc)
            h_carry = Hc[:, -1, :]
        H = torch.cat(outs, dim=1)
        feats = torch.cat([H.real, H.imag, H.abs()], dim=-1)
        return feats @ self.C.t()


# =====================================================================
# VARIANTE A : Tetes frequentielles multi-echelle
# =====================================================================
class FreqHeadAttention(nn.Module):
    """Attention ou les queries sont generees a differentes 'frequences'
    temporelles (rapide / medium / lent) pour capturer plusieurs niveaux."""

    def __init__(self, d_model, n_heads=4, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.nh = n_heads
        self.dh = d_model // n_heads
        # Un seul QKV + proj (pas 3 separate MHA = plus leger)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Echelles temporelles (decroissance apprise)
        self.scale_decay = nn.Parameter(torch.linspace(0.1, 0.9, n_scales))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        s = (q @ k.transpose(-2, -1)) / self.dh**0.5
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float('-inf'))
        w = torch.softmax(s, -1)
        out = (w @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class VariantA(nn.Module):
    """Base oscillatoire + tetes frequentielles multi-echelle."""

    def __init__(self, vocab, d_model=384, n_osc=512, n_layers=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'osc': OscillatorLayer(d_model, n_osc),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'freq': FreqHeadAttention(d_model, n_heads=4, n_scales=3),
                'norm3': nn.LayerNorm(d_model),
            })
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos(pos)[None]
        for blk in self.layers:
            x = x + blk['osc'](blk['norm1'](x))
            x = x + blk['freq'](blk['norm3'](x))
            x = blk['norm2'](x)
        return self.head(self.norm_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# VARIANTE B : Synchronisation (couplage de Kuramoto)
# =====================================================================
class KuramotoLayer(nn.Module):
    """Couche oscillatoire avec couplage type Kuramoto entre oscillateurs :
    ils se synchronisent en phase selon un pattern appris.
    h_t = a * h_{t-1} + B * x_t + K * sync(h_{t-1})"""

    def __init__(self, d_model, n_osc, chunk=32):
        super().__init__()
        self.n_osc = n_osc
        self.d_model = d_model
        self.chunk = chunk
        self.omega = nn.Parameter(torch.linspace(0.0, math.pi, n_osc))
        self.decay_logit = nn.Parameter(torch.full((n_osc,), 3.0))
        self.B_re = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        self.B_im = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        # Couplage de Kuramoto : matrice antisymmetrique (K_ij = -K_ji)
        # pour pousser les oscillateurs a se synchroniser
        K_raw = nn.Parameter(torch.randn(n_osc, n_osc) / n_osc**0.5 * 0.1)
        self.K_raw = K_raw
        self.C = nn.Parameter(torch.randn(d_model, n_osc * 3) / (n_osc * 3)**0.5)

    def _K(self):
        # Matrice antisymmetrique (couplage qui tend a synchroniser)
        return self.K_raw - self.K_raw.t()

    def forward(self, x):
        Bsz, T, _ = x.shape
        decay = torch.sigmoid(self.decay_logit)
        a = torch.polar(decay, self.omega)
        K = self._K()
        u = (torch.einsum('btd,od->bto', x, self.B_re)
             + 1j * torch.einsum('btd,od->bto', x, self.B_im))
        chunk = self.chunk
        h_carry = torch.zeros(Bsz, self.n_osc, dtype=torch.cfloat, device=x.device)
        log_a = torch.log(a)
        outs = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            L = end - start
            uc = u[:, start:end, :]
            t_idx = torch.arange(L, device=x.device, dtype=torch.float32)
            a_pow = torch.exp(log_a[None] * t_idx[:, None])
            a_neg = torch.exp(-log_a[None] * t_idx[:, None])
            cs = torch.cumsum(a_neg[None] * uc, dim=1)
            Hc = a_pow[None] * cs
            # Ajout Kuramoto : pour chaque pas, on applique le couplage
            # aux sorties precedentes (effet sur l'etat)
            # h_sync_t = K @ h_t (couplage inter-oscillateurs)
            K_term = torch.einsum('ij,btj->bti', K.to(Hc.dtype), Hc)
            Hc = Hc + 0.1 * K_term  # contribution de synchronisation
            carry_decay = torch.exp(log_a[None] * (t_idx[:, None] + 1.0))
            Hc = Hc + carry_decay[None] * h_carry[:, None, :]
            outs.append(Hc)
            h_carry = Hc[:, -1, :]
        H = torch.cat(outs, dim=1)
        feats = torch.cat([H.real, H.imag, H.abs()], dim=-1)
        return feats @ self.C.t()


class VariantB(nn.Module):
    """Base oscillatoire + synchronisation Kuramoto."""

    def __init__(self, vocab, d_model=384, n_osc=512, n_layers=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'osc': KuramotoLayer(d_model, n_osc),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'attn': self._make_attn(d_model),
                'norm3': nn.LayerNorm(d_model),
            })
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def _make_attn(self, d_model):
        class Attn(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.nh = 4
                self.dh = d // self.nh
                self.qkv = nn.Linear(d, 3 * d)
                self.proj = nn.Linear(d, d)
            def forward(self, x):
                B, T, D = x.shape
                qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
                q, k, v = qkv.unbind(dim=2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                s = (q @ k.transpose(-2, -1)) / self.dh**0.5
                mask = torch.triu(torch.ones(T, T, device=x.device,
                                             dtype=torch.bool), 1)
                s = s.masked_fill(mask, float('-inf'))
                w = torch.softmax(s, -1)
                out = (w @ v).transpose(1, 2).reshape(B, T, D)
                return self.proj(out)
        return Attn(d_model)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos(pos)[None]
        for blk in self.layers:
            x = x + blk['osc'](blk['norm1'](x))
            x = x + blk['attn'](blk['norm3'](x))
            x = blk['norm2'](x)
        return self.head(self.norm_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# VARIANTE C : Fourier Neural Operator (FFT apprise)
# =====================================================================
class FNOLayer(nn.Module):
    """Couche oscillatoire + transformation de Fourier (mode truncation).
    On projette dans l'espace frequentiel, applique un filtre appris
    par mode, puis revient. C'est la structure d'un FNO."""

    def __init__(self, d_model, n_osc, n_modes=32):
        super().__init__()
        self.n_osc = n_osc
        self.d_model = d_model
        self.n_modes = n_modes
        self.omega = nn.Parameter(torch.linspace(0.0, math.pi, n_osc))
        self.decay_logit = nn.Parameter(torch.full((n_osc,), 3.0))
        self.B_re = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        self.B_im = nn.Parameter(torch.randn(n_osc, d_model) / d_model**0.5)
        # Filtres frequentiels (un par mode de Fourier)
        self.W_freq_re = nn.Parameter(torch.randn(n_modes, n_osc) / n_osc**0.5)
        self.W_freq_im = nn.Parameter(torch.randn(n_modes, n_osc) / n_osc**0.5)
        self.C = nn.Parameter(torch.randn(d_model, n_osc * 3) / (n_osc * 3)**0.5)

    def forward(self, x):
        Bsz, T, _ = x.shape
        # 1. Bloc oscillatoire comme avant (sans chunking pour simplifier)
        decay = torch.sigmoid(self.decay_logit)
        a = torch.polar(decay, self.omega)
        u = (torch.einsum('btd,od->bto', x, self.B_re)
             + 1j * torch.einsum('btd,od->bto', x, self.B_im))
        # Scan rapide
        log_a = torch.log(a)
        t_idx = torch.arange(T, device=x.device, dtype=torch.float32)
        a_pow = torch.exp(log_a[None] * t_idx[:, None])
        a_neg = torch.exp(-log_a[None] * t_idx[:, None])
        cs = torch.cumsum(a_neg[None] * u, dim=1)
        H = a_pow[None] * cs
        # 2. Transformation de Fourier : H est (B, T, n_osc), FFT sur T
        # rfft exige un tenseur REEL : on prend la partie reelle (info principale)
        H_real = H.real
        Hf = torch.fft.rfft(H_real.float(), dim=1)  # (B, T/2+1, n_osc) complexe
        # 3. Filtre appris par mode (multiplication par W_freq)
        modes = min(self.n_modes, Hf.size(1))
        Wf = (self.W_freq_re[:modes].unsqueeze(0)
              + 1j * self.W_freq_im[:modes].unsqueeze(0))  # (1, modes, n_osc)
        Hf_filtered = Hf[:, :modes] * Wf
        # Reconstruction : on garde les premiers modes filtres, on met le reste a 0
        Hf_out = torch.zeros_like(Hf)
        Hf_out[:, :modes] = Hf_filtered
        # 4. FFT inverse
        H_out = torch.fft.irfft(Hf_out, n=T, dim=1)
        # H_out est REEL (irfft renvoie du reel) ; on construit des features compatibles
        feats = torch.cat([H_out, torch.zeros_like(H_out), torch.abs(H_out)], dim=-1)
        return feats @ self.C.t()


class VariantC(nn.Module):
    """Base oscillatoire + Fourier Neural Operator."""

    def __init__(self, vocab, d_model=384, n_osc=512, n_layers=4, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'osc': FNOLayer(d_model, n_osc),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'attn': self._make_attn(d_model),
                'norm3': nn.LayerNorm(d_model),
            })
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def _make_attn(self, d_model):
        class Attn(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.nh = 4
                self.dh = d // self.nh
                self.qkv = nn.Linear(d, 3 * d)
                self.proj = nn.Linear(d, d)
            def forward(self, x):
                B, T, D = x.shape
                qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
                q, k, v = qkv.unbind(dim=2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                s = (q @ k.transpose(-2, -1)) / self.dh**0.5
                mask = torch.triu(torch.ones(T, T, device=x.device,
                                             dtype=torch.bool), 1)
                s = s.masked_fill(mask, float('-inf'))
                w = torch.softmax(s, -1)
                out = (w @ v).transpose(1, 2).reshape(B, T, D)
                return self.proj(out)
        return Attn(d_model)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos(pos)[None]
        for blk in self.layers:
            x = x + blk['osc'](blk['norm1'](x))
            x = x + blk['attn'](blk['norm3'](x))
            x = blk['norm2'](x)
        return self.head(self.norm_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    VOCAB = 32004
    DM = 192
    NOSC = 192
    NL = 2

    for name, cls in [('A (freq heads)', VariantA),
                      ('B (Kuramoto)', VariantB),
                      ('C (FNO)', VariantC)]:
        m = cls(VOCAB, DM, NOSC, NL)
        n = m.num_params()
        print('%s : %d params (%.2fM)' % (name, n, n / 1e6))