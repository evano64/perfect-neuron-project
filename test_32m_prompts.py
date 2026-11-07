import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VariantB
from transformers import AutoTokenizer

N_GEN = 120
TEMP = 0.8

tok = AutoTokenizer.from_pretrained('camembert-base')

# CamemBERT: tokenizer.encode("texte") = [0 (cls), ...tokens..., 2 (sep)]
# Donc PAS besoin d'ajouter <s> ou </s> manuellement

# 4 variantes de prompts
prompts = [
    "Il etait une fois",                    # sans speciaux
    "La France est un pays",                # sans speciaux
    "Bonjour, comment allez-vous",          # conversation
    "Le chat est assis sur le canape",      # phrase simple
]

print("=" * 70)
print("32M — Test sans token special </s> dans les prompts")
print("=" * 70)

ckpt = torch.load('results_2b/ckpt_B_latest.pt', map_location='cpu', weights_only=False)
print(f"Step: {ckpt.get('step','?')}, Loss: {ckpt.get('loss','?')}")

m = VariantB(32004)
m.load_state_dict(ckpt['model'], strict=False)
m.float().eval()
del ckpt

for p in prompts:
    # Encoder SANS ajouter de tokens speciaux
    out = tok.encode(p, return_tensors='pt')
    prompt_len = out.shape[1]

    t0 = time.time()
    with torch.no_grad():
        for i in range(N_GEN):
            logits = m(out)
            probs = torch.softmax(logits[:, -1, :] / TEMP, dim=-1)
            v, _ = torch.topk(probs, 50)
            probs[probs < v[:, -1:]] = 0
            probs = probs / probs.sum(-1, keepdim=True)
            ntx = torch.multinomial(probs, 1).clamp(0, 32003)
            out = torch.cat([out, ntx], dim=1)
    elapsed = time.time() - t0
    text = tok.decode(out[0])
    gen_ids = out[0].tolist()[prompt_len:]
    unique = len(set(gen_ids))
    bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
    repeat = (1 - len(set(bigrams)) / max(len(bigrams), 1)) * 100

    print(f'\n[{elapsed:.0f}s] "{p}"')
    print(f'  tokens: {out.shape[1]} total ({prompt_len} prompt + {len(gen_ids)} gen)')
    print(f'  -> {text}')
    print(f'  unique={unique}/{N_GEN} ({unique/N_GEN*100:.0f}%) | repeat={repeat:.0f}%')

# Aussi tester avec le token initial <s> explicite (ce que faisait avant)
print("\n" + "=" * 70)
print("AVEC </s> (ancienne methode, pour comparaison)")
print("=" * 70)

p = "Il etait une fois"
# Ancienne methode: ajouter les tokens speciaux manuellement
out_old = tok.encode(p, return_tensors='pt')
# CamemBERT encode deja avec [0] = <s> au debut
# L'ancien code faisait tok.encode() qui ajoute [0, ..., 2]
# Donc le </s> etait deja la, on generait APRES le </s>

t0 = time.time()
with torch.no_grad():
    for i in range(N_GEN):
        logits = m(out_old)
        probs = torch.softmax(logits[:, -1, :] / TEMP, dim=-1)
        v, _ = torch.topk(probs, 50)
        probs[probs < v[:, -1:]] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        ntx = torch.multinomial(probs, 1).clamp(0, 32003)
        out_old = torch.cat([out_old, ntx], dim=1)
elapsed = time.time() - t0
text_old = tok.decode(out_old[0])
gen_ids_old = out_old[0].tolist()[out.shape[1]:]
unique_old = len(set(gen_ids_old))
bigrams_old = [(gen_ids_old[i], gen_ids_old[i+1]) for i in range(len(gen_ids_old)-1)]
repeat_old = (1 - len(set(bigrams_old)) / max(len(bigrams_old), 1)) * 100

print(f'\n[{elapsed:.0f}s] "{p}" (avec sep token)')
print(f'  -> {text_old}')
print(f'  unique={unique_old}/{N_GEN} ({unique_old/N_GEN*100:.0f}%) | repeat={repeat_old:.0f}%')

del m
