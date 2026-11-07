import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VariantB
from model_def import OscModel
from transformers import AutoTokenizer

PROMPT = "Il etait une fois"
N_GEN = 120

tok = AutoTokenizer.from_pretrained('camembert-base')
out_ids_base = tok.encode(PROMPT, return_tensors='pt')
prompt_len = out_ids_base.shape[1]

print("=" * 70)
print("32M — test avec 3 temperatures differentes")
print("=" * 70)

ckpt32 = torch.load('results_2b/ckpt_B_latest.pt', map_location='cpu', weights_only=False)
m32 = VariantB(32004)
m32.load_state_dict(ckpt32['model'], strict=False)
m32.float().eval()
del ckpt32

for temp in [0.6, 0.8, 1.0, 1.2]:
    out = out_ids_base.clone()
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_GEN):
            logits = m32(out)
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
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
    print(f"\ntemp={temp} [{elapsed:.0f}s] unique={unique}/{N_GEN} ({unique/N_GEN*100:.0f}%) repeat={repeat:.0f}%")
    print(f"  -> {text}")

del m32

print("\n" + "=" * 70)
print("100M — ref avec temp=0.8")
print("=" * 70)

ckpt100 = torch.load('results_fr100m_4l/ckpt_latest.pt', map_location='cpu', weights_only=False)
m100 = OscModel()
m100.load_state_dict(ckpt100['model'], strict=False)
m100.float().eval()
del ckpt100

out = out_ids_base.clone()
t0 = time.time()
with torch.no_grad():
    for i in range(N_GEN):
        logits = m100(out)
        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
        v, _ = torch.topk(probs, 50)
        probs[probs < v[:, -1:]] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        ntx = torch.multinomial(probs, 1).clamp(0, 32004)
        out = torch.cat([out, ntx], dim=1)
elapsed = time.time() - t0
text = tok.decode(out[0])
gen_ids = out[0].tolist()[prompt_len:]
unique = len(set(gen_ids))
bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
repeat = (1 - len(set(bigrams)) / max(len(bigrams), 1)) * 100
print(f"\ntemp=0.8 [{elapsed:.0f}s] unique={unique}/{N_GEN} ({unique/N_GEN*100:.0f}%) repeat={repeat:.0f}%")
print(f"  -> {text}")
