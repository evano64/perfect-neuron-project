import torch, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check best.pt
ckpt = torch.load('results_fr100m_4l/best.pt', map_location='cpu', weights_only=False)
print('=== best.pt ===')
step = ckpt.get('step', '?')
print('Step:', step)
print('Keys:', list(ckpt.keys())[:10])
sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
for k, v in sd.items():
    if 'decay' in k:
        print('  %s: min=%.6f max=%.6f mean=%.6f unique=%d' % (k, v.min(), v.max(), v.mean(), len(torch.unique(v))))

# Check ckpt_latest
ckpt2 = torch.load('results_fr100m_4l/ckpt_latest.pt', map_location='cpu', weights_only=False)
print()
print('=== ckpt_latest.pt ===')
step2 = ckpt2.get('step', '?')
print('Step:', step2)
sd2 = ckpt2.get('model', ckpt2.get('state_dict', ckpt2))
for k, v in sd2.items():
    if 'decay' in k:
        print('  %s: min=%.6f max=%.6f mean=%.6f unique=%d' % (k, v.min(), v.max(), v.mean(), len(torch.unique(v))))

# Check step_200000
ckpt3 = torch.load('results_fr100m_4l/step_200000.pt', map_location='cpu', weights_only=False)
print()
print('=== step_200000.pt ===')
step3 = ckpt3.get('step', '?')
print('Step:', step3)
sd3 = ckpt3.get('model', ckpt3.get('state_dict', ckpt3))
for k, v in sd3.items():
    if 'decay' in k:
        print('  %s: min=%.6f max=%.6f mean=%.6f unique=%d' % (k, v.min(), v.max(), v.mean(), len(torch.unique(v))))

# Also check what model_def.py currently has
print()
print('=== model_def.py current init ===')
import ast
with open('model_def.py', 'r') as f:
    content = f.read()
if '0.5' in content and 'decay_logit' in content:
    print('model_def.py: decay_logit init = 0.5 (learnable)')
elif '3.0' in content and 'decay_logit' in content:
    print('model_def.py: decay_logit init = 3.0 (frozen)')
else:
    print('model_def.py: could not determine decay_logit init')
