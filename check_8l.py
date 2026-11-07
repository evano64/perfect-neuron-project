import torch, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Check 8L best.pt
ckpt = torch.load('results_depth_8l/best.pt', map_location='cpu', weights_only=False)
print('=== 8L best.pt ===')
step = ckpt.get('step', '?')
print('Step:', step)
print('Loss:', ckpt.get('loss', '?'))
sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
for k, v in sd.items():
    if 'decay' in k:
        print('  %s: min=%.6f max=%.6f mean=%.6f std=%.6f unique=%d' % (k, v.min(), v.max(), v.mean(), v.std(), len(torch.unique(v))))

# Check ckpt_latest
ckpt2 = torch.load('results_depth_8l/ckpt_latest.pt', map_location='cpu', weights_only=False)
print()
print('=== 8L ckpt_latest.pt ===')
step2 = ckpt2.get('step', '?')
print('Step:', step2)
print('Loss:', ckpt2.get('loss', '?'))
sd2 = ckpt2.get('model', ckpt2.get('state_dict', ckpt2))
for k, v in sd2.items():
    if 'decay' in k:
        print('  %s: min=%.6f max=%.6f mean=%.6f std=%.6f unique=%d' % (k, v.min(), v.max(), v.mean(), v.std(), len(torch.unique(v))))

# Also check model_def.py for is_causal
print()
print('=== model_def.py ===')
with open('model_def.py', 'r') as f:
    for i, line in enumerate(f, 1):
        if 'is_causal' in line or 'causal' in line.lower():
            print('  Line %d: %s' % (i, line.strip()))
        if 'decay_logit' in line:
            print('  Line %d: %s' % (i, line.strip()))
