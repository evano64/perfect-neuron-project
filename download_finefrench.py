"""
Download Fine-French dataset, filter quality docs, tokenize, save as uint16 numpy.
Target: ~5B tokens from high-quality French web data.
Resumes from existing file if present.
"""
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import time, os

TARGET_TOKENS = 5_000_000_000
OUTPUT_FILE = 'finefrench_5b_tokens.npy'
MAX_LEN = 256

print('Loading CamemBERT tokenizer...')
tok = AutoTokenizer.from_pretrained('camembert-base')
print(f'Vocab size: {tok.vocab_size}')

# Check existing progress
existing_tokens = 0
if os.path.exists(OUTPUT_FILE):
    existing = np.load(OUTPUT_FILE, mmap_mode='r')
    existing_tokens = len(existing)
    print(f'Resuming: {existing_tokens:,} tokens ({existing_tokens/1e9:.2f}B) already saved')
    del existing
else:
    print('Starting fresh')

remaining = TARGET_TOKENS - existing_tokens
print(f'Need {remaining/1e9:.2f}B more tokens')

print('Streaming Fine-French dataset...')
ds = load_dataset('legmlai/finefrench-v1', split='train', streaming=True)
print('Dataset loaded (streaming mode)')

all_tokens = []
total_tokens = existing_tokens
total_docs = 0
skipped_docs = 0
t_start = time.time()

for i, example in enumerate(ds):
    if total_tokens >= TARGET_TOKENS:
        break

    if example.get('bad_prompt_detected', 1) == 1:
        skipped_docs += 1
        if skipped_docs % 100000 == 0:
            print(f'  Skipped {skipped_docs:,} low-quality docs')
        continue

    text = example.get('text', '').strip()
    if len(text) < 50:
        skipped_docs += 1
        continue

    try:
        tokens = tok.encode(text, add_special_tokens=True)
        if len(tokens) > MAX_LEN + 1:
            for start in range(0, len(tokens) - MAX_LEN, MAX_LEN):
                chunk = tokens[start:start + MAX_LEN + 1]
                if len(chunk) == MAX_LEN + 1:
                    all_tokens.extend(chunk)
                    total_tokens += len(chunk)
        elif len(tokens) >= 2:
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
    except Exception:
        skipped_docs += 1
        continue

    total_docs += 1

    if total_docs % 10000 == 0:
        elapsed = time.time() - t_start
        speed = (total_tokens - existing_tokens) / elapsed
        pct = total_tokens / TARGET_TOKENS * 100
        print(f'  Docs: {total_docs:,} | Tokens: {total_tokens/1e9:.2f}B ({pct:.1f}%) | Skipped: {skipped_docs:,} | Speed: {speed/1e6:.1f}M tok/s')

    if len(all_tokens) >= 600_000_000:
        arr = np.array(all_tokens, dtype=np.uint16)
        if existing_tokens == 0 and total_tokens <= TARGET_TOKENS:
            np.save(OUTPUT_FILE, arr)
        else:
            existing = np.load(OUTPUT_FILE, mmap_mode='r').copy()
            np.save(OUTPUT_FILE, np.concatenate([existing, arr]))
            del existing
        all_tokens = []
        print(f'  Saved checkpoint: {total_tokens/1e9:.2f}B tokens')

print(f'\n=== DONE ===')
print(f'Total docs: {total_docs:,}')
print(f'Total tokens: {total_tokens/1e9:.2f}B')
print(f'Skipped docs: {skipped_docs:,}')

if all_tokens:
    arr = np.array(all_tokens, dtype=np.uint16)
    if existing_tokens == 0:
        np.save(OUTPUT_FILE, arr)
    else:
        existing = np.load(OUTPUT_FILE, mmap_mode='r').copy()
        np.save(OUTPUT_FILE, np.concatenate([existing, arr]))
        del existing

final = np.load(OUTPUT_FILE, mmap_mode='r')
if len(final) > TARGET_TOKENS:
    final = final[:TARGET_TOKENS]
    np.save(OUTPUT_FILE, final)

print(f'Saved {OUTPUT_FILE}: {len(final):,} tokens, dtype={final.dtype}, range=[{final.min()},{final.max()}]')
print(f'File size: {final.nbytes / 1e9:.2f} GB')
