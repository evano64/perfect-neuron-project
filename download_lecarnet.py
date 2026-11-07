"""Download LeCarnet dataset, tokenize with CamemBERT, save as numpy uint16"""
import os,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
os.chdir('C:/Users/User/.gemini/antigravity/scratch/perfect_neuron_project')

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

print('Loading CamemBERT tokenizer...')
tok=AutoTokenizer.from_pretrained('camembert-base')
print('Vocab size:', tok.vocab_size)

print('Downloading LeCarnet from HuggingFace...')
ds=load_dataset('MaxLSB/LeCarnet',split='train')
print('Dataset size:', len(ds))
print('Sample:', ds[0]['text'][:200])

print('Tokenizing...')
all_tokens=[]
for i,sample in enumerate(ds):
    text=sample['text']
    ids=tok.encode(text,add_special_tokens=False)
    all_tokens.extend(ids)
    # Add separator token between stories
    all_tokens.append(3)  # <s> token as separator
    if (i+1)%100000==0:
        print(f'  [{i+1}/{len(ds)}] tokens so far: {len(all_tokens):,}')

all_tokens=np.array(all_tokens,dtype=np.uint16)
print(f'Total tokens: {len(all_tokens):,}')
print(f'Token range: [{all_tokens.min()}, {all_tokens.max()}]')
print(f'Approx size: {all_tokens.nbytes/1e6:.1f} MB')

out_file='lecarnet_tokens.npy'
np.save(out_file,all_tokens)
print(f'Saved: {out_file}')
