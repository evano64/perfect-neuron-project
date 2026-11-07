# Oscillatory Neural Language Model (ONLM)

Hybrid architecture combining complex-valued recurrent oscillator layers with causal self-attention for autoregressive language modeling on French text.

## Key Results

- **1.0 loss advantage** over matched transformer baseline at equivalent token counts
- **1.8x better perplexity** (213 vs 378) at 100M parameters
- **Collapse threshold** discovered below ~50M parameters (the perplexity trap)

## Architecture

```
Input tokens → Embedding + Position
  → [OscillatorBlock → Causal Attention → LayerNorm] × 4
  → Final LayerNorm → Linear Head (weight-tied)
```

- 103M params (23% oscillatory)
- Complex-valued recurrence with learned frequencies
- Antisymmetric coupling matrix
- Chunk processing (32 tokens)

## Files

| File | Description |
|------|-------------|
| `model_def.py` | Model definition (OscModel) |
| `variants.py` | 32M oscillator variants (A/B/C) |
| `train_fr_100m_4l.py` | 100M oscillator training script |
| `train_baseline.py` | Baseline transformer (ablation) |
| `compare_gen_full.py` | Generation comparison (32M vs 100M) |
| `test_32m_prompts.py` | 32M prompt format tests |
| `WHITE_PAPER.md` | Research paper (draft) |
| `FABLE_AUDIT.md` | Code audit (July 2026) |

## Dataset

- **Lucie FR**: 2B tokens, CamemBERT tokenizer (32K vocab)
- Context: 256 tokens, batch size 32

## Training

```bash
python train_fr_100m_4l.py    # Oscillator model
python train_baseline.py      # Baseline transformer
```

Requires: PyTorch + ROCm (AMD GPU), transformers, numpy

## Results

| Model | Params | Tokens | Val Loss | Perplexity |
|-------|--------|--------|----------|------------|
| **ONLM 100M** | 103M | 1.3B | **5.62** | **~213** |
| Baseline | 101M | 1.5B | 6.63 | 378 |

## License

MIT
