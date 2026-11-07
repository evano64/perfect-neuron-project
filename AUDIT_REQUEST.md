# Audit Request: Oscillatory Neuron LLM Project

## Project Summary
We are building an oscillatory neuron-based LLM to beat SmolLM2-135M on benchmarks. The project has undergone 16 major discoveries and 12 failures. Key findings include a collapse threshold effect, perplexity trap, and architecture limitations.

## Key Files for Review

### 1. Model Architecture (`model_def.py`)
- **OscModel**: 103M params, d=1152, 4 layers, 16 heads, 1536 oscillators/layer (22.9% oscillatory)
- **KuramotoOsc**: Complex-valued oscillators with phase coupling, decay, and frequency
- **Key fix**: `is_causal=True` in attention (previously `False` causing collapse)
- **Label smoothing**: 0.1, dropout: 0.1

### 2. Baseline Architecture (`train_baseline.py`)
- **BaselineTransformer**: 100.9M params, same d=1152, 4 layers, 16 heads, SwiGLU MLP
- **Weight tying**: Yes (same as oscillator model)
- **Purpose**: Fair comparison with standard transformer

### 3. Training Configuration (`train_fr_100m_4l.py`)
- **Dataset**: Lucie FR 2B tokens (2,002,510,286 tokens, uint16)
- **Context**: 256 tokens
- **Batch size**: 32
- **LR**: Cosine schedule, max 3e-3, warmup 1000 steps
- **Total steps**: 500,000 (~3.4B tokens, 1.7 epochs)

### 4. Critical Findings

#### A. Collapse Threshold Effect
- **32M model** (19.7M params): Loss 3.5, but generation collapses (28% unique, 67% repeat)
- **100M model** (103M params): Loss 5.45, generates coherent French (90% unique, 0% repeat)
- **Conclusion**: Below ~50M params, oscillator models collapse regardless of loss

#### B. Perplexity Trap
- 32M loss 3.5 looks good but generates garbage
- 100M loss 5.45 generates coherent French
- **Loss ≠ generation quality** for oscillator models

#### C. Architecture Analysis
- Model is NOT truly Kuramoto (no phase coupling in generation)
- It's a complex linear recurrent unit (closer to LRU/Griffin/S4D)
- Decay is frozen at 0.95 (not learned)
- Phase coupling matrix K is anti-symmetric but not used in generation

### 5. Training Progress (100M FR 4L)
- **Current step**: ~208K
- **Current loss**: 5.4484
- **Best eval loss**: 5.3670 at step 171.5K
- **Tokens seen**: ~2.1B (0.7 epochs)
- **Speed**: 4.9 st/s, RAM 4.1 GB

### 6. Generation Quality Comparison
| Model | Params | Loss | Unique | Repeat | Quality |
|-------|--------|------|--------|--------|---------|
| 32M | 19.7M | 3.5 | 28% | 67% | Collapse |
| 100M | 103M | 5.45 | 90% | 0% | Coherent |

## Questions for Review

1. **Architecture**: Is the oscillator component providing value over standard MLP/SwiGLU?
2. **Collapse**: What is the minimum viable size for oscillator models?
3. **Perplexity trap**: How should we evaluate oscillator models (loss vs generation)?
4. **Scaling**: What architectural changes would improve scaling?
5. **Paper**: Should we rewrite around collapse/threshold findings instead of Kuramoto novelty?

## Files Available for Review
- `model_def.py`: Model definition
- `train_fr_100m_4l.py`: Oscillator training script
- `train_baseline.py`: Baseline transformer training script
- `variants.py`: 32M model variants
- `AUDIT_COMPLET.md`: Full project audit
- `ANALYSE_CLAUDE.md`: External analysis
- `WHITE_PAPER.md`: Draft white paper
- `HISTORIQUE_COMPLET.txt`: Complete project history
- `results_fr100m_4l/train.log`: Training log (208K steps)
- `results_baseline/train.log`: Baseline training log (starting)
