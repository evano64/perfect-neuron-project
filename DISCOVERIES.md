# Key Discoveries

## Architecture

1. **Oscillator layers outperform MLP at same parameter count**: 1.0 loss advantage, 3.8x better perplexity in controlled ablation
2. **Collapse threshold at ~50M params**: Below this, oscillator models degenerate into repetitive generation despite low loss
3. **Depth limitation**: Cannot stack beyond ~8 oscillator layers (16L model failed at loss 7.5)
4. **The model is not Kuramoto**: It's a complex linear recurrent unit, closer to LRU/Griffin than canonical Kuramoto dynamics

## Training

5. **Loss ≠ generation quality**: The "perplexity trap" — 32M model had loss 3.5 (appearing competitive) but generated catastrophic repetition
6. **Cosine LR > exponential**: Oscillator models benefit from high LR for longer (99% LR at step 10K vs 73% with exponential)
7. **Label smoothing helps**: 0.1 smoothing prevents overconfidence in oscillator-attention hybrids
8. **FR is 2x easier than EN**: CamemBERT tokenizer entropy ~5 bits vs ~10 bits for English BPE

## Methodology Bugs

9. **is_causal=False creates bidirectional shortcuts**: Loss drops to 1.66 but generation collapses (the historical bug)
10. **F.scaled_dot_product_attention ignores model.eval()**: Dropout fires during eval and generation, inflating all reported metrics
11. **Evaluation on training data**: Early evals drew from the same data as training, providing no generalization signal

## Data

12. **Fine-French 5B > Lucie 2B**: Higher quality filtered data for continued training
13. **Transfer learning works**: Lucie checkpoint immediately useful on Fine-French (loss 5.58 vs 8.4 from scratch)

## Generation

14. **Unique ratio > 70% is healthy**: Below this, repetition loops emerge
15. **Bigram repeat < 10% is healthy**: Above this, the model enters collapse mode
16. **Temperature doesn't rescue collapse**: 32M model failed at all temperatures (0.6-1.2)
17. **Prompt format matters**: `</s>` token in prompts worsens collapse but isn't the root cause
