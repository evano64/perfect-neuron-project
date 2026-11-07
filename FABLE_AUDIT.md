# AUDIT FABLE 5 — Projet ONLM
*5 juillet 2026 — audit complet du code, de la méthodologie et des claims*

---

## 0. Résumé exécutif

Le projet est dans un état bien meilleur qu'il y a une semaine : le 100M génère du français cohérent, les découvertes (seuil de capacité, piège de la perplexité) sont réelles et documentées. **MAIS** cet audit a trouvé **3 nouveaux bugs jamais détectés par les audits précédents**, dont un **critique** qui aurait invalidé l'expérience la plus importante du projet (le baseline), et un qui **fausse toutes les mesures d'éval et de génération faites jusqu'ici**.

| # | Bug | Sévérité | Impact |
|---|-----|----------|--------|
| B1 | `train_baseline.py` : attention **sans masque causal** | 🔴 CRITIQUE | Le baseline aurait triché en bidirectionnel — l'expérience de comparaison aurait été invalide |
| B2 | `dropout_p=0.1` **codé en dur** dans SDPA → actif même en eval/génération | 🔴 MAJEUR | Toutes les eval loss et TOUTES les générations du 100M ont été mesurées avec 10% de dropout d'attention actif |
| B3 | Eval sur les **données d'entraînement** (pas de held-out) | 🟠 MOYEN | "Eval loss" = train loss re-mesurée ; aucune mesure de généralisation |

---

## 1. BUG B1 (critique) : le baseline avait le même bug is_causal que tout le projet

Dans `train_baseline.py` (écrit hier) :

```python
self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
...
h, _ = self.attn(h, h, h)   # ← AUCUN attn_mask, AUCUN is_causal
```

`nn.MultiheadAttention` sans `attn_mask` est **bidirectionnel**. C'est *exactement* le bug qui a coûté des dizaines d'heures au projet (E12 de l'historique, découverte D2). L'ironie est maximale : l'expérience destinée à valider le fix du bug… contenait le bug.

**Conséquence si non détecté** : le baseline aurait affiché une loss artificiellement basse (~1.5-2.5), on aurait conclu "les transformers purs écrasent les oscillateurs", et le paper aurait été bâti sur une comparaison invalide.

**Bonus** : le script avait aussi un top-k cassé dans la génération (`probs[probs[v[:,-1:]<1e-6]]=0` — indexation invalide, crash garanti), et tournait à 0.5 st/s (contention GPU + `nn.MultiheadAttention` sans fast path).

**Fix appliqué** : réécriture du bloc attention avec `F.scaled_dot_product_attention(..., is_causal=True)` — le même chemin de code que le modèle oscillateur, ce qui rend l'ablation propre : **seul le bloc oscillateur diffère**.

> Note de process : le baseline lancé hier est mort au bout de 100 steps (tué par un timeout de shell) avant d'avoir gaspillé du compute. Coup de chance. Il doit être relancé en processus détaché.

---

## 2. BUG B2 (majeur) : dropout d'attention actif en eval et en génération

Dans `train_fr_100m_4l.py` **et** `model_def.py` :

```python
a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)
```

`F.scaled_dot_product_attention` est une **fonction**, pas un module : elle ignore `model.eval()`. Le `dropout_p=0.1` codé en dur est donc appliqué **partout, tout le temps** :

- pendant toutes les **eval** (les loss 5.44-5.52 rapportées incluent du bruit de dropout) ;
- pendant toutes les **générations de test CPU** du 100M (`model_def.py` a le même code).

**Conséquences concrètes :**

1. La vraie eval loss du 100M est **plus basse** que les 5.44 rapportés.
2. La qualité de génération du 100M ("90% unique, 0% repeat") a été mesurée **avec 10% des liens d'attention aléatoirement coupés**. Le modèle est probablement *meilleur* que ce qu'on a vu.
3. La comparaison 32M vs 100M était **biaisée contre le 100M** : le 32M (`variants.py`, VariantB) utilise une attention manuelle *sans* dropout. Le gap réel est donc encore plus grand — la conclusion tient, mais le chiffre est faux.

**Fix appliqué** : `dropout_p=0.1 if self.training else 0.0` dans les deux fichiers.

⚠️ Le process d'entraînement en cours (PID 11688) a chargé l'ancien code : le fix prendra effet **au prochain restart** (les coupures de courant s'en chargeront probablement). Les checkpoints ne sont pas affectés — c'est un bug d'exécution, pas de poids.

---

## 3. BUG B3 (moyen) : pas de held-out — l'eval mesure le train

L'eval tire des fenêtres aléatoires de `lucie_fr_2b.npy`… le même tableau que le training :

```python
ei = [random.randint(0, nt-BLOCK-1) for _ in range(16)]   # même mm que le train
```

À ~1 epoch la mémorisation est limitée, donc le chiffre n'est pas absurde, mais :
- "best eval 5.3670" n'est pas une mesure de généralisation ;
- impossible de détecter un début d'overfitting sur la fin du cosine (LR → 1e-5, 500K steps ≈ 2 epochs).

**Fix appliqué** : les **10 derniers M de tokens** du dataset sont réservés à l'eval (train sur `[0, nt-10M]`, eval sur la fin). Attention : au prochain restart, l'eval loss va **sauter** (nouvelle distribution) — ne pas paniquer, c'est attendu, et `best_eval` repartira sur une base saine.

---

## 4. Problèmes de mesure secondaires (fixés aussi)

| Problème | Détail | Fix |
|----------|--------|-----|
| **PPL gonflée par le label smoothing** | `ppl=exp(eval_loss_smoothed)` → les "ppl 230" sont surestimées. La vraie NLL est plus basse. | L'eval logge désormais `smooth` ET `nll_hard` + `ppl=exp(nll_hard)` |
| **Compteur `ts` reset à chaque restart** | Le log affiche 0.89 Btok alors que le vrai total est ~2.2B | `ts` sauvegardé/restauré dans les checkpoints |
| **Gen crash "Float vs BFloat16"** | Hors autocast, `x.float()` (float32) rencontre `B_re`/`C` en bf16 dans les einsum/matmul complexes | Cast `.float()` explicite sur `B_re`, `B_im`, `C` dans `KuramotoOsc.forward` — la génération in-training marchera enfin |
| **Log de génération hors boucle** | Seul le dernier prompt était loggé (`log` dédenté) | Log par prompt + `unique_ratio` et `bigram_repeat` par prompt |

---

## 5. Architecture — confirmations et un point neuf

Les audits précédents (ANALYSE_CLAUDE.md, AUDIT_COMPLET.md) restent valides : ce n'est pas du Kuramoto (récurrence linéaire diagonale complexe type LRU), le couplage `0.1·K·H` est un mixage linéaire statique, il manque un MLP, contexte 256 trop court pour que la récurrence brille. Je n'y reviens pas.

**Point neuf — incohérence du couplage K aux frontières de chunks** : dans le forward, l'ordre est

```python
Hc = a_pow * cumsum(...)          # récurrence
Hc = Hc + 0.1 * K @ Hc            # couplage appliqué AVANT l'ajout du carry
Hc = Hc + carry_decay * h_carry   # le carry n'est PAS couplé
```

Le terme K est appliqué à la partie intra-chunk de l'état mais **pas** à la contribution portée depuis les chunks précédents. La dynamique effective change donc tous les 32 tokens. Ce n'est pas fatal (K est petit), mais :
1. c'est une preuve supplémentaire que K n'est pas une dynamique couplée mais un post-traitement ;
2. pour la v2, si on garde un couplage, il faut l'appliquer *après* l'ajout du carry, ou mieux, l'intégrer dans la récurrence.

---

## 6. État du training 100M (au moment de l'audit)

- Step 216 800 / 500 000, train loss ~5.51, eval (contaminée + dropout) oscillant 5.44-5.50, best 5.367 @ 171.5K
- LR 0.00182 (61% du max) — encore très haut, la vraie descente viendra dans la 2e moitié du cosine
- ~2.2B tokens réels vus (~21 tok/param — GPT-2 small en avait ~75-85)
- Vitesse 4.6-4.7 st/s (léger ralentissement dû à la cohabitation avec le baseline hier : 4.9 → 4.6)

**Lecture honnête** : le plateau apparent 5.44-5.52 depuis 40K steps n'est pas alarmant à ce niveau de LR, mais l'absence de MLP dans le bloc reste le suspect n°1 si le modèle plafonne au-dessus de ~4.5 en fin de cosine.

---

## 7. Le paper (WHITE_PAPER.md) — toujours pas conforme aux audits

Le draft actuel contient **encore** tout ce que les deux audits précédents ont demandé de retirer :

- l'abstract vend toujours "competitive perplexity results" et "matching Stentor-30M" (§Abstract, §3.3.1) — comparaison FR/EN invalide ET le 32M est en collapse ;
- le bug causal est toujours contribution n°2 (§1.4) ;
- le tableau 3.3.1 compare des loss inter-langues/inter-tokenizers ;
- la vraie histoire (collapse / perplexity trap / seuil de capacité) n'y figure pas.

**Le paper est le livrable le plus en retard du projet.** Rien de ce qui précède ne peut être publié avant réécriture — et maintenant il faudra aussi re-mesurer les chiffres de génération après le fix B2 (ils seront probablement *meilleurs*).

---

## 8. Plan d'action issu de cet audit

### Fait pendant l'audit
1. ✅ `train_baseline.py` réécrit : SDPA causal identique au modèle osc, top-k réparé, held-out, NLL hard, `ts` persistant
2. ✅ `train_fr_100m_4l.py` corrigé : dropout eval, gen bf16, held-out, NLL hard, `ts` persistant (effectif au prochain restart)
3. ✅ `model_def.py` corrigé : dropout eval + casts float pour la génération CPU

### À faire ensuite (ordre de priorité)
4. Relancer le baseline (détaché) et le laisser atteindre 50K puis 100K steps → comparaison **à tokens égaux** avec les checkpoints osc step_50000/step_100000
5. Re-mesurer la génération du 100M avec le `model_def.py` corrigé (sans dropout parasite) — mettre à jour les chiffres 90%/0%
6. Au prochain restart du 100M : vérifier le saut d'eval (held-out) et noter la vraie `nll_hard`
7. Réécrire le paper autour de : collapse à petite capacité + perplexity trap + seuil + ablation baseline (quand les chiffres seront là)
8. v2 : gated oscillator (`alpha_osc` init ~0), MLP/SwiGLU, coupling appris intégré à la récurrence (cf. §5)

---

## 9. Verdict

Le projet continue de produire des résultats empiriques intéressants, et surtout il s'améliore méthodologiquement — mais cet audit prouve que **le réflexe "vérifier la causalité et le mode eval" doit devenir systématique** : le bug le plus documenté de l'histoire du projet a été réintroduit dans le script censé le contrôler, et un dropout fantôme a contaminé toutes les mesures publiables. Les deux sont maintenant fixés, et le baseline — l'expérience qui décide de tout — peut enfin tourner sur des bases valides.

*— Fable 5*
