# Analyse approfondie de ton projet ONLM

*Claude Fable 5 — juillet 2026*

---

D'abord, bravo : entraîner un modèle de langue avec une architecture custom sur un GPU consumer (RX 7900 XT + ROCm !), documenter les échecs, et écrire un white paper structuré, c'est déjà bien au-dessus de la moyenne des projets perso. Mais tu m'as demandé une analyse en profondeur, donc je vais être franc, y compris sur les points qui fâchent.

---

## 🔴 Problème n°1 (le plus important) : ce n'est pas vraiment du Kuramoto

C'est le cœur de ma critique. En regardant tes maths :

```
H_t = a^t · Σ a^(-s) · u_s + a^t · H_0    avec a = σ(decay_logit) · e^(jω)
H_t ← H_t + 0.1 · K · H_t
```

Ce que tu décris est une **récurrence linéaire diagonale complexe** avec un mixage linéaire appliqué après coup. Le modèle de Kuramoto, lui, est défini par :

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

La signature de Kuramoto, c'est le **couplage non-linéaire des phases** (le sin). C'est précisément pour ça qu'il n'existe pas de solution analytique fermée au système de Kuramoto en général — le fait même que tu aies une "analytical closed-form recurrence" est la preuve que ta dynamique est linéaire, donc pas Kuramoto.

**Conséquence directe** : ton architecture est en réalité très proche du **LRU (Linear Recurrent Unit, Orvieto et al. 2023, "Resurrecting Recurrent Neural Networks for Long Sequences")** — récurrence diagonale complexe, decay appris via sigmoid, fréquence apprise ω, exactement ta paramétrisation σ(decay) · e^(jω). C'est aussi apparenté à S4D/S5 et à la famille Griffin/Hawk/RecurrentGemma de Google (hybrides récurrence linéaire + attention, comme toi).

Ton tableau de nouveauté (1.3) est donc incomplet sur le point le plus critique. Tu compares à AKOrN, coRNN, Jamba... mais le travail le plus proche du tien (LRU + hybrides Griffin/Zamba) n'y figure pas. Un reviewer le verrait en 30 secondes.

**Deux options honnêtes :**

1. **Repositionner** : "hybride récurrence linéaire complexe + attention" — et comparer sérieusement à LRU/Griffin. C'est moins sexy mais défendable.
2. **Rendre le modèle vraiment Kuramoto** : couplage non-linéaire en phase, itératif (comme AKOrN le fait), et mesurer le paramètre d'ordre de synchronisation R(t). Là tu aurais un vrai claim de nouveauté pour le LM.

---

## 🟠 Problème n°2 : le "bug causal" n'est pas une contribution

Je comprends que ça t'ait coûté cher en temps de debug, et c'est bien de le documenter. Mais présenter l'oubli de `is_causal=True` comme une "Critical Bug Discovery" et une des 3 contributions du papier affaiblit ta crédibilité :

- C'est le bug le plus classique de l'entraînement autorégressif, connu depuis GPT-1. La loss de 1.66 anormalement basse + génération dégénérée est le symptôme textbook du data leakage.
- L'affirmation "this bug has not been reported in prior oscillatory network literature" est trompeuse : ce n'est pas un phénomène spécifique aux hybrides oscillateurs, c'est un bug d'implémentation universel.

**Recommandation** : garde-le comme une note "lessons learned" ou dans un appendice, retire-le des contributions.

---

## 🟠 Problème n°3 : les comparaisons ne sont pas valides

C'est le gros souci méthodologique :

| Problème | Détail |
|----------|--------|
| **Datasets différents** | Loss 3.5 sur du français (Lucie, CamemBERT 32k vocab) vs loss 3.5 sur de l'anglais (FineWeb-Edu, autre tokenizer) : ces deux nombres ne sont pas comparables. La perplexité dépend du tokenizer, de la langue et du corpus. |
| **Label smoothing 0.1** | Ça relève artificiellement le plancher de cross-entropy. Si tes baselines n'en utilisent pas, la comparaison est encore biaisée (dans un sens ou l'autre). |
| **Pas d'ablation** | L'expérience indispensable qui manque : le même modèle, mêmes données, même tokenizer, sans le bloc oscillateur (remplacé par un MLP ou rien, à budget de paramètres égal). Sans ça, impossible de dire si les oscillateurs apportent quoi que ce soit. C'est LA question que tout lecteur se posera. |
| **Modèle "32M" à 19.7M params** | Incohérence de nommage qui sème le doute sur le reste des chiffres. |
| **100M à 0.18B tokens** | Loss 5.7 sur un modèle massivement sous-entraîné : on ne peut rien en conclure, ni positif ni négatif. |

**Fix minimal** : entraîne un baseline transformer pur sur Lucie avec le même tokenizer, même compute, même config. Ou rapporte en bits-per-byte pour neutraliser le tokenizer.

---

## 🟡 Points techniques à creuser

- **Stabilité numérique** : Σ a^(-s) · u_s avec |a| < 1 signifie que a^(-s) explose exponentiellement. En bfloat16, avec des chunks de 32 tokens, tu es probablement à la limite. C'est peut-être une cause cachée de l'échec du modèle 16 couches, plutôt qu'un problème "fondamental" des oscillateurs profonds.

- **Le couplage K** : appliqué comme H ← H + 0.1·K·H après la récurrence, c'est un simple mixage linéaire statique, pas une dynamique couplée pendant l'évolution. Le facteur 0.1 hardcodé mériterait d'être appris ou ablaté.

- **Contexte de 256 tokens** : c'est très court, et paradoxal — l'argument de vente des récurrences continues, c'est justement le long contexte (là où l'attention O(T²) souffre). À 256 tokens, l'attention est quasi gratuite et tes oscillateurs n'ont aucun terrain pour briller.

- **LR 3e-3 pour un 100M** : très agressif. Ton observation que le cosine aide est intéressante, mais confondue avec le fait que l'exponentielle décroissait peut-être juste trop vite.

---

## 🟡 Détails de forme

- **LaTeX cassé** dans le README GitHub (les erreurs "'_' allowed only in math mode" sont visibles sur ton screenshot) — corrige avant de partager, ça fait mauvaise impression immédiate.
- **Références douteuses** : "Stentor (Li et al. 2024)", "OLMo Hybrid (AI2 2025)"... vérifie que tout existe réellement et est cité correctement. Une référence introuvable = crédibilité détruite.
- **Licence CC BY-NC-ND** pour du code : les licences Creative Commons sont déconseillées pour le code (elles ne traitent ni brevets ni linking). Et surtout, ND interdit les dérivés : personne ne peut builder dessus, ni même fine-tuner ou forker. Si ton objectif est l'impact recherche, c'est contre-productif. Apache 2.0, ou au pire CC BY-NC si tu tiens au non-commercial.

---

## ✅ Ce qui est bien (sincèrement)

- **Honnêteté scientifique** : tu rapportes l'échec du 16L, le bug, les limitations. C'est rare et précieux.
- **Le tableau de related work en 1.3** est une excellente pratique (il faut juste le compléter, cf. point 1).
- **Infra** : ROCm + PyTorch sur RX 7900 XT, entraînement bfloat16, chunk processing — c'est du vrai travail d'ingénierie.
- **L'analyse de génération** (diversité de tokens, taux de répétition) au-delà de la seule perplexité.
- **Section 5.3 (Open Questions)** : la question "les fréquences apprises correspondent-elles à des unités linguistiques ?" est la piste la plus intéressante du projet.

---

## 🎯 Feuille de route que je te suggère

1. **Court terme** : ablation oscillateur vs. sans oscillateur, mêmes données/compute. C'est l'expérience qui décide de tout.
2. **Repositionner ou renforcer le claim Kuramoto** (option 1 ou 2 du problème n°1).
3. **Analyse spectrale** : plot l'histogramme des ω appris, calcule le paramètre d'ordre de synchronisation, regarde si ça corrèle avec des frontières de mots/phrases. Si oui, tu as un résultat vraiment nouveau, même à petite échelle.
4. **Long contexte** : teste à 2k-8k tokens, là où ta récurrence a un avantage théorique.
5. **Nettoyer** : LaTeX, références, nommage des modèles, licence.

---

## Verdict global

Projet ambitieux avec une vraie exécution technique, mais dont les deux claims principaux (nouveauté "Kuramoto" et découverte du bug causal) ne tiennent pas en l'état face à un examen sérieux — le premier parce que la dynamique est linéaire et recoupe LRU/Griffin, le second parce que c'est un bug standard. **La bonne nouvelle** : les expériences qui sauveraient le projet (ablation, analyse spectrale, long contexte) sont faisables sur ton hardware actuel. C'est un projet à **70% du chemin** — le dernier 30% (rigueur comparative) est ce qui fera la différence entre "projet perso sympa" et "contribution qu'on cite".
