# Audit Complet du Projet ONLM

*Février 2026 — Rapport d'audit*

---

## 1. Verdict global

Le projet est très intéressant techniquement et potentiellement publiable, mais pas pour les raisons initialement mises en avant.

### Ce qui ressort vraiment

- Le **32M ne semble pas réellement fonctionnel en génération**, même avec une loss basse.
- Le **100M génère du français nettement meilleur**, malgré une loss bien plus haute.
- Donc, dans ton architecture, **la validation loss seule n'est pas un bon proxy de qualité générative**.

Le vrai résultat empirique du projet est :
> **Les modèles oscillatoires semblent avoir un seuil de capacité minimale, en dessous duquel ils collapsent en répétitions ou en charabia, malgré une loss flatteuse.**

---

## 2. Diagnostic central

### 2.1 Le 32M est en collapse

Les générations du 32M montrent :
- répétitions massives (`tirant tirant tirant`, `abîmes abîmes`, `ffler ffler`)
- fragments de tokens / pseudo-mots (`Leseosing`, `lapotition`, `lhblen`, etc.)
- absence de régime de sampling vraiment sain :
  - temp 0.6 → répétition
  - temp 0.8 → charabia
  - temp 1.0 → bruit plus divers
  - temp 1.2 → chaos / `<unk>`

**Conclusion :** Le 32M n'a probablement jamais réellement appris à générer du bon français, même si sa loss descendait.

### 2.2 Le 100M est le premier modèle vraiment prometteur

Le 100M montre :
- phrases françaises grammaticalement plausibles
- bonne diversité lexicale
- beaucoup moins de répétition
- structure syntaxique globalement saine

Mais encore :
- sémantique floue
- quelques mots inventés / sous-mots mal recollés
- logique discursive incomplète

**Conclusion :** Le 100M est dans une zone saine d'apprentissage :
- il a déjà appris une bonne partie de la syntaxe
- il n'a pas encore totalement appris la cohérence sémantique longue
- c'est normal à ce stade

---

## 3. Découverte clé : le "piège de la perplexité"

### 3.1 Pourquoi la loss du 32M est trompeuse

La loss est calculée sur des sous-séquences aléatoires du corpus.

Donc un modèle peut :
- très bien prédire les tokens fréquents localement (`de`, `la`, `et`, `est`, etc.)
- faire baisser la cross-entropy
- **sans savoir dérouler une génération longue cohérente**

**En pratique :** Le 32M semble avoir appris :
- des régularités locales
- des transitions fréquentes à courte portée

Mais pas :
- une dynamique stable sur 50–120 tokens
- une mémoire séquentielle utile
- une cohérence globale

### 3.2 Conséquence importante

Pour ton architecture, **loss basse ≠ bon modèle génératif**.

Donc :
- il faut arrêter de sélectionner les checkpoints uniquement à la loss
- il faut introduire des métriques de génération automatiques

---

## 4. Interprétation architecturale

### 4.1 Le modèle n'est probablement pas "vraiment Kuramoto" au sens strict

Le point théorique important de l'audit :

La formulation que tu donnes ressemble davantage à :
- une récurrence complexe linéaire
- avec decay appris
- fréquence apprise
- puis couplage/mixage linéaire

qu'à un système de Kuramoto strict, où le couplage central est non-linéaire en phase, typiquement via `sin(θⱼ - θᵢ)`.

**Donc :** En l'état, ton modèle est probablement plus proche de :
- LRU
- récurrences complexes diagonales
- familles type S4D / Griffin / hybrides RNN-attention

que d'un Kuramoto canonique.

**Conséquence pour le paper :**
Tu as deux options :
1. **Repositionner** honnêtement le modèle comme un hybride "complex linear recurrent + attention"
2. Ou **rendre la dynamique réellement Kuramoto-like** avec un vrai couplage non-linéaire des phases

### 4.2 Hypothèse forte sur le collapse

Le collapse du 32M est compatible avec l'idée suivante :
- l'espace d'état oscillatoire du 32M est trop pauvre
- la dynamique tombe dans des attracteurs de faible dimension
- ces attracteurs apparaissent comme :
  - répétitions
  - cycles
  - charabia tokenizer
- Le 100M aurait suffisamment de capacité pour éviter cette synchronisation prématurée

**En résumé :** Les modèles ONLM semblent avoir un **seuil critique de capacité**.

---

## 5. Ce qu'il ne faut plus dire tel quel dans le papier

### 5.1 "Le 32M est compétitif"

Non, plus vraiment.

Si le 32M a une belle loss mais ne sait pas générer, il ne faut plus le vendre comme :
- compétitif
- baseline égale aux petits transformers
- preuve forte d'efficacité

Le 32M doit plutôt être présenté comme :
- **un exemple de collapse à petite échelle**

### 5.2 "Le bug causal est une contribution majeure"

Non.

Le fait d'avoir oublié `is_causal=True` est :
- un vrai bug
- important
- utile à documenter

Mais ce n'est pas une contribution scientifique principale.

Il faut le repositionner comme :
- note méthodologique
- lesson learned
- appendice

### 5.3 Comparer des losses FR/EN ou tokenizers différents

Pas valide.

Comparer :
- un loss FR sur Lucie + CamemBERT
- à un loss EN sur FineWeb / GPT-style tokenizer

n'est pas une comparaison solide.

Il faut :
- soit comparer sur même dataset, même tokenizer, même contexte
- soit utiliser une métrique plus robuste comme **bits-per-byte** / bits-per-character

---

## 6. Ce qu'il faut faire maintenant sur le 100M

### 6.1 Sauvegarder très souvent

Le 100M peut être dans son sweet spot génératif avant sa meilleure loss.

**À faire :**
- checkpoint tous les 10k ou 20k steps
- garder aussi les logs de génération

### 6.2 Ne pas sélectionner le meilleur modèle par loss seule

Il faut désormais choisir les checkpoints selon un compromis :
- loss correcte
- faible répétition
- diversité suffisante
- texte lisible

**Métriques à logger automatiquement :** Pour 5 prompts fixes, à chaque checkpoint :
- `unique_ratio`
- `bigram_repeat`
- `trigram_repeat`
- nombre de `<unk>`
- longueur avant boucle
- entropie moyenne des logits
- éventuellement probabilité max moyenne

### 6.3 Early stopping basé sur la génération

Si tu observes :
- remontée durable de la répétition
- baisse forte de la diversité
- apparition d'artefacts fréquents
- texte moins bon malgré loss plus basse

alors il faut considérer que **le modèle se dégrade générativement, même si la loss continue à baisser**.

---

## 7. Problèmes pratiques de génération déjà identifiés

### 7.1 Ne pas mettre `</s>` dans le prompt

Tu utilisais des prompts comme :
```
<s> Il etait une fois</s>
```

C'est une mauvaise idée. `</s>` signifie "fin de séquence", donc tu demandes au modèle de continuer après une fin.

**À faire :** Tester plutôt :
- `Il était une fois`
- `<s> Il était une fois`

Mais pas avec `</s>`.

### 7.2 Config d'inférence recommandée pour le 100M

Pour éviter les boucles :
- `temperature = 0.8`
- `top_p = 0.92`
- `top_k = 50`
- `repetition_penalty = 1.10 à 1.15`
- `no_repeat_ngram_size = 3`

Et bloquer les tokens spéciaux inutiles :
- `<unk>`
- `<pad>`
- `<mask>`

---

## 8. Ce qu'il faut mesurer en plus

### 8.1 Vraie validation NLL sans label smoothing

Tu entraînes avec `label_smoothing=0.1`.

Le problème : la loss smoothed n'est pas une vraie perplexité standard, elle est plus difficile à interpréter.

**À faire :** Logger 2 choses :
- `val_loss_smooth` = la loss d'entraînement standard
- `val_nll_hard` = cross-entropy normale sans smoothing

Puis : `ppl = exp(val_nll_hard)`

C'est cette mesure qu'il faut utiliser pour parler de perplexité.

### 8.2 Mesurer la santé générative automatiquement

Il te faut un script qui :
1. prend N prompts fixes
2. génère 120–200 tokens
3. calcule :
   - unique ratio
   - repeat bigram
   - repeat trigram
   - nombre de tokens spéciaux
   - longueur avant la première répétition lourde

C'est indispensable pour piloter ce projet.

### 8.3 Si possible : mesurer la synchronisation des oscillateurs

Très bonne expérience à faire si tu peux :

Calculer un ordre de synchronisation type :
```
R(t) = |1/N * Σₖ e^(jθₖ(t))|
```

**Hypothèse :**
- sur le 32M en boucle : R(t) montre très haut → sur-synchronisation
- sur le 100M sain : R(t) reste modéré → diversité dynamique

Si cette corrélation existe, tu as une figure scientifique très forte.

---

## 9. Recommandations d'architecture pour éviter le collapse

### 9.1 Ajouter un vrai MLP / SwiGLU

Si ton bloc est trop centré sur oscillateur + attention sans vraie non-linéarité feedforward, il manque une brique essentielle.

**Recommandation :** Structure de bloc type :
```python
x = x + attn(rmsnorm1(x))
x = x + alpha_osc * osc(rmsnorm2(x))
x = x + mlp(rmsnorm3(x))
```

Avec un MLP de type :
- SwiGLU
- ou GELU

### 9.2 Gater la contribution de l'oscillateur

Ne laisse pas l'oscillateur dominer trop tôt.

**Recommandation :** Ajouter un paramètre appris :
```python
x = x + alpha_osc * osc(norm(x))
```
avec :
- `alpha_osc` initialisé très petit (0.01 ou 0.0)

Idée : le modèle commence proche d'un transformer, il apprend ensuite progressivement à utiliser l'oscillateur. C'est probablement une des modifications les plus importantes.

### 9.3 Réduire / apprendre le couplage

Si tu as un terme du style :
```python
H = H + 0.1 * K @ H
```
le `0.1` fixe peut être trop agressif.

**Recommandation :**
- faire un `coupling_scale` appris
- l'initialiser petit
- normaliser K

Exemples :
- `coupling_scale` init ≈ 0.005
- `K / sqrt(n_osc)` ou autre normalisation

### 9.4 Normaliser les états oscillatoires

Pour éviter qu'un petit nombre d'oscillateurs domine toute la dynamique :
- RMSNorm sur les features oscillatoires
- ou normalisation de magnitude de H

**But :**
- éviter la domination de certains modes
- limiter la dérive numérique
- limiter la synchronisation excessive

---

## 10. Recommandations de training pour éviter le collapse

### 10.1 Réduire ou supprimer le label smoothing

`label_smoothing=0.1` est probablement fort pour un LM causal.

**À tester :** 0.0 ou 0.02

Le smoothing peut aider, mais il peut aussi rendre la qualité générative plus difficile à lire.

### 10.2 Ajouter une loss anti-répétition

Très recommandé : unlikelihood training léger.

Idée : pénaliser le modèle quand il essaie de répéter encore des tokens/ngrammes récents.

Ça peut aider directement contre :
- `tirant tirant tirant`
- `abîmes abîmes`
- `ffler ffler`

### 10.3 Conserver le 100M comme axe principal

Le 32M a probablement surtout une valeur comme :
- cas pathologique
- preuve de collapse à petite échelle

**Le vrai avenir du projet est le 100M et au-dessus.**

---

## 11. Expériences prioritaires à faire

### Priorité absolue

**A. Baseline transformer pur**
Entraîner un transformer pur de taille similaire, sur :
- même dataset
- même tokenizer
- même contexte
- même compute

Sans ça, tu ne peux pas savoir si :
- le collapse du 32M est spécifique aux oscillateurs
- ou juste normal pour un petit modèle dans ce setup

**C'est l'expérience la plus importante.**

### Priorité haute

**B. Continuer le 100M**
Le continuer jusqu'à voir :
- si la génération continue de s'améliorer
- ou si lui aussi finit par collapsar quand la loss baisse

Question cruciale : est-ce que le 100M reste bon quand la loss descend plus bas ?

### Priorité haute

**C. Instrumenter la génération**
Mettre en place :
- prompts fixes
- métriques automatiques
- comparaison checkpoint par checkpoint

### Priorité moyenne

**D. Faire une version ONLM-v2**
Avec :
- residual gated oscillator
- MLP/SwiGLU
- coupling appris et plus petit
- moins ou pas de label smoothing
- éventuellement unlikelihood

### Priorité moyenne

**E. Modèle intermédiaire 50–70M**
Si possible, entraîner un modèle intermédiaire pour vérifier s'il existe un seuil de capacité.

---

## 12. Nouvelle histoire du papier

### Ce que le papier ne doit plus raconter

Pas : "notre 32M est compétitif en loss"

### Ce que le papier devrait raconter

Plutôt :
- Les ONLM montrent un régime de collapse à petite capacité
- La perplexité peut être trompeuse pour juger la qualité générative dans cette famille
- Un modèle plus grand (100M) montre une génération syntaxiquement cohérente bien avant d'avoir une loss très basse
- Il semble exister un seuil critique de capacité/dynamique
- Les métriques de génération doivent compléter la NLL

---

## 13. Forces du projet

Ce qui est vraiment solide :
- tu as construit et entraîné une architecture originale
- sur hardware AMD consumer, ce qui est impressionnant
- tu as documenté les échecs honnêtement
- tu as observé un vrai phénomène empirique non trivial
- tu ne t'es pas contenté de regarder la loss
- tu as des résultats 32M vs 100M qui racontent une vraie histoire

---

## 14. Faiblesses actuelles

- revendication "Kuramoto" probablement trop forte en l'état
- comparaisons extérieures non parfaitement valides
- absence de baseline iso-setup
- 32M initialement sur-interprété
- manque de métriques de génération automatisées
- validation loss difficile à lire avec label smoothing
- README / paper à repositionner

---

## 15. Plan d'action concret

### Maintenant
1. Continuer le 100M
2. Sauvegarder souvent
3. Générer à chaque checkpoint avec prompts fixes
4. Logger répétition/diversité
5. Ne pas choisir le meilleur checkpoint seulement par loss
6. Ne plus utiliser `</s>` dans les prompts

### Ensuite
7. Calculer une vraie `val_nll_hard`
8. Ajouter repetition penalty / top-p / top-k propres
9. Entraîner un baseline transformer pur
10. Réécrire le paper autour du collapse + perplexity trap + threshold effect

### Pour la v2
11. Gated oscillator residual
12. MLP/SwiGLU
13. Couplage appris plus petit
14. Moins de label smoothing
15. Unlikelihood loss légère

---

## 16. Résumé ultra-court

**En une phrase :**

Le 32M a une loss trompeuse et semble collapsé, alors que le 100M est le premier vrai modèle générativement sain ; le vrai résultat du projet est l'existence probable d'un seuil critique de capacité et d'un découplage entre perplexité et qualité de génération dans cette architecture.
