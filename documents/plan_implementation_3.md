# Plan d'implémentation 3 — sous-structures (H&R) + file MoCo + tête de projection

Objet : remplacer l'objectif actuel (positifs jitter/masquage sur la protéine entière,
hard-neg, pas de tête de projection) par la combinaison **MoCo-v2 adaptée aux protéines** :
positifs = **sous-structures aléatoires distinctes** (Hermosilla & Ropinski, 2205.15675),
négatifs = **file MoCo** (1911.05722), + **tête de projection** (dimcol, 2110.09348).
Cadrage : non supervisé, sans SupCon ni TM-aux.

## Pourquoi ces trois choix (rappel post-mortem run 1)
- **Sous-structures** → corrige H1 : nos positifs étaient triviaux (deux copies quasi
  identiques → loss ~1e-5, invariance au bruit inutile). Deux régions distinctes du même
  fold = tâche non triviale, impossible à résoudre par invariance au bruit.
- **File MoCo** → corrige H2/H5 : petit batch = peu de négatifs ; la file découple le
  nombre de négatifs de la taille du batch, sans hard-neg (qu'on **retire** car il créait
  des faux négatifs et du collapse).
- **Tête de projection** → corrige H3 : prévient le collapse dimensionnel ; l'embedding
  d'évaluation est pris **avant** la tête (la tête est jetée).

---

## Composant 1 — Échantillonnage de sous-structures (Hermosilla & Ropinski)

### Spécification
- Pour chaque protéine, tirer **deux** sous-structures **distinctes** et **connexes**
  → la paire positive. Négatifs = autres protéines (via la file MoCo).
- Mode par défaut : **segment contigu** de la chaîne (préserve la localité de séquence,
  fidèle à H&R « preserves the local information of protein sequences »). Option :
  **boule spatiale** (M plus proches voisins 3D d'un résidu centre) — à comparer en ablation.
- **Taille = hyperparamètre critique** (sweet spot InfoMin) : fraction `f` des résidus
  (départ : `f ∈ [0.5, 0.8]`, tirée aléatoirement par vue). Trop petit → les deux vues ne
  partagent plus le fold (faux positifs) ; trop grand → vues quasi identiques (retour au
  problème). Garde-fou : taille min ≥ 17 résidus (contrainte kNN k=16) ; rejeter/clip sinon.

### Implémentation
La sous-structure est un **re-lifting CA-only restreint au sous-ensemble S de résidus**,
réutilisant les formules géométriques du lifter. Tous les features géométriques dérivent
des `ca_coords`, donc recalculables sur S :
- **rank0** : sous-ensemble des features par résidu (3Di, dièdres, aa, pe) → `feat[S]`.
- **rank1** : recomputer le graphe kNN sur `ca_coords[S]`, `k = min(16, |S|-1)` ;
  `vector = c_j - c_i`, `distance`, encodage (RBF in-model). (le modèle équivariant lit
  `rank1['vector']` → indispensable.)
- **rank2** : restreindre `sse_map_0` à S, remapper les ids de SSE en contigu, retirer les
  SSE sans membre dans S, **recomputer** les 12 features (PCA : valeurs propres + 5
  descripteurs de forme) sur les CA membres dans S. (DSSP n'est pas rejoué — on garde
  l'assignation SSE par résidu de la protéine entière, puis on la sous-ensemble.)
- **rank3** : recomputer taille `|S|`, rayon de giration, PCA globale sur `ca_coords[S]`.

Fichiers :
- **Refactor** : extraire de `topotein_lifter.py` les fonctions géométriques
  (`lift_rank1_edges`, PCA rang-2/3) en helpers réutilisables sur un sous-ensemble.
- **`contrastive_engine.py`** : remplacer `StructuralAugmentations.__call__` (jitter+crop+
  mask) par `sample_substructure(features, mode, f)` appelé **deux fois** pour produire les
  deux vues. Garder éventuellement un masquage de features léger ; **supprimer** le jitter
  de coordonnées comme augmentation principale.
- Le calcul se fait dans le DataLoader (workers), comme le jitter actuel.

### Garde-fous / tests
- Mesurer le **recouvrement** (Jaccard des résidus) et le **TM-score inter-vues** sur un
  échantillon : viser un recouvrement modéré (ni 1, ni 0) et un TM-vue élevé mais < 1.
- Vérifier que les deux vues sont bien **distinctes** (pas le même S).
- Vérifier la validité PCC (k ajusté, SSE non vides).

---

## Composant 2 — File de négatifs MoCo

### Spécification
- Deux encodeurs : **online** `f_q` (entraîné) et **momentum** `f_k` (EMA, sans grad) :
  `θ_k ← m·θ_k + (1-m)·θ_q`, `m ∈ [0.99, 0.999]`.
- Pour une protéine : vue1 → `q = g(f_q(vue1))` ; vue2 → `k⁺ = g_k(f_k(vue2))` (no-grad).
- **File FIFO** de taille `K` (départ : 4096–16384) de clés négatives des batches passés.
- Perte InfoNCE MoCo :
  `L = -log [ exp(q·k⁺/τ) / ( exp(q·k⁺/τ) + Σ_{k⁻∈file} exp(q·k⁻/τ) ) ]`.
- Enfiler `k⁺` du batch, défiler les plus anciennes, à chaque step.
- Pas de **ShuffleBN** nécessaire (le modèle utilise LayerNorm/GVPNorm, pas de BatchNorm).

### Implémentation
- Nouveau module `MoCo` (wrapper) : détient `f_q`, `f_k` (copie EMA), la tête de projection
  (online + momentum), et le buffer `queue` (K × dim) + `queue_ptr`.
- `train_contrastive.py` : remplacer le calcul `NTXentLoss(z[:B], z[B:])` par le step MoCo
  (forward q sur vue1 via f_q ; forward k⁺ sur vue2 via f_k en `no_grad` ; perte ; EMA ;
  enqueue/dequeue). La collate doit **garder vue1 et vue2 séparées** (ne plus concaténer).
- Retirer le `HardNegativeBatchSampler` ; garder un **batching budgété par résidus** mais
  **aléatoire** (pour la mémoire MPS), sans groupement longueur/SSE.

### Mémoire (16 GB MPS — point de vigilance)
- `f_k` = 2ᵉ jeu de poids ; il tourne en `no_grad` (pas d'activations conservées) → surcoût
  surtout = poids + une passe avant transitoire. La file = K×dim floats (16384×128×4 ≈ 8 Mo,
  négligeable).
- Le modèle équivariant est lourd : prévoir de réduire `vector_dim` (ex. 16→8), `K`, ou la
  profondeur si OOM. Benchmarker le throughput avant le sweep complet.

---

## Composant 3 — Tête de projection

### Spécification
- L'encodeur renvoie une **représentation** `h` (pooling de lecture, **non normalisée**).
- Tête `g` = MLP 2 couches (`dim → dim → dim`, SiLU), suivie d'une **L2-normalisation** ;
  la perte contrastive opère sur `g(h)`.
- **L'embedding d'évaluation** (TM-rho, clustering, retrieval) = `normalize(h)` — la tête
  `g` est **jetée** à l'inférence. C'est le point clé qui prévient le collapse dimensionnel.

### Implémentation
- Séparer dans le modèle (`asymmetric_/equivariant_topotein.py`) la représentation `h` de
  la sortie normalisée actuelle : exposer `forward(..., return_repr=True)` qui renvoie `h`
  pré-normalisation.
- Module `ProjectionHead(dim)` ; côté MoCo, une tête online (entraînée) et une tête momentum
  (EMA), comme l'encodeur.
- Eval : utiliser `normalize(h)`.

---

## Changements connexes (post-mortem)
- **Retirer** hard-neg mining, SupCon, TM-aux (déjà acté).
- **Sélection de modèle** : meilleur epoch sur une métrique cible (TM-recall / TM-rho), pas
  la val loss (qui s'améliore pendant que la cible se dégradait en run 1).
- **Calibrer le plafond** : mesurer le TM-rho de Foldseek / du 3Di brut sur la base, pour
  savoir ce qu'« apprendre » veut dire ici.

## Hyperparamètres (valeurs de départ)
| Param | Départ | Note |
|---|---|---|
| taille sous-structure `f` | 0.5–0.8 (aléatoire/vue) | sweet spot à balayer |
| mode | segment contigu | option : boule spatiale |
| file `K` | 8192 | ↓ si OOM |
| momentum `m` | 0.99 | 0.999 si file grande |
| température `τ` | 0.2 | (0.1 a favorisé le collapse en run 1) |
| tête de projection | MLP dim→dim→dim, SiLU | jetée à l'éval |
| `vector_dim` (équiv.) | 8 | ↓ depuis 16 pour la mémoire |
| optim | AdamW, lr 1e-4 | + warmup conseillé |

## Tests / smoke tests (incrémental, un composant à la fois)
1. **Sous-structures** : self-test — deux vues distinctes, PCC valide, équivariance tient
   encore (rejouer le self-test de `equivariant_topotein.py` sur une sous-structure) ;
   distribution recouvrement / TM inter-vues raisonnable.
2. **Tête de projection** : l'éval utilise bien `h` (pré-tête) ; la tête ne fuit pas dans `h`.
3. **File MoCo** : la file se remplit ; `f_k` suit `f_q` (EMA) ; enqueue/dequeue corrects ;
   pas de doublon q/k de la même protéine dans les négatifs.
4. **Intégration** : un mini-run — la **loss par batch ne s'effondre PAS à ~1e-5** (preuve
   que la tâche est non triviale), santé collapse OK (rang effectif, mean cos).

## Portes de décision (qu'est-ce qui valide vs falsifie)
- **Porte A (tâche non triviale)** : loss par batch loin de 0 ; sinon la taille de
  sous-structure est trop grande (vues trop semblables) → réduire `f`.
- **Porte B (apprentissage réel)** : TM-rho / TM-recall **s'améliorent avec l'entraînement**
  (et non dégradent depuis l'init comme en run 1). C'est le test central.
- **Porte C (au-dessus des features)** : bat la baseline features-only / init aléatoire.
- **Porte D (santé)** : pas de collapse (rang effectif stable, mean cos bas).

## Protocole ordonné
1. Refactor helpers géométriques du lifter → réutilisables sur sous-ensemble.
2. Implémenter `sample_substructure` + remplacer l'augmentation ; smoke test 1.
3. Ajouter la tête de projection + séparer `h` / sortie ; smoke test 2.
4. Implémenter le wrapper MoCo (f_k EMA + file) + adapter la boucle/collate ; smoke test 3.
5. Mini-run d'intégration ; smoke test 4 (porte A + santé).
6. Sweep : taille `f`, `τ`, `K`, contigu vs spatial ; appliquer les portes B/C/D.

## Risques & repli
- **Collapse persistant** malgré tête + MoCo → repli sur un objectif **sans négatifs**
  (VICReg / Barlow Twins), qui empêche le collapse par construction.
- **Mémoire** (2 encodeurs équivariants sur 16 GB) → réduire `vector_dim`, `K`, profondeur ;
  ou momentum encoder sur CPU si nécessaire.
- **Taille de sous-structure** mal réglée = le nouveau point de défaillance principal →
  prioriser son balayage et la mesure du TM inter-vues.
- **Sous-structures spatiales** peuvent fragmenter les SSE → préférer le segment contigu
  par défaut.
