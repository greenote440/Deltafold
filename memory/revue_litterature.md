# Revue de littérature — consolidation du plan d'expérimentation

Objet : critiquer le plan d'expérimentation actuel (cf. `protocol.tex` et
`plan_implementation.md`) sous tous les angles, et fournir une bibliographie récente
pour alimenter la lecture. Chaque axe liste des **questions ouvertes** (critiques) et
des **lectures** renvoyant à la bibliographie catégorisée en fin de fichier.

Tags de catégorie : `[REPREND]` méthodo proche de la nôtre · `[CRITIQUE]` pointe une
faiblesse de l'approche · `[ALTERNATIVE]` propose autre chose · `[SIMILAIRE]`
problème de clustering/structure proche · `[FONDATION]` socle théorique.

## Décision de cadrage (actée)
Le modèle cible est **non supervisé pur** : InfoNCE sur des positifs d'augmentation
uniquement, **sans SupCon** (pas de positifs « même cluster Foldseek ») et **sans
terme auxiliaire TM-score**. Justification : éviter toute fuite côté entraînement
(le SupCon et, plus subtilement, le TM-aux injectent une similarité structurale
corrélée aux étiquettes d'évaluation). Conséquence majeure pour la critique :
l'augmentation devient **la seule source de signal** (axe A critique) et le risque de
collapse augmente (axe B critique), puisqu'on retire les deux termes qui
soutenaient le signal. La circularité **côté évaluation** (on évalue contre Foldseek)
demeure et doit être traitée séparément (axe D).

---

## Axe A — Paires positives & augmentation (jitter + masquage)

Questions ouvertes :
- À 0,3–0,5 Å, le jitter est-il dans le « sweet spot » de l'InfoMin ? Trop faible →
  les deux vues sont quasi identiques, la tâche est triviale et invite au collapse ;
  trop fort → on quitte la variété physique. Où est l'optimum, et comment le régler
  autrement qu'au pifomètre ? [vues] [denoise] [rigid]
- Le couple (jitter + masquage) définit les invariances apprises. Sont-elles les
  bonnes pour la similarité de *fold* ? On apprend l'invariance à un bruit synthétique,
  pas à la vraie variation entre homologues (conformations, indels). [vues] [reprend1]
- Un bruit non contraint sur les Cα casse-t-il la géométrie locale (longueurs/angles
  de liaison, clashs) ? Faut-il des perturbations contraintes / rigidity-aware, ou un
  objectif de débruitage (interprétable comme champ de forces pour petit σ) ? [rigid] [denoise] [geossl]
- Sans TM-aux ni SupCon, mesurer empiriquement le TM-score entre les deux vues : s'il
  reste > 0,9, la crainte « non physique » est faible ; sinon l'augmentation fabrique
  de faux positifs. (déjà acté comme expérience dans `plan_implementation.md` Mod 2.)

Lectures : `[reprend1]` GearNet (crops + masquage d'arêtes, sans bruit de coords),
`[reprend2]` Hermosilla & Ropinski (CL 3D avec bruit de coordonnées), `[vues]` What
Makes for Good Views (InfoMin), `[denoise]` Pre-training via Denoising, `[rigid]`
RigidSSL, `[geossl]` SE(3)-Invariant Denoising Distance Matching.

---

## Axe B — Objectif contrastif & collapse

Questions ouvertes :
- InfoNCE à τ = 0,1 + petits batches (base 3–4k, budget résidus) : risque de
  **collapse dimensionnel** (sous-espace de faible rang) déjà observé dans nos runs.
  Faut-il passer à un objectif **sans négatifs** qui régularise explicitement la
  variance/covariance (VICReg, Barlow Twins) plutôt que de compter sur les négatifs ? [dimcol] [vicreg] [barlow]
- Le collapse observé est-il complet (tous les vecteurs parallèles) ou dimensionnel
  (quelques directions mortes) ? Diagnostiquer via rang effectif / spectre de la
  covariance (déjà prévu dans les health checks). [dimcol] [alignunif]
- Quelle température ? τ trop bas → underflow numérique et sur-concentration sur de
  rares négatifs ; lien direct avec le collapse. [alignunif] [vues]
- Une **tête de projection** + normalisation (whitening) avant la perte aide-t-elle ?
  (standard depuis SimCLR, absent ici.) [dimcol]

Lectures : `[dimcol]` Understanding Dimensional Collapse, `[vicreg]` VICReg,
`[barlow]` Barlow Twins, `[alignunif]` Alignment & Uniformity.

---

## Axe C — Négatifs & faux négatifs

Questions ouvertes :
- Le hard-negative mining par batch (regrouper des protéines de longueur/composition
  SSE similaires) **fabrique-t-il des faux négatifs** : deux protéines du même fold,
  superficiellement semblables, poussées loin l'une de l'autre alors qu'elles
  devraient être proches ? C'est exactement le scénario que l'objectif veut éviter. [hardneg] [debias]
- Sur le virome (≈ 40 % de singletons, très divergent), la notion de « négatif » est-elle
  bien posée ? Faut-il un objectif **débiaisé** corrigeant la probabilité de faux
  négatifs ? [debias]
- Compromis hard-negatives vs faux-négatifs : durcir les négatifs améliore le signal
  mais augmente le taux de faux négatifs — où placer le curseur ? [hardneg]

Lectures : `[hardneg]` Contrastive Learning with Hard Negative Samples, `[debias]`
Debiased Contrastive Learning.

---

## Axe D — Évaluation, fuite et splits (le plus critique)

Questions ouvertes :
- **Circularité résiduelle** : même en non supervisé, on évalue l'ARI/NMI contre les
  clusters Foldseek — c'est-à-dire contre l'outil qu'on veut « battre ». Tant que la
  vérité terrain est Foldseek, « 2,5× meilleur que Foldseek » n'a pas de sens. Faut-il
  un **benchmark structural indépendant** (SCOP/CATH, ou TM-align direct comme cible
  continue) ? [leak-ppi] [datasail] [afdb]
- Le split phylo retire la fuite **taxonomique** mais pas **structurale** : des
  near-duplicates structuraux peuvent traverser train/val. Adopter un découpage
  leakage-aware (par cluster structural, pas par taxon). [datasail] [leak-ppi]
- Foldseek lui-même est imparfait (seuils E-value/recouvrement) : utiliser ses clusters
  comme vérité importe son bruit dans toutes nos métriques. Quelle référence « or » ? [afdb]
- Garder les témoins du plan (embedding aléatoire, modèle non entraîné, permutation des
  clusters) — ils détectent les artefacts mais ne soignent pas la circularité.

Lectures : `[datasail]` DataSAIL (splits anti-fuite), `[leak-ppi]` Revealing Data
Leakage in Protein Interaction Benchmarks, `[afdb]` Barrio-Hernandez (clustering AFDB
Foldseek/MMseqs2).

---

## Axe E — Métriques de clustering

Questions ouvertes :
- L'ARI seul masque le sens de l'erreur (sur-découpage vs sur-fusion) → compléter par
  homogénéité/complétude/V-measure et Fowlkes–Mallows (cf. `plan_implementation.md`
  Mod 4). Lues **par paires** uniquement.
- Métriques sans clustering (TM-recall, corrélation distance↔TM) : moins sensibles aux
  paramètres HDBSCAN, plus robustes. À privilégier comme métriques primaires. [tmvec] [foldex]
- Le nombre de clusters est un descriptif dépendant d'HDBSCAN, pas une métrique de
  qualité (cf. Mod 5).

Lectures : `[tmvec]` TM-Vec, `[foldex]` FoldExplorer (métriques de recherche
structurale), + références métriques dans `plan_implementation.md` Mod 4.

---

## Axe F — Géométrie : invariance (notre choix) vs équivariance

Questions ouvertes :
- Notre modèle est **SE(3)-invariant par construction** (features de distances/angles/
  valeurs propres). Réduire toute l'information vectorielle à des scalaires invariants
  **perd-il** de la géométrie utile, vs un réseau **équivariant** à canaux vectoriels
  (GVP-GNN ; TCPNet de Topotein) ? [gvp] [geogeo] [topotein]
- Ablation à faire : même complexe, encodeur invariant scalaire vs encodeur équivariant.
  Le gain d'équivariance justifie-t-il la complexité ? [geogeo]

Lectures : `[gvp]` GVP-GNN, `[geogeo]` Improving Molecular Modeling with Geometric
GNNs (étude empirique invariant vs équivariant), `[topotein]` Topotein/TCPNet.

---

## Axe G — Encodage de la distance (cf. Mod 1)

Questions ouvertes :
- L'encodage sinusoïdal NLP appliqué à une distance physique n'est pas canonique ;
  RBF gaussiennes (SchNet) ou base de Bessel (DimeNet) sont liées à une échelle de
  longueur. Laquelle, calibrée comment ? [schnet] [dimenet]
- Bessel > gaussien à dimension moindre (DimeNet) : tester les deux en ablation. [dimenet]

Lectures : `[schnet]` SchNet, `[dimenet]` DimeNet.

---

## Axe H — Positionnement vs état de l'art (critique « existentielle »)

Questions ouvertes :
- Pourquoi un encodeur appris plutôt que **Foldseek / FoldExplorer / Progres**
  directement, qui font déjà de la recherche structurale par embedding et trouvent des
  homologues lointains ? Quel est l'apport net (vitesse ? clustering global du virome ?
  nouveauté biologique) ? Si la seule référence est Foldseek et qu'on ne le bat pas
  nettement, quelle est la contribution ? [foldex] [progres] [tmvec] [plmsearch]
- En quoi notre embedding diffère-t-il de ces méthodes (qui combinent souvent séquence
  + structure) ? [foldex] [saprot]

Lectures : `[foldex]` FoldExplorer, `[progres]` Progres (structure graph embeddings),
`[tmvec]` TM-Vec, `[plmsearch]` PLMSearch, `[saprot]` SaProt.

---

## Axe I — Justification du choix TDL (complexe combinatoire) vs GNN géométrique plat

Questions ouvertes :
- La hiérarchie (résidu/SSE/protéine) du complexe combinatoire **aide-t-elle vraiment**
  vs un GNN géométrique plat (GearNet, GVP) ? Ablation des rangs (retirer le rang 2,
  le rang 3) indispensable. [reprend1] [gvp] [topotein]
- Les benchmarks TDL montrent que le gain n'est pas systématique : à quantifier sur
  *notre* tâche, pas supposé. [topobench] [tdlchallenge]

Lectures : `[topotein]` Topotein, `[hoan]` Going Beyond Graph Data, `[topobench]`
TopoBench, `[tdlchallenge]` ICML TDL Challenge 2024.

---

## Axe J — Domaine : clustering structural du virome

Questions ouvertes :
- Le virome est un cas extrême (hyper-divergence, ≈ 40 % de singletons) : les méthodes
  généralistes (clustering AFDB) s'y transfèrent-elles ? [afdb] [viralafdb] [caudo]
- Notre dataset (Nomburg) : quelle est la définition exacte de ses « familles » (à
  confirmer dans l'article) et comment se compare-t-elle aux taxonomies structurales
  virales récentes ? [nomburg] [viralafdb] [caudo]

Lectures : `[nomburg]` Nomburg (notre dataset), `[viralafdb]` Viral AlphaFold Database,
`[caudo]` Taxonomie structurale des Caudoviricetes.

---

## Axe K — Construction de la sous-base de prototypage (~3 600 protéines) et impact sur le protocole

Construction réelle (cf. `filter_dataset.py`) : (1) pLDDT ≥ 70 + appartenance à un
cluster Foldseek **multi-membres** (singletons et `undefined_family` exclus) →
~32 000 protéines ; (2) downsampling en gardant **au plus 2 protéines par cluster**
+ quelques protéines-cibles choisies à la main (LigT/OB-fold…) → **3 647 protéines**.

Critiques / questions ouvertes :
- **Singletons supprimés** : le modèle ne voit jamais le cas « pas de voisin
  structural » alors qu'ils font ~40 % du jeu complet. À l'évaluation sur le jeu
  complet, risque de **sur-clusteriser** les singletons (faux positifs). De plus, la
  métrique `singleton_frac` n'a aucun sens sur une base sans singletons. [imbssl] [afdb]
- **Distribution des tailles aplatie** : « max 2 par cluster » détruit la loi de
  puissance naturelle (quelques gros clusters, beaucoup de petits) et la rend
  quasi-uniforme. → **décalage de distribution** entre l'entraînement (proto) et
  l'évaluation (complet) ; les statistiques de positifs/négatifs vues par le
  contrastif sont artificielles. [imbssl] [vues]
- **Petite taille (3,6k)** : le contrastif est gourmand en négatifs/données (grands
  batches : SimCLR, file de négatifs : MoCo). Sur 3,6k, pression d'uniformité plus
  faible → **collapse plus probable**, sur-apprentissage facile, résultats peu
  transférables. Nuance : la SSL est plus robuste au déséquilibre que le supervisé
  (Liu et al.), mais ça ne corrige pas le décalage de distribution induit par le
  sous-échantillonnage. [moco] [imbssl]
- **Double intrication avec Foldseek** : les clusters Foldseek servent à la fois à
  **sélectionner** la sous-base et (plus tard) d'**étiquettes d'évaluation** → biais
  de sélection aligné sur la vérité terrain, qui s'ajoute à la circularité de l'axe D.
- **Stratification par mots-clés** (LigT, OB-fold…) : injecte un choix de folds fait
  main → sous-base non représentative du virome ; les résultats peuvent ne pas
  généraliser. [viralafdb]
- **Clusters de taille 2 à l'évaluation** : sur la proto-base, les « vrais » clusters
  ont ≤ 2 membres → HDBSCAN (qui a un `min_cluster_size`) peut les classer en bruit ;
  l'ARI y devient très sensible aux paramètres et peu comparable au jeu complet.

Impact sur le protocole expérimental :
- Les résultats sur la proto-base servent **uniquement au débogage du pipeline**, pas
  à une conclusion biologique : le protocole doit l'écrire explicitement (fait dans
  `protocol.tex` §\ref prototyping) et prévoir un **re-run sur le jeu complet** avant
  toute revendication.
- Les métriques (ARI, `singleton_frac`) calculées sur la proto-base sont optimistes et
  non transférables ; ne pas les comparer aux baselines du jeu complet.
- Séparer clairement, dans le protocole, le régime « prototypage » du régime
  « production » (jeu complet, avec singletons et distribution réelle).

Lectures : `[moco]` MoCo (négatifs/batch), `[imbssl]` SSL robuste au déséquilibre,
`[vues]` InfoMin, `[afdb]` clustering AFDB, `[viralafdb]` Viral AFDB.

---

## Axe L — Valeur ajoutée scientifique et reproduction de Nomburg

C'est la question existentielle du projet. Réponse directe, en deux temps.

### 1. Reconstruire les clusters Foldseek n'est PAS une contribution
Si le modèle ne fait que retrouver les clusters Foldseek, il **distille Foldseek** :
c'est une validation (preuve que l'embedding capture la structure), pas un apport.
La valeur doit venir de quelque chose que le pipeline de Nomburg (ColabFold →
MMseqs2 → Foldseek → recherche vs bases non-virales → annotation) ne fait pas, ou
fait moins bien.

### 2. L'argument « TDL pour passer à l'échelle » est faible — à corriger
La prémisse « l'apport de la TDL est de manipuler de plus grandes bases » ne tient
pas en l'état :
- **Foldseek passe déjà à l'échelle de l'univers protéique** : Barrio-Hernandez et
  al. ont clusterisé les 214 M de structures de l'AFDB avec Foldseek. [afdb]
- L'**atlas métagénomique ESM** (617 M de structures) existe déjà, prédites et
  explorées sans TDL. [esmatlas]
- La TDL est **plus** coûteuse par échantillon qu'un GNN plat ou qu'un alignement ;
  « TDL = scalabilité » n'est donc pas évident et doit être justifié, pas supposé.
- L'avantage de scalabilité réel est celui d'un **embedding** (récupération
  vectorielle amortie O(1), index ANN) — mais il est **partagé** avec FoldExplorer /
  Progres / TM-Vec, il n'est pas propre à la TDL. [foldex] [progres] [tmvec]

Conclusion : ne pas vendre la TDL sur la scalabilité. La vendre sur ce qui lui est
**propre** : la **représentation hiérarchique multi-échelle**.

### 3. Où est la vraie valeur ajoutée (par force croissante)
1. **Représentation continue réutilisable** : un embedding de taille fixe alimente
   des tâches aval (prédiction de fonction, modèles génératifs, recherche) que des
   clusters discrets ne permettent pas. Apport modeste, déjà bien exploré. [deepfri] [survey]
2. **Homologie partielle / au niveau domaine via la hiérarchie TDL** : le complexe
   combinatoire (résidu → SSE → protéine) est l'outil naturel pour détecter des
   **sous-structures partagées** (p. ex. un domaine OB-fold inséré dans des protéines
   par ailleurs différentes) — précisément le type d'homologie lointaine/partielle
   qui intéresse Nomburg (le domaine RNA ligase T). **C'est le différenciateur TDL le
   plus défendable**, mais il faut le **démontrer** (ablation des rangs : la hiérarchie
   trouve-t-elle des homologues de sous-domaine que les méthodes plates ratent ?). [topotein] [hoan]
3. **Sensibilité à l'homologie distante** : si l'embedding est plus sensible que
   Foldseek/DALI, on peut **mordre dans les 62 % « sombres »** (structurellement
   distincts, sans homologue) de Nomburg et proposer de nouvelles annotations. C'est
   l'apport le plus fort (digne de Nature), mais le plus dur à prouver et il faut une
   validation forte. [esmatlas] [foldex] [tmvec]

### 4. Comment retrouver (et étendre) les résultats de Nomburg
Pipeline de Nomburg à répliquer : prédiction → clustering 2 étapes (MMseqs2 séquence
20 % identité, puis Foldseek structure ; 5 770 clusters multi-membres + 12 422
singletons) → recherche structurale vs protéines **non-virales** → annotation par
analogie → validation expérimentale (RNA ligase T / cGAMP).

Protocole de reproduction par embedding/TDL :
- **(Validation)** Reproduire le clustering structural et le comparer aux 5 770
  clusters de Nomburg (ARI + métriques directionnelles de la Mod 4). Sanity check,
  pas contribution.
- **Analogues inter-règnes** : encoder les protéines virales **et** non-virales (AFDB/
  PDB) dans le même espace, puis chercher les plus proches voisins non-viraux d'une
  protéine virale → retrouver les analogues hôte/pathogène de Nomburg. C'est ici que
  la couverture « millions de protéines » devient réellement utile.
- **Transfert d'annotation** : hériter la fonction du plus proche voisin annoté →
  reproduire les inférences fonctionnelles (jusqu'à 25 % des non-annotées). [deepfri]
- **Cas-test falsifiable** : vérifier que l'embedding place les homologues RNA ligase
  T des poxvirus aviaires près des phosphodiestérases de phages connues. À noter : la
  sous-base de prototypage **garde déjà** les protéines-cibles LigT / OB-fold / SSB
  (cf. `filter_dataset.py`) — le projet est donc déjà câblé pour ce test précis.
- **Extension = la vraie contribution** : trouver de **nouveaux** analogues dans les
  62 % sombres que Nomburg n'a pas annotés, idéalement avec une validation
  computationnelle forte (et, à terme, expérimentale).

### 5. Vérité inconfortable à assumer
Si le seul résultat est « un modèle TDL reconstruit les clusters Foldseek », c'est un
résultat **outil/ingénierie** (un encodeur structural scalable), publiable en
bioinformatique, mais **pas** une découverte biologique au sens de Nomburg. Et le
créneau « recherche structurale par embedding sensible » est **déjà encombré**
(FoldExplorer, Progres, TM-Vec, SaProt). Le seul élément qui peut vraiment nous
différencier est la **hiérarchie TDL pour l'homologie partielle/de domaine** — donc
toute la stratégie expérimentale devrait viser à **prouver cet apport spécifique**
(ablations des rangs, cas de sous-domaine, comparaison à un GNN plat), faute de quoi
on ré-implémente FoldExplorer en plus lourd.

Lectures : `[nomburg]`, `[afdb]`, `[esmatlas]`, `[foldex]`, `[progres]`, `[tmvec]`,
`[deepfri]`, `[saprot]`, `[topotein]`.

---

# Bibliographie catégorisée

> Les identifiants arXiv des articles ML « classiques » (VICReg, Barlow Twins,
> dimensional collapse, SchNet, DimeNet, débiaisé, alignment/uniformity) sont donnés
> de mémoire — à recontrôler avant citation formelle dans le mémoire.

### Notre méthodo / proche (CL sur structure protéique)
- `[reprend1]` Zhang et al., *Protein Representation Learning by Geometric Structure
  Pretraining* (GearNet), ICLR 2023. `[REPREND]` arXiv:2203.06125 —
  https://arxiv.org/abs/2203.06125
- `[reprend2]` Hermosilla, Ropinski, *Contrastive Representation Learning for 3D
  Protein Structures*, 2022. `[REPREND]` arXiv:2205.15675 —
  https://arxiv.org/abs/2205.15675
- `[ccpl]` *CCPL: Cross-modal Contrastive Protein Learning*, 2023. `[REPREND]`
  arXiv:2303.11783 — https://arxiv.org/abs/2303.11783
- `[splm]` Wang et al., *S-PLM: Structure-Aware Protein Language Model via Contrastive
  Learning*, Advanced Science, 2025. `[ALTERNATIVE]` —
  https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202404212
- `[sspro]` *SS-Pro: a simplified Siamese contrastive learning approach for protein
  surface representation*, Front. Comput. Sci., 2024. `[REPREND]` —
  https://link.springer.com/article/10.1007/s11704-024-3806-9
- `[survey]` *A Survey on Protein Representation Learning: Retrospect and Prospect*,
  2023. `[FONDATION]` arXiv:2301.00813 — https://arxiv.org/abs/2301.00813

### Alternative : vraies paires / cibles structurales
- `[prottucker]` Heinzinger et al., *Contrastive learning on protein embeddings
  enlightens midnight zone* (ProtTucker), NAR Genom. Bioinform., 2022. `[ALTERNATIVE]`
  doi:10.1093/nargab/lqac043 (à vérifier).
- `[tmvec]` Hamamsy et al., *Protein remote homology detection and structural
  alignment using deep learning* (TM-Vec), Nat. Biotechnol., 2024. `[ALTERNATIVE]` —
  https://www.nature.com/articles/s41587-023-01917-2
- `[foldex]` *FoldExplorer: Fast and Accurate Protein Structure Search with
  Sequence-Enhanced Graph Embedding*, 2023. `[ALTERNATIVE]` `[SIMILAIRE]`
  arXiv:2311.18219 — https://arxiv.org/abs/2311.18219
- `[progres]` *Fast protein structure searching using structure graph embeddings*
  (Progres), Bioinformatics Advances, 2025. `[ALTERNATIVE]` `[SIMILAIRE]` —
  https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf042/8107707
- `[plmsearch]` *PLMSearch: protein language model powers accurate and fast sequence
  search for remote homology*, 2024. `[ALTERNATIVE]` (lien à vérifier).

### Critique de l'augmentation / bruit / vues
- `[vues]` Tian et al., *What Makes for Good Views for Contrastive Learning?*
  (InfoMin), NeurIPS 2020. `[CRITIQUE]` arXiv:2005.10243 —
  https://arxiv.org/abs/2005.10243
- `[denoise]` Zaidi et al., *Pre-training via Denoising for Molecular Property
  Prediction*, ICLR 2023. `[CRITIQUE]` arXiv:2206.00133 —
  https://arxiv.org/abs/2206.00133
- `[rigid]` *Rigidity-Aware Geometric Pretraining for Protein Design and Conformational
  Ensembles* (RigidSSL), 2026. `[CRITIQUE]` arXiv:2603.02406 —
  https://arxiv.org/abs/2603.02406
- `[geossl]` *Molecular Geometry Pretraining with SE(3)-Invariant Denoising Distance
  Matching*, 2022. `[ALTERNATIVE]` arXiv:2206.13602 —
  https://arxiv.org/abs/2206.13602

### Collapse & objectifs sans négatifs
- `[dimcol]` Jing et al., *Understanding Dimensional Collapse in Contrastive
  Self-Supervised Learning*, ICLR 2022. `[CRITIQUE]` arXiv:2110.09348 —
  https://arxiv.org/abs/2110.09348
- `[vicreg]` Bardes, Ponce, LeCun, *VICReg: Variance-Invariance-Covariance
  Regularization for SSL*, ICLR 2022. `[ALTERNATIVE]` arXiv:2105.04906 —
  https://arxiv.org/abs/2105.04906
- `[barlow]` Zbontar et al., *Barlow Twins: Self-Supervised Learning via Redundancy
  Reduction*, ICML 2021. `[ALTERNATIVE]` arXiv:2103.03230 —
  https://arxiv.org/abs/2103.03230
- `[alignunif]` Wang, Isola, *Understanding Contrastive Representation Learning through
  Alignment and Uniformity on the Hypersphere*, ICML 2020. `[FONDATION]`
  arXiv:2005.10242 — https://arxiv.org/abs/2005.10242

### Négatifs / faux négatifs
- `[hardneg]` Robinson et al., *Contrastive Learning with Hard Negative Samples*,
  ICLR 2021. `[CRITIQUE]` arXiv:2010.04592 — https://arxiv.org/abs/2010.04592
- `[debias]` Chuang et al., *Debiased Contrastive Learning*, NeurIPS 2020.
  `[CRITIQUE]` arXiv:2007.00224 — https://arxiv.org/abs/2007.00224

### Taille de données / négatifs / déséquilibre (sous-base de prototypage)
- `[moco]` He et al., *Momentum Contrast for Unsupervised Visual Representation
  Learning* (MoCo), CVPR 2020. `[ALTERNATIVE]` arXiv:1911.05722 —
  https://arxiv.org/abs/1911.05722
- `[imbssl]` Liu et al., *Self-supervised Learning is More Robust to Dataset
  Imbalance*, ICLR 2022. `[CRITIQUE]` arXiv:2110.05025 —
  https://arxiv.org/abs/2110.05025

### Évaluation / fuite / splits
- `[datasail]` *Data splitting to avoid information leakage with DataSAIL*, Nat.
  Commun., 2025. `[CRITIQUE]` — https://www.nature.com/articles/s41467-025-58606-8
- `[leak-ppi]` *Revealing data leakage in protein interaction benchmarks*, 2024.
  `[CRITIQUE]` arXiv:2404.10457 — https://arxiv.org/abs/2404.10457
- `[afdb]` Barrio-Hernandez et al., *Clustering predicted structures at the scale of
  the known protein universe*, Nature, 2023. `[SIMILAIRE]` `[ALTERNATIVE]` —
  https://www.nature.com/articles/s41586-023-06510-w

### Géométrie : invariance vs équivariance
- `[gvp]` Jing et al., *Learning from Protein Structure with Geometric Vector
  Perceptrons* (GVP-GNN), ICLR 2021. `[ALTERNATIVE]` arXiv:2009.01411 —
  https://arxiv.org/abs/2009.01411
- `[geogeo]` *Improving Molecular Modeling with Geometric GNNs: an Empirical Study*,
  2024. `[CRITIQUE]` arXiv:2407.08313 — https://arxiv.org/abs/2407.08313
- `[gunet]` *Multi-Scale Protein Structure Modelling with Geometric Graph U-Nets*,
  2025. `[CRITIQUE]` `[ALTERNATIVE]` (concurrent direct de la thèse multi-échelle ;
  bat invariants/équivariants en fold classification) arXiv:2512.06752 —
  https://arxiv.org/abs/2512.06752

### Encodage de distance
- `[schnet]` Schütt et al., *SchNet: A continuous-filter convolutional neural network
  for modeling quantum interactions*, NeurIPS 2017. `[ALTERNATIVE]` arXiv:1706.08566 —
  https://arxiv.org/abs/1706.08566
- `[dimenet]` Gasteiger et al., *Directional Message Passing for Molecular Graphs*
  (DimeNet), ICLR 2020. `[ALTERNATIVE]` arXiv:2003.03123 —
  https://arxiv.org/abs/2003.03123

### PLM structure-aware (3Di, comme nous)
- `[saprot]` Su et al., *SaProt: Protein Language Modeling with Structure-aware
  Vocabulary*, ICLR 2024. `[REPREND]` `[ALTERNATIVE]` —
  https://www.biorxiv.org/content/10.1101/2023.10.01.560349

### Topological deep learning
- `[hoan]` Hajij et al., *Topological Deep Learning: Going Beyond Graph Data*, 2022.
  `[FONDATION]` arXiv:2206.00606 — https://arxiv.org/abs/2206.00606
- `[topotein]` Wang, Jamasb et al., *Topotein: Topological Deep Learning for Protein
  Representation Learning*, 2025. `[REPREND]` arXiv:2509.03885 —
  https://arxiv.org/abs/2509.03885
- `[topobench]` *TopoBench: A Framework for Benchmarking Topological Deep Learning*,
  2024. `[CRITIQUE]` arXiv:2406.06642 — https://arxiv.org/abs/2406.06642
- `[tdlchallenge]` *ICML Topological Deep Learning Challenge 2024: Beyond the Graph
  Domain*, 2024. `[FONDATION]` arXiv:2409.05211 — https://arxiv.org/abs/2409.05211

### Échelle / dark proteome / fonction par structure (valeur ajoutée, axe L)
- `[esmatlas]` Lin et al., *Evolutionary-scale prediction of atomic-level protein
  structure with a language model* (ESMFold / ESM Metagenomic Atlas, 617 M
  structures), Science, 2023. `[SIMILAIRE]` `[ALTERNATIVE]` —
  https://www.science.org/doi/10.1126/science.ade2574
- `[deepfri]` Gligorijević et al., *Structure-based protein function prediction using
  graph convolutional networks* (DeepFRI), Nat. Commun., 2021. `[ALTERNATIVE]` —
  https://www.nature.com/articles/s41467-021-23303-9

### Domaine : virome / clustering structural viral
- `[nomburg]` Nomburg et al., *Birth of protein folds and functions in the virome*,
  Nature, 2024. `[SIMILAIRE]` — https://doi.org/10.1038/s41586-024-07809-y
- `[viralafdb]` *The Viral AlphaFold Database of monomers and homodimers...*, Science
  Advances, 2025. `[SIMILAIRE]` — https://www.science.org/doi/10.1126/sciadv.adz8560
- `[caudo]` *A novel approach to Caudoviricetes taxonomy utilising whole proteome
  structure-structure comparison*, bioRxiv, 2025. `[SIMILAIRE]` —
  https://www.biorxiv.org/content/10.1101/2025.08.06.668922

---

## Synthèse : les 3 questions les plus structurantes
1. **Évaluation (axe D)** : tant que la vérité terrain est Foldseek, le projet est
   circulaire. Décider d'un benchmark structural indépendant est la priorité numéro un.
2. **Collapse (axe B)** : en non supervisé pur (sans TM-aux/SupCon), le collapse est le
   risque numéro un ; envisager VICReg/Barlow + tête de projection avant de pousser les
   epochs.
3. **Positionnement (axe H)** : clarifier l'apport vs Foldseek/FoldExplorer/Progres,
   sinon le meilleur résultat possible reste « on imite Foldseek ».
