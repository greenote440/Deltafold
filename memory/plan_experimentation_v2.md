# Plan d'expérimentation — prochaine run (sous-base corrigée)

Objet : transformer les questions de `revue_litterature.md` en un protocole testable.
Ce document (1) répond systématiquement aux articles tagués `[CRITIQUE]`, (2) traite
les deux questions clés (apport des complexes combinatoires / analogues sombres ;
circularité), puis (3) spécifie sous-base, configurations, ablations, métriques,
baselines et portes de décision pour la run suivante. Cadrage acté :
**non supervisé pur, sans TM-aux ni SupCon**.

---

## 1. Réponses systématiques aux articles critiques

| Article | Préoccupation | Décision / action pour la run |
|---|---|---|
| `[dimcol]` Dimensional collapse | Collapse si variance d'augmentation > variance des données ; régularisation implicite des couches | Ajouter une **tête de projection** (absente aujourd'hui) ; **calibrer σ** sous la variance structurale inter-protéines ; suivre le **rang effectif** (porte de santé). Tester aussi VICReg/Barlow. |
| `[vues]` InfoMin | Vues optimales = minimiser l'info partagée tout en gardant l'info utile ; un « sweet spot » d'augmentation | **Mesurer le TM-score entre les deux vues** ; balayer σ pour situer le sweet spot (ni triviale ni destructrice). |
| `[denoise]` Pre-training via Denoising | Bruit gaussien justifié seulement petit, autour de l'équilibre (≈ champ de forces) | Garder σ **petit** (0,3–0,5 Å) ; envisager un régularisateur de **débruitage** non supervisé (sans fuite) comme variante. |
| `[rigid]` RigidSSL | Perturbations naïves par atome = non physiques | Tester des perturbations **rigidity-aware** (corps rigides / torsions) vs jitter cartésien ; valider que le jitter ne casse pas la géométrie (clashs). |
| `[hardneg]` Hard negatives | Durcir les négatifs aide mais crée des faux négatifs | Traiter la dureté comme **hyperparamètre** ; balayer ; coupler au débiaisage. |
| `[debias]` Debiased CL | Faux négatifs (mêmes folds échantillonnés comme négatifs) | Le hard-neg-by-batch regroupe des protéines similaires → risque élevé de faux négatifs ; tester un **objectif débiaisé** ou exclure les voisins structuraux probables des négatifs. |
| `[imbssl]` SSL & déséquilibre | La SSL est plus robuste au déséquilibre que le supervisé | Argument **pour** le non supervisé sur le virome long-tail ; **corollaire** : ne pas aplatir la distribution (cf. sous-base, §4) pour garder ce bénéfice. |
| `[datasail]` DataSAIL | Splits aléatoires fuient ; besoin de splits « cold » | Adopter un **split cold / cluster-aware** (par cluster structural), pas aléatoire ni seulement par taxon. |
| `[leak-ppi]` Data leakage benchmarks | Near-duplicates structuraux gonflent les scores | **Dédupliquer** train/val par similarité structurale ; rapporter la performance **en fonction de la similarité au train**. |
| `[geogeo]` Geometric GNN empirique | Réduire les vecteurs à des scalaires ne capture pas toute la géométrie | Justifie l'**ablation invariant (scalaire, notre choix) vs équivariant** ; ne pas supposer que l'invariant suffit. |
| `[topobench]` TopoBench | Le gain de la TDL est dépendant de la tâche, non garanti | Interdiction de **supposer** que la hiérarchie aide : la **prouver** par ablation des rangs et comparaison aux GNN plats (§2). |

---

## 2. Question clé — montrer l'apport des complexes combinatoires (hiérarchie) et trouver des analogues sombres

### 2.1 Le principe : tester là où les méthodes globales échouent
Foldseek/TM-align comparent des structures **globalement** (chaîne entière) ;
FoldExplorer/Progres/TM-Vec produisent **un seul vecteur** par protéine. La revendication
propre au complexe combinatoire est l'accès **multi-échelle** : représentations de rang-2
(SSE / sous-domaine) et rang-3 (protéine). Pour démontrer un apport, il faut des tâches où
la similarité **partielle / de sous-domaine** compte et où le global échoue.

### 2.2 Le test décisif : récupération d'homologie partielle / de domaine
- Construire un jeu de paires qui ne partagent **qu'un seul domaine** (p. ex. un OB-fold
  inséré dans des contextes différents ; domaines multi-domaines ; insertions). Le TM-score
  **global** de ces paires est faible → Foldseek global les manque.
- Métrique : **rappel de ces paires « sous-domaine »** par notre embedding multi-échelle
  (voisinage au niveau rang-2), comparé aux méthodes globales. Si on les retrouve et qu'elles
  les manquent, c'est l'apport propre de la hiérarchie.

### 2.3 Baselines forts (pas d'épouvantail)
La barre honnête n'est pas « Foldseek tout court » mais :
- Foldseek mode global + mode local/TMalign ;
- FoldExplorer / TM-Vec (embedding global) ; `[foldex]` `[tmvec]`
- **GNN géométrique plat** (GearNet, GVP) — même complexe sans hiérarchie ; `[gvp]`
- **Décomposition en domaines (Chainsaw/Merizo) + Foldseek par domaine** — le vrai
  concurrent pour l'homologie partielle ;
- **Geometric Graph U-Nets** (multi-échelle géométrique récent qui bat invariants/
  équivariants en fold classification) — concurrent direct de la thèse « multi-échelle ». `[gunet]`

Si la hiérarchie TDL ne bat pas « décomposition en domaines + Foldseek » ni les U-Nets
géométriques sur l'homologie partielle, la revendication tombe — à assumer.

### 2.4 Ablation des rangs (falsifiable)
Entraîner : PCC complet (rangs 0–3) vs **− rang 2** (sans SSE) vs **− rang 3** (sans
global, ≈ plat) vs GNN plat. Mesurer le **delta sur la récupération partielle**, pas sur
le whole-chain (où tout le monde se vaut). Un delta nul = la hiérarchie n'apporte rien
(résultat négatif honnête).

### 2.5 Exposer les embeddings multi-échelles comme livrable
Produire et indexer les embeddings **de rang-2 (sous-domaine)** en plus du rang-3. Une
recherche au niveau domaine est quelque chose qu'un vecteur global unique ne permet pas.

### 2.6 Trouver de nouveaux analogues sombres — protocole
1. Encoder les protéines virales **sombres** (les 62 % sans homologue AFDB de Nomburg) **et**
   une grande base de référence (AFDB / PDB / atlas ESM) dans le **même espace**. `[esmatlas]` `[afdb]`
2. Pour chaque requête sombre, chercher les plus proches voisins **à plusieurs échelles**
   (rang-3 global ET rang-2 sous-domaine). Une protéine sans homologue **global** peut avoir
   un **sous-domaine** qui matche un fold connu → analogue nouveau que le global a manqué.
3. **Contrôle des faux positifs** : FDR via leurres (séquences/structures mélangées) et via
   le témoin de permutation ; ne retenir que les hits sous un seuil de FDR.
4. **Contrôles positifs Nomburg** : vérifier que les homologues RNA ligase T (poxvirus aviaires)
   tombent près des phosphodiestérases de phages, et les cas OB-fold/SSB (déjà dans la
   sous-base via les protéines-cibles).
5. **Validation externe** : pour les hits, transfert d'annotation (EC/GO/Pfam) du voisin
   annoté, et cohérence avec la littérature ; à terme, validation expérimentale (hors scope).

---

## 3. Question — circularité résiduelle

### 3.1 Le problème
Évaluer contre les clusters Foldseek = circulaire (Foldseek est à la fois l'outil à battre
et la vérité terrain). Vrai même en non supervisé.

### 3.2 Solutions (à mettre en place)
- **Vérité indépendante de Foldseek** : classification structurale **CATH/SCOP** (curatée,
  pas issue de Foldseek). Évaluer la récupération de fold/superfamille — c'est ce qu'utilisent
  ProtTucker/TM-Vec, et ça brise la circularité. `[tmvec]` `[prottucker]`
- **Cible structurale continue par TM-align** (et non Foldseek) : corréler la distance
  d'embedding au **TM-score TM-align**. TM-align est l'étalon d'alignement, pas l'outil qu'on
  prétend battre → bien moins circulaire que les clusters Foldseek.
- **Triangulation multi-références** : un résultat est crédible s'il s'accorde avec **plusieurs**
  références indépendantes (Foldseek **et** DALI **et** CATH), pas une seule.
- **Validation par la fonction** (signal externe, non utilisé à l'entraînement ni par le
  clustering Foldseek) : nos clusters prédisent-ils mieux l'**EC/GO/Pfam** que le hasard et que
  Foldseek ? Une meilleure cohérence fonctionnelle = apport **non circulaire**. `[deepfri]`
- **Reframing du but** : l'objectif n'est pas « battre Foldseek au clustering » (circulaire, et
  Foldseek le fait déjà à l'échelle de l'AFDB). C'est une **représentation multi-échelle
  réutilisable + découverte d'analogues sombres**, évaluée sur des signaux **indépendants**
  (CATH, TM-align, fonction, cas Nomburg). Changer la revendication dissout la circularité.

### 3.3 Règle de lecture
Séparer nettement : **(a) validation que l'embedding capture la structure** = accord avec
Foldseek (circulaire, OK comme sanity check) ; **(b) preuve de contribution** = accord avec des
références **indépendantes** (CATH / TM-align / fonction / analogues sombres). Seul (b) soutient
une revendication scientifique. Les témoins (embedding aléatoire, modèle non entraîné,
permutation des clusters) bornent l'artefact mais **ne soignent pas** la circularité.

---

## 4. Sous-base corrigée (corriger l'axe K)

Problèmes de la base actuelle (3 647) : singletons retirés, distribution aplatie (max 2/cluster),
folds choisis à la main, intrication sélection/évaluation. Corrections :
- **Garder une part de singletons** représentative (le modèle doit voir le cas « pas de voisin »).
- **Préserver la distribution des tailles** (échantillonnage proportionnel, pas « max 2 ») pour
  ne pas casser le long-tail (et garder le bénéfice `[imbssl]`).
- **Retirer la stratification par mots-clés** du jeu d'entraînement/éval (la garder seulement
  comme contrôles positifs **séparés**, pas dans les métriques globales).
- **Split cold / cluster-aware** + **déduplication** des near-duplicates train/val. `[datasail]` `[leak-ppi]`
- Garder une taille de prototypage (~3–5k) mais **représentative** ; documenter que les chiffres
  ne transfèrent qu'approximativement au jeu complet.

---

## 5. Configurations & ablations

Config de base de la run : sous-base corrigée · InfoNCE non supervisé **+ tête de projection** ·
augmentation jitter calibré + masquage (pas de crop) · **encodage distance RBF** (Mod 1) · PCC
complet · invariant scalaire · hard-neg avec test de débiaisage.

Ablations **un facteur à la fois** (sinon explosion combinatoire) :

| Axe | Conditions | Question testée |
|---|---|---|
| A. Architecture | PCC complet vs −rang2 vs −rang3(plat) vs GearNet/GVP | La hiérarchie apporte-t-elle (surtout en homologie partielle, §2.4) ? |
| B. Objectif | InfoNCE vs InfoNCE+proj vs VICReg vs Barlow ; balayage τ | Quel objectif évite le collapse ? `[dimcol][vicreg][barlow]` |
| C. Augmentation | balayage σ ; rigid-aware vs naïf ; crop on/off | Sweet spot des vues ; physique. `[vues][rigid][denoise]` |
| D. Encodage distance | sinusoïdal vs RBF vs Bessel | Encodage canonique. `[schnet][dimenet]` |
| E. Négatifs | hard-neg on/off ; débiaisé on/off | Faux négatifs. `[hardneg][debias]` |
| F. Géométrie (stretch) | invariant scalaire vs équivariant (vecteurs) | Perd-on de la géométrie ? `[geogeo][gvp]` |

---

## 6. Métriques (par catégorie, avec ce qu'elles testent)

1. **Santé / collapse (porte bloquante)** : écart-type d'embedding par dim, cosinus moyen
   hors-diagonale, **rang effectif** de la covariance, **alignement & uniformité**. Si non sain →
   résultats invalides, corriger l'objectif avant tout. `[dimcol][alignunif]`
2. **Accord structural sans clustering (primaire)** : **TM-recall@k** vs vérité **TM-align** ;
   Spearman distance-embedding ↔ TM-score TM-align. Moins dépendant d'HDBSCAN.
3. **Clustering vs références INDÉPENDANTES** : ARI + **V-measure / homogénéité / complétude /
   Fowlkes–Mallows** vs **CATH** (et vs Foldseek **en sanity seulement**) ; indices de
   fragmentation/fusion (Mod 4) ; nombre de clusters (descriptif, Mod 5).
4. **Spécifique TDL** : rappel/AUC d'**homologie partielle** (§2.2) ; **delta d'ablation des
   rangs** (§2.4).
5. **Découverte d'analogues sombres** : nb de protéines sombres avec analogue de sous-domaine
   confiant ; **FDR** (leurres/permutation) ; contrôles positifs Nomburg.
6. **Cohérence fonctionnelle (externe, non circulaire)** : pureté des clusters vis-à-vis
   EC/GO/Pfam, vs baselines. `[deepfri]`
7. **Robustesse** : performance **en fonction de la similarité au train** (fuite) ; stabilité
   sur ≥ 2 graines.

---

## 7. Baselines & contrôles
- **Plancher** : embedding aléatoire (≈ 0), modèle non entraîné.
- **À battre / comparer** : Foldseek (global + local), FoldExplorer, TM-Vec, GearNet/GVP plat,
  décomposition en domaines + Foldseek, Geometric U-Nets.
- **Contrôles** : témoin de permutation des clusters (Mod 3) ; embedding mélangé.

---

## 8. Critères de succès / portes de décision (falsification)
- **Porte 0 (santé)** : si le collapse persiste après tête de projection / VICReg → corriger
  l'objectif avant toute interprétation.
- **Porte 1 (hiérarchie)** : si PCC complet **ne bat pas** −rang2 / GNN plat / (domaines+Foldseek)
  sur l'**homologie partielle** → la hiérarchie n'apporte rien → pivoter ou publier le négatif.
- **Porte 2 (non-circularité)** : si l'accord avec **CATH / fonction** n'excède pas une simple
  distillation de Foldseek → pas de contribution démontrée.
- **Porte 3 (sombre)** : si aucun analogue sombre ne passe le contrôle FDR ni ne reproduit les
  cas Nomburg → la promesse « découverte » n'est pas tenue.

---

## 9. Protocole ordonné de la run
1. Construire la **sous-base corrigée** (§4) + split cold + déduplication.
2. (Encodage RBF calculable **dans le modèle** depuis la distance brute → pas de re-lifting.)
3. **Smoke test** court : confirmer la santé (rang effectif, uniformité) avec tête de projection.
4. **Run de base** + ablations un-facteur (§5), ≥ 2 graines.
5. Calculer les métriques (§6) **dont CATH, TM-align, fonction** (références indépendantes).
6. Tests **homologie partielle** (§2.2) et **analogues sombres** (§2.6) avec FDR.
7. Comparer aux **baselines** (§7) ; appliquer les **portes de décision** (§8).

---

## 10. Risques
- Multiplicité d'expériences → fixer la config de base et n'ablater qu'un facteur à la fois.
- CATH/SCOP peu couvrant pour le virome → utiliser TM-align continu + fonction comme références
  complémentaires.
- Décomposition en domaines bruitée → la traiter comme baseline imparfaite, pas comme vérité.
- Tête de projection + RBF changent les dims → checkpoints incompatibles (ré-entraînement, déjà acté).
