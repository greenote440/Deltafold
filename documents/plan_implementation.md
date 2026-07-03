# Plan d'implémentation — modifications du modèle DeltaFold

Document de travail. On y consigne, remarque par remarque, les modifications à
apporter au modèle suite à la revue avec les superviseur·e·s. Chaque entrée suit
le même gabarit (remarque → problème → changement → implémentation → impact →
validation → statut) pour pouvoir être traitée et cochée indépendamment.

Langue : français ici (doc interne) ; le mémoire/protocole reste en anglais. Dis-moi
si tu veux basculer ce plan en anglais.

## Suivi

| # | Modification | Origine | Priorité | Statut |
|---|--------------|---------|----------|--------|
| 1 | Encodage de la distance : sinusoïdal → RBF gaussiennes | Superviseure bio | Haute | À faire |
| 2 | Stratégie de paires positives : augmentation (jitter/masquage) vs vraies paires TM | Superviseure bio | Haute | Acté : non supervisé, sans TM-aux |
| 3 | Témoin ARI par permutation des clusters (null SupCon) | Superviseure bio | Haute | À faire |
| 4 | Métriques directionnelles d'accord avec Foldseek (homogénéité/complétude, fragmentation/fusion) | Superviseure bio | Moyenne | À faire |
| 5 | Nombre de clusters comme métrique (descriptive, pas qualité) | Superviseure bio | Basse | Déjà loggé, à présenter |
| 6 | Export de clusters pour validation manuelle (FASTA + structures) | Superviseure bio | Moyenne | À faire (après run) |

Statuts : `À faire` · `En cours` · `Fait` · `Abandonné`.

## Décisions actées
- **Non supervisé pur** : InfoNCE sur positifs d'augmentation uniquement. On retire
  le **SupCon** (positifs « même cluster Foldseek ») ET le **terme auxiliaire
  TM-score**, pour éliminer toute fuite côté entraînement. → tranche la Mod 2.
  Impact `protocol.tex` : retirer TM-aux de la section *Training method* et du
  tableau d'hyperparamètres, retirer la mention SupCon, ajuster la commande de
  référence (`--unsupervised`, sans `--tm-aux-weight`).
- **Base de prototypage** (~3 600 protéines) pour les runs courants ; section
  ajoutée à `protocol.tex` (§ prototyping). Résultats = débogage seulement, re-run
  sur jeu complet avant toute conclusion (cf. revue, axe K).
- Conséquence : le collapse (axe B) devient le risque n°1 (plus de TM-aux/SupCon
  pour soutenir le signal) ; la circularité d'évaluation (axe D) reste à traiter.

---

## Modification 1 — Encodage de la distance (sinusoïdal → RBF)

### Remarque (superviseure)
Question du « choix canonique » dans la construction du vecteur associé à la
distance : qu'est-ce qui fixe les fréquences, le nombre de bandes et l'échelle ?

### Problème
L'arête entre deux C-alpha porte la distance scalaire `d_ij` **plus** un encodage
sinusoïdal de cette distance en 16 dimensions, repris tel quel de l'encodage
positionnel des Transformers (`v_2k = sin(d·ω_k)`, `v_2k+1 = cos(d·ω_k)`,
`ω_k = 10000^{-2k/16}`). Trois choix arbitraires, non canoniques :

- la constante de base `10000` et donc la bande de fréquences ;
- le nombre de bandes (16) ;
- l'échelle/les unités de `d` (le sinusoïdal n'est pas invariant au passage Å → nm,
  alors qu'une distance physique a une échelle pertinente : contacts ~4–12 Å).

Conséquence mesurable : avec base 10000 et 16 dims sur des distances de 0–~12 Å,
seules les bandes de haute fréquence varient ; les dernières ont `ω_k ≈ 3e-4`,
donc `sin(d·ω_k) ≈ 0` et `cos ≈ 1` pour toute distance de contact. Près de la
moitié des 16 dimensions sont quasi constantes : capacité gaspillée et encodage
mal adapté à la plage d'entrée.

### Changement proposé
Remplacer l'encodage sinusoïdal par une **expansion en fonctions de base radiales
gaussiennes** (RBF, à la SchNet), liée explicitement à l'échelle physique des
contacts :

```
e_k(d) = exp( -(d - μ_k)^2 / (2 σ^2) ),   k = 1..K
```

- centres `μ_k` régulièrement espacés sur `[d_min, d_max]` calibrés sur la
  distribution réelle des distances d'arête (étape 1 ci-dessous) ;
- largeur `σ ≈` espacement entre centres ;
- `K` au choix (commencer à 16 pour garder la dimension, tester 32).

On conserve la distance scalaire brute en plus de l'expansion (comme aujourd'hui).
Alternatives à garder en tête pour l'ablation : base de Bessel (DimeNet),
fréquences de Fourier recalibrées sur l'échelle Å, centres/largeurs RBF
apprenables.

### Implémentation
Approche recommandée : **calculer la RBF dans le modèle au forward**, à partir de
la distance scalaire déjà stockée (`r1['distance']`), plutôt que dans le lifter.
Cela évite de re-lifter les ~67k protéines ; seul le modèle change.

- `asymmetric_topotein.py`
  - `AsymmetricTopoNet.__init__` : `rank1_emb` passe de `Linear(17, …)` à
    `Linear(1 + K, …)` (distance brute + K canaux RBF). Ajouter un buffer non
    entraînable pour les centres `μ_k` (et `σ`).
  - `forward` : remplacer
    `cat([distance, distance_encoding])` par
    `cat([distance, rbf(distance)])`, où `rbf` calcule les K gaussiennes.
- `topotein_lifter.py`
  - À terme seulement : on peut cesser de stocker `distance_encoding` (gain de
    place). **Ne pas** réutiliser `get_positional_encoding` pour les distances —
    elle reste pour le rang 0 ; créer une fonction RBF distincte si on déplace le
    calcul hors du modèle.
- Pas de re-lifting nécessaire si le calcul est fait dans le modèle (la distance
  brute est déjà présente dans les `.pt`).

### Impact
- **Checkpoints incompatibles** : la dimension d'entrée de `rank1_emb` change →
  ré-entraînement obligatoire (cohérent, le run actuel est de toute façon
  préliminaire).
- Aucune incidence sur les autres rangs ni sur le reste du pipeline.

### Validation / ablation
1. **Calibrer la plage** : échantillonner les distances d'arête sur un sous-ensemble
   de `.pt` et tracer l'histogramme pour fixer `d_min`, `d_max`, `K`, `σ`.
2. **Sanity** : vérifier qu'un petit MLP reconstruit `d` à partir de `rbf(d)`
   (l'encodage doit être informatif et inversible sur la plage).
3. **Utilisation des dimensions** : variance par dimension / rang effectif de
   l'encodage, à comparer au sinusoïdal (où ~la moitié sont mortes).
4. **Ablation à iso-réglages** : sinusoïdal vs RBF, même split, mêmes graines ;
   comparer TM-recall et HDBSCAN ARI, plus les health checks (collapse).

### Risques / notes
- Si la plage est mal calibrée, la RBF souffre du même défaut que le sinusoïdal —
  d'où l'étape 1 obligatoire.
- Tester `K=16` d'abord pour isoler l'effet de la base (à dimension égale) avant
  d'augmenter `K`.

### Statut
À faire.

---

## Modification 2 — Stratégie de paires positives (augmentation vs vraies paires TM)

### Remarque (superviseure)
Chercher dans la littérature si des travaux utilisent le contrastive learning sur
des protéines, et s'ils emploient le jittering de quelques ångström + masquage.
Crainte : (a) crée-t-on des protéines physiquement impossibles ? (b) perd-on en
précision par rapport à chercher de vraies paires à TM-score élevé pour
l'entraînement ?

### Réponse synthétique (à lui transmettre)
Sa crainte est fondée et documentée. Deux écoles coexistent dans la littérature :

1. **Positifs par augmentation** (vues bruitées de la même protéine). C'est ce que
   fait notre modèle (jitter + masquage), inspiré de SimCLR. Utilisé notamment par
   GearNet (crops de sous-séquence/sous-espace + masquage d'arêtes, **sans** bruit
   de coordonnées) et par Hermosilla & Ropinski (bruit gaussien sur coordonnées +
   rotations/crops).
2. **Positifs par vraie similarité structurale** (paires de protéines réellement
   proches). C'est exactement l'alternative qu'elle propose : ProtTucker tire ses
   triplets de la classification CATH (vraies classes de repliement) ; TM-Vec
   apprend à prédire directement le TM-score de vraies paires. Pas d'augmentation
   non physique, mais il faut précalculer la similarité structurale.

Sur la **plausibilité physique** : la littérature de débruitage (Zaidi et al. ;
Godwin et al., « Noisy Nodes ») justifie le bruit gaussien **seulement** pour de
petites amplitudes autour d'un minimum d'énergie (interprétation champ de forces) ;
au-delà, on quitte la variété physique (longueurs/angles de liaison, clashs). Un
travail récent (RigidSSL, 2026) argue explicitement que les perturbations naïves
ne sont pas réalistes et propose des perturbations « rigidity-aware ». Donc :
oui, un jitter de **plusieurs** ångström peut produire des structures non
physiques.

Nuance importante sur notre cas : notre jitter est de **0,3–0,5 Å** (sous-ångström,
ordre des fluctuations thermiques / incertitude de coordonnées), pas « plusieurs
ångström ». À cette amplitude on reste près de la variété physique ; le risque
qu'elle décrit augmente avec σ. Par ailleurs le masquage agit sur les *features*
(dropout de descripteurs), il ne déforme pas la géométrie — il ne crée donc pas de
structure non physique. Le vrai point faible passé était le **crop de SSE** (qui
faisait chuter le TM-score entre deux vues à 0,5–0,65, c.-à-d. des « positifs » pas
structurellement identiques) — déjà désactivé chez nous.

Sur la **précision** : il y a un vrai compromis, sans gagnant universel.
- Augmentation : auto-supervisé, bon marché, mais on apprend une invariance à des
  perturbations potentiellement non physiques.
- Vraies paires TM : positifs physiquement réels, mais (i) coût de précalcul de la
  similarité structurale (de l'ordre de O(N²) alignements), (ii) bruit d'étiquette
  selon le seuil de TM choisi, (iii) on supervise avec le signal même qu'on veut
  apprendre (risque de fuite). Notre pipeline injecte déjà de la vraie similarité
  via le terme auxiliaire de régression TM-score (+ cache TM) : on est donc
  partiellement dans l'approche « vraies paires ».

### État de l'art (synthèse)

| Travail | CL sur protéines ? | Augmentations | Positifs |
|---|---|---|---|
| GearNet (Zhang et al., ICLR 2023) | oui (Multiview Contrast) | crops sous-séquence/sous-espace + masquage d'arêtes ; **pas** de bruit de coordonnées | deux vues croppées de la même protéine |
| Hermosilla & Ropinski (2022) | oui | bruit gaussien sur coordonnées + rotations/crops | deux vues bruitées de la même protéine |
| ProtTucker (Heinzinger et al., 2022) | oui (triplet) | aucune (pas d'augmentation) | **vraies** classes CATH (repliement) |
| TM-Vec (Hamamsy et al., Nat. Biotech. 2024) | apparenté (prédiction TM) | aucune | **vraies** paires, cible = TM-score |
| Zaidi et al. (2023) / Noisy Nodes | débruitage (pas CL) | bruit gaussien **petit** autour de l'équilibre | n/a (justifie le bruit faible) |
| RigidSSL (2026) | pré-entraînement géométrique | perturbations **rigidity-aware** (anti bruit naïf) | n/a |

### Foldseek ou MMseqs2 pour construire les couples ?

Question de la superviseure : clusteriser les protéines très proches (Foldseek ou
MMseqs2) et utiliser ces clusters comme couples d'entraînement. Précisions :

- **MMseqs2** clusterise par **séquence**. Pour des protéines virales
  hyper-divergentes (Nomburg : 62 % structurellement distinctes, séquence encore
  pire), on obtiendrait surtout des singletons et, pour le reste, des positifs
  « faciles » (forte identité de séquence) — donc on raterait justement
  l'homologie structurale lointaine qui est tout l'enjeu. Peu adapté seul.
- **Foldseek** clusterise par **structure** (alphabet 3Di) : c'est l'outil
  pertinent pour notre but. Le pipeline standard à grande échelle (Barrio-Hernandez
  et al., *Nature* 2023, clustering de l'AFDB) fait d'ailleurs MMseqs2 (séquence)
  **puis** Foldseek (structure), car les deux capturent des choses différentes.

### Argument décisif pour NOTRE cas : fuite d'étiquettes
Nos étiquettes d'évaluation (les « familles ») **sont** des clusters structuraux
de type Foldseek (le dataset Nomburg est clusterisé structuralement, et
`filter_dataset.py` charge un TSV de clusters Foldseek). Donc construire les
positifs d'entraînement à partir de clusters Foldseek revient à **s'entraîner sur
les étiquettes de test** : la comparaison « 2,5× meilleur que Foldseek » devient
circulaire (on distille Foldseek), et l'ARI/NMI vs familles est gonflé. C'est
l'objection numéro un, propre à notre benchmark — sauf à changer l'évaluation pour
un benchmark structural indépendant (SCOP/CATH, ou un test TM-score non dérivé du
même clustering). À noter aussi : un cluster peut traverser le split phylo →
fuite train/val.

### A-t-on déjà fait ça ?
- **Dans la littérature** : oui. ProtTucker utilise les classes CATH (clusters
  structuraux) comme positifs ; l'AFDB est clusterisée par Foldseek
  (Barrio-Hernandez 2023) ; plusieurs méthodes contrastives récentes s'appuient
  sur Foldseek/MMseqs2 (ProTrek, etc.).
- **Dans notre code** : oui, et **très probablement dans le run préliminaire
  lui-même**. Par défaut `supervised = not --unsupervised` ; la commande utilisée
  n'avait pas `--unsupervised`, et `./data/cluster.tsv` existe → `acc_to_cluster`
  rempli → la perte bascule sur `supervised_ntxent_loss` (SupCon binaire, positifs
  = même cluster Foldseek) + auxiliaire TM-score. Donc la fuite d'étiquettes a
  vraisemblablement déjà eu lieu dans les résultats préliminaires (abandonnés). À
  confirmer via le log console (`supervised=True`) ou le `model_config` du
  checkpoint. Conséquence : pour le **nouveau** run, choisir explicitement
  `--unsupervised` si l'on veut une évaluation sans fuite vs Foldseek.

### Arguments pour / contre (synthèse)

| | Jitter + masquage (augmentation) | Couples par clusters Foldseek/MMseqs2 |
|---|---|---|
| **Pour** | pas de fuite (positifs intrinsèques, indépendants des familles) ; couvre les 40 % de singletons ; bon marché, auto-supervisé ; on peut honnêtement « battre Foldseek » | positifs réels et physiquement valides ; enseigne directement « homologues structuraux ⇒ proches » ; capte la vraie variation entre homologues ; Foldseek passe à l'échelle (coût faible sur 67k) |
| **Contre** | positifs synthétiques (invariance à des perturbations choisies, pas à la vraie variation) ; plausibilité physique limite si σ grand ; n'enseigne pas directement l'homologie lointaine | **fuite d'étiquettes** vs notre éval (rédhibitoire ici) ; singletons sans positif ; MMseqs2 ~inutile sur protéines virales ; seuil (E-value/TM) = bruit d'étiquette + risque de fuite train/val |

### Impact sur `protocol.tex`
Changements à envisager (non encore appliqués — à valider) :
1. Section augmentation / contrastive : ajouter l'argument **anti-fuite** comme
   justification explicite du choix d'augmentation (positifs intrinsèques ; les
   clusters Foldseek sont réservés à l'évaluation), et mentionner l'alternative
   « couples par clusters » avec la raison de ne pas l'utiliser comme positif dur.
2. Section *Baselines* : préciser que, comme on veut battre Foldseek, on s'interdit
   d'entraîner sur des positifs dérivés de Foldseek, sous peine de circularité.
3. Optionnel : noter que la vraie similarité structurale est injectée de façon
   **douce** (régression TM-score / soft-SupCon), pas comme définition dure des
   positifs.

### Modification envisageable
Trois options, par coût croissant :
1. **Statu quo + bornage** : garder l'augmentation mais documenter σ = 0,3–0,5 Å,
   et ajouter un garde-fou (vérifier que le TM-score entre deux vues reste très
   élevé, p. ex. > 0,9 ; rejeter sinon). Faible coût.
2. **Hybride** : positifs = augmentation **et** voisins structuraux réels
   (TM-score élevé via le cache TM déjà présent), pondérés. Coût moyen.
3. **Vraies paires** : remplacer l'augmentation par des positifs à TM-score élevé
   (style ProtTucker/TM-Vec). N'est défendable **que** si l'évaluation passe sur un
   benchmark structural indépendant (sinon fuite, cf. argument décisif).

### Validation / ablation
- Mesurer la distribution du **TM-score entre les deux vues augmentées** (jitter +
  masquage) sur un échantillon : si elle reste > 0,9, la crainte « non physique »
  est empiriquement faible à notre σ.
- Ablation des trois options ci-dessus à iso-réglages : TM-recall, HDBSCAN ARI,
  health checks (collapse).

### Statut
À arbitrer (décision de design à prendre avec les superviseur·e·s).

### Références
- Zhang et al., *Protein Representation Learning by Geometric Structure
  Pretraining* (GearNet), ICLR 2023. arXiv:2203.06125.
- Hermosilla, Ropinski, *Contrastive Representation Learning for 3D Protein
  Structures*, 2022. arXiv:2205.15675.
- Heinzinger et al., *Contrastive learning on protein embeddings enlightens
  midnight zone* (ProtTucker), NAR Genomics and Bioinformatics, 2022.
- Hamamsy et al., *Protein remote homology detection and structural alignment
  using deep learning* (TM-Vec), Nature Biotechnology, 2024.
- Zaidi et al., *Pre-training via Denoising for Molecular Property Prediction*,
  ICLR 2023. arXiv:2206.00133. (et Godwin et al., *Noisy Nodes*, 2022.)
- *Rigidity-Aware Geometric Pretraining for Protein Design and Conformational
  Ensembles* (RigidSSL), 2026. arXiv:2603.02406.
- Barrio-Hernandez et al., *Clustering predicted structures at the scale of the
  known protein universe*, Nature 622:637–645, 2023 (clustering AFDB :
  MMseqs2 puis Foldseek).

---

## Modification 3 — Témoin ARI par permutation des clusters

### Remarque (superviseure)
Construire un témoin pour calibrer l'ARI : prendre le clustering Foldseek, garder
le **nombre** et les **tailles** de clusters mais **mélanger** l'appartenance des
protéines ; utiliser ces faux clusters pour créer les paires positives (un SupCon
dont les positifs ne correspondent plus au vrai clustering Foldseek). L'ARI obtenu
(évalué contre les **vrais** clusters Foldseek) devient la baseline.

### Verdict : pertinent, à faire — avec des garde-fous
C'est un **contrôle par permutation d'étiquettes** classique et bien pensé : il
préserve les statistiques marginales (nombre et tailles de clusters) auxquelles
l'ARI et HDBSCAN sont sensibles, ce qui en fait un null plus honnête qu'un simple
« embedding aléatoire ». Il isole la contribution du **vrai** signal d'étiquette :
la quantité interprétable est `ARI(SupCon réel) − ARI(SupCon permuté)`.

### Ce qu'il teste vraiment (et ses limites)
- C'est le bon null **pour un modèle SupCon** (supervisé). Or notre modèle « cible »
  devrait être **non supervisé** (sans fuite) : pour celui-là, la permutation n'est
  pas applicable (il n'utilise pas d'étiquettes), et les bons témoins restent
  l'embedding aléatoire, le modèle non entraîné, et Foldseek.
- Subtilité : même avec des étiquettes mélangées, **les deux vues augmentées d'une
  même protéine partagent toujours la même étiquette** → il reste un vrai positif
  auto-supervisé. Le témoin n'est donc pas un null pur « zéro signal » mais
  « augmentation + étiquettes aléatoires ». À interpréter comme tel ; le complément
  utile est de mesurer aussi le modèle **augmentation seule** (non supervisé).
- Le témoin **détecte** une fuite/artefact, il ne la **corrige pas** : si le SupCon
  réel bat largement le SupCon permuté, cela prouve surtout que le modèle a appris
  les étiquettes Foldseek — c'est-à-dire exactement la circularité. Donc ce témoin
  est surtout précieux pour **chiffrer** la fuite, pas pour valider un apprentissage
  structural réel.

### À faire correctement
- Mélanger les étiquettes **uniquement dans le split d'entraînement** ; toujours
  évaluer l'ARI contre les **vrais** clusters Foldseek (et côté validation).
- Répéter sur **plusieurs permutations** (graines) pour obtenir une **distribution**
  nulle et un intervalle, pas un seul nombre.
- Logger les health checks (collapse) : un SupCon sur étiquettes aléatoires peut
  s'effondrer, ce qui rend l'ARI HDBSCAN instable — à interpréter avec prudence.
- Le garder **en plus** des autres baselines (embedding aléatoire ≈ 0, modèle non
  entraîné, Foldseek), pas à leur place.

### Implémentation
- Réutiliser le chemin SupCon existant (`supervised_ntxent_loss`) en remplaçant
  `acc_to_cluster` par une version **permutée** : conserver l'histogramme des
  tailles de clusters et réaffecter aléatoirement les protéines du split train.
- Petit utilitaire : lire `data/cluster.tsv`, permuter les membres en gardant les
  tailles, écrire un `cluster_shuffled.tsv`, pointer la run témoin dessus.
- Tout le reste identique au run réel (même dataset filtré 3–4k, même split phylo,
  même archi, mêmes graines hors permutation).

### Impact sur `protocol.tex`
Dans la section *Baselines* : ajouter ce témoin de permutation comme baseline
calibrée pour l'ARI, à côté de (embedding aléatoire, modèle non entraîné, Foldseek).
Préciser la lecture : `ARI_réel − ARI_permuté` mesure l'apport des vraies
étiquettes, et pour le modèle non supervisé ce sont les autres témoins qui priment.

### Statut
À faire (pour le nouveau run sur la base filtrée).

### Référence
- ARI ajusté du hasard : Hubert & Arabie, *Comparing partitions*, Journal of
  Classification, 1985 (l'ARI corrige déjà les marginaux ; d'où l'intérêt d'un
  témoin qui teste le **pipeline**, pas seulement le hasard).

---

## Modification 4 — Métriques directionnelles d'accord avec Foldseek

### Remarque (superviseure)
En plus de l'ARI : (1) une « sorte d'ARI » = % des protéines co-clusterisées par
Foldseek qui se retrouvent dans le même cluster appris ; (2) un degré
d'atomisation = en combien de clusters appris éclate en moyenne un cluster
Foldseek (et l'inverse : fusionne-t-on plusieurs clusters Foldseek ?).

### Verdict : très pertinent
L'ARI est un résumé **symétrique** : il ne dit pas dans quel sens on se trompe.
Ces deux métriques décomposent les deux modes d'échec opposés que l'ARI fusionne
(sur-découpage vs sur-fusion). C'est un vrai gain de diagnostic.

### Correspondance avec des métriques standard
- **Métrique 1** (% co-clusterisés Foldseek restant ensemble) = **complétude /
  rappel de paires** (`TP/(TP+FN)`). Noms standard : *completeness*
  (Rosenberg & Hirschberg), *rappel BCubed*. Version sans clustering déjà présente :
  le **TM-recall** (fraction des vrais homologues parmi les k plus proches voisins).
- **Métrique 2 (fragmentation)** = en moyenne, nb de clusters appris par cluster
  Foldseek (idéal = 1). Analogue entropique : `H(appris | Foldseek)`.
- **Inverse (fusion)** = nb de clusters Foldseek par cluster appris (idéal = 1) =
  **homogénéité / précision**. Analogue : `H(Foldseek | appris)`.
- Couple homogénéité + complétude → moyenne harmonique = **V-measure**.
- Couple précision + rappel de paires → moyenne géométrique = **Fowlkes–Mallows**.
- Disponibles dans `sklearn.metrics` (`homogeneity_score`, `completeness_score`,
  `v_measure_score`, `fowlkes_mallows_score`).

### Garde-fous (sinon trompeuses)
- **Toujours par paires, jamais seules.** Chacune est triviale à maximiser : tout
  dans un cluster → complétude 100 %, homogénéité 0 ; tout en singletons → l'inverse.
- **Même fuite que l'ARI** : ce sont des accords avec Foldseek ; si entraînement
  SupCon sur étiquettes Foldseek, toutes gonflées. Elles diagnostiquent la *forme*
  de l'accord, pas s'il est mérité.
- **Dépendance algo/k** : la fragmentation/fusion dépend fortement de k (k-means)
  ou des paramètres HDBSCAN. Fixer k = nb de clusters Foldseek pour une lecture
  propre, ou rapporter la distribution.
- **Singletons** (~40 %) : un singleton Foldseek est trivialement « complet ».
  Exclure les singletons ou rapporter une version multi-membres uniquement, sinon
  les métriques sont noyées par les cas triviaux. Penser à pondérer par la taille.

### Implémentation
- Réutiliser les embeddings + le clustering déjà calculés dans `epoch_eval.py`.
- Ajouter : `homogeneity/completeness/v_measure/fowlkes_mallows` (sklearn) ; plus
  deux indices maison = moyenne (pondérée par taille) du nb de clusters appris par
  cluster Foldseek (fragmentation) et l'inverse (fusion), sur clusters multi-membres.
- Rapporter sur le split validation, contre les vrais clusters Foldseek, à k fixé.

### Impact sur `protocol.tex`
Section *Metrics* : ajouter homogénéité + complétude + V-measure à côté de l'ARI et
TM-ρ, et les indices de fragmentation/fusion. Mentionner explicitement la lecture
par paires et la mise en garde fuite/algorithme.

### Statut
À faire.

### Références
- Rosenberg, Hirschberg, *V-Measure: A Conditional Entropy-Based External Cluster
  Evaluation Measure*, EMNLP 2007 (homogénéité, complétude, V-measure).
- Fowlkes, Mallows, *A Method for Comparing Two Hierarchical Clusterings*, JASA 1983.
- Amigó et al., *A comparison of extrinsic clustering evaluation metrics based on
  formal constraints*, Information Retrieval 2009 (BCubed précision/rappel).

---

## Modification 5 — Nombre de clusters produits (métrique descriptive)

### Remarque (superviseure)
Ajouter comme métrique le nombre de clusters que le modèle produit.

### Verdict : ton intuition est juste — faible comme métrique de qualité
Le nombre de clusters dépend essentiellement des paramètres HDBSCAN
(`min_cluster_size`, `min_samples`, `epsilon`) : on peut obtenir presque n'importe
quel nombre en réglant. Donc **pas une métrique de qualité en soi**. Deux nuances
qui le rendent quand même utile, mais comme **statistique descriptive / diagnostic** :

- Utile **relativement** au nombre de clusters Foldseek (la référence) : ~2900 chez
  Foldseek vs 200 ou 20000 chez nous = signal de sur-fusion ou sur-fragmentation.
  C'est la version grossière des métriques directionnelles (Mod 4).
- Utile comme **health check** : un modèle effondré donne un compte dégénéré (un
  géant + tout en bruit, ou une myriade de micro-clusters). Sa **stabilité** (à
  travers paramètres/graines) est plus informative que sa valeur brute.
- Déjà loggé : `n_clusters` et `singleton_frac` dans `epoch_eval.py`.
- Sous k-means à k fixé, ce n'est même pas une variable libre.

### À faire correctement
Rapporter à **paramètres HDBSCAN fixés**, **relativement** au compte Foldseek, et
avec la **distribution des tailles** (pas le seul compte). Suivre sa stabilité.

### Impact sur `protocol.tex`
Le présenter comme statistique descriptive (relatif à Foldseek, params fixés, +
distribution des tailles), **pas** comme métrique de qualité.

### Statut
Déjà loggé ; à présenter correctement.

---

## Modification 6 — Export de clusters pour validation manuelle

### Remarque (superviseure)
Après la run, lui envoyer quelques clusters préparés en fichiers FASTA (séquences)
pour les regarder concrètement dans des logiciels d'alignement.

### Verdict : excellent — mais attention au piège séquence vs structure
La validation qualitative « est-ce que les clusters ont un sens biologique » est
indispensable et complète le quantitatif. Mais un **alignement de séquence peut
induire en erreur ici** : tout le projet repose sur l'idée que ces protéines sont
proches en **structure** malgré une **séquence** divergente. Un vrai cluster
d'homologie structurale lointaine montrera souvent une **faible identité de
séquence** — ce n'est pas un défaut, c'est le résultat attendu. Si la superviseure
juge un cluster « mauvais » parce que les séquences ne s'alignent pas, la
conclusion serait fausse. Donc : valider en **structure** (Foldseek / TM-align /
DALI, superposition PyMOL/ChimeraX), pas seulement en séquence.

### Recommandations
- **Fournir aussi les structures** (PDB) par cluster, pas seulement le FASTA, pour
  permettre une superposition structurale. Optionnel : précalcul TM-align
  intra-cluster (matrice de TM-scores par cluster).
- **Sélection principielle** (éviter le cherry-picking) : quelques clusters cœur
  (gros, haute confiance) ; quelques clusters « pont » qui fusionnent plusieurs
  familles Foldseek (les revendications nouvelles, à scruter en priorité) ;
  quelques clusters de **désaccord** avec Foldseek (split/merge) ; plus un petit
  échantillon aléatoire, par honnêteté.

### Implémentation
- Réutiliser les sorties de clustering (`epoch_eval.py` / `parse_clusters.py`) +
  extraire la séquence depuis les PDB (mapping `AA_3_TO_1` déjà dans le lifter) →
  un `.fasta` par cluster, plus un dossier de PDB par cluster.
- À lancer **après** le nouveau run (sur la base filtrée). Le script d'export est à
  écrire (je peux le faire quand tu veux).

### Statut
À faire (après run).

---

## Backlog — remarques à trier

À compléter ensemble au fil de la revue des remarques de la superviseure. Pistes
déjà entrevues qui pourraient devenir des modifications :

- Vecteur déplacement `c_j - c_i` : dépendance au repère (pas de repère canonique)
  → question d'équivariance/invariance SE(3) si on l'utilise un jour.
- Choix de `k = 16` pour le graphe de contacts (vs rayon de coupure, vs autre k).
- (autres à ajouter)

## Journal

- _(date)_ — Création du plan ; ajout de la modification 1 (encodage de distance).
- _(date)_ — Ajout de la modification 2 (stratégie de paires positives) + revue de
  littérature sur le contrastive learning protéique et le bruit de coordonnées.
- _(date)_ — Modification 2 enrichie : Foldseek vs MMseqs2, argument de fuite
  d'étiquettes, état du code (SupCon déjà présent), impact `protocol.tex`.
- _(date)_ — Constat : le run préliminaire a très probablement utilisé SupCon
  (positifs = cluster Foldseek) → fuite ; résultats préliminaires abandonnés.
  Nouveau run prévu sur la base filtrée (~3–4k protéines). Ajout de la
  modification 3 (témoin ARI par permutation des clusters).
- _(date)_ — Ajout de la modification 4 (métriques directionnelles d'accord avec
  Foldseek : homogénéité/complétude/V-measure, fragmentation/fusion).
- _(date)_ — Ajout des modifications 5 (nombre de clusters, descriptif ; déjà
  loggé) et 6 (export FASTA + structures pour validation manuelle ; mise en garde
  séquence vs structure).
