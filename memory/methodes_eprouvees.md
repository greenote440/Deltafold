# Méthodes éprouvées — objectif contrastif & création de paires

Synthèse de lecture (références `[CRITIQUE]`/objectif/paires de `revue_litterature.md`),
pour éclairer le choix **avant** de proposer une solution. Constat de départ : notre
échec vient de positifs triviaux (jitter+masquage sur la protéine entière → tâche
résolue à loss ~1e-5, le modèle apprend une invariance au bruit inutile).

## A. Création des paires (positifs) — le levier le plus important

Leçon convergente : **les méthodes protéiques qui marchent fabriquent les positifs
par crop / sous-structure / vraie similarité, jamais par bruit de coordonnées sur la
protéine entière.**

1. **Sous-structures de la même protéine** — Hermosilla & Ropinski 2022
   (arXiv:2205.15675). Positifs = deux sous-structures aléatoires **distinctes** de la
   même chaîne (transformation préservant la localité → sous-structures connexes) ;
   négatifs = mini-batch. Tâche non triviale (reconnaître deux régions du même fold).
   SOTA en fold classification, fonction, similarité, binding — **même après
   déduplication des protéines proches du test set**. → répond à la fois à notre échec
   (trivialité) et à la fuite.
2. **Crops sous-séquence / sous-espace + masquage d'arêtes** — GearNet « Multiview
   Contrast », ICLR 2023 (arXiv:2203.06125). Positifs = deux crops de la même protéine
   (sous-séquence ≤ 50 résidus, ou sous-espace spatial), puis masquage d'arêtes.
   Éprouvé (pré-entraînement structural SOTA). Converge avec (1) : **crop, pas jitter**.
3. **Vraies paires par similarité structurale** (positifs = protéines réellement
   proches, pas la même protéine) :
   - **ProtTucker** (Heinzinger 2022) — triplets depuis les classes **CATH** (vrais
     folds). Éprouvé en homologie lointaine.
   - **TM-Vec** (Hamamsy, Nat. Biotechnol. 2024) — cible = **TM-score réel** de vraies
     paires (réseaux jumeaux). Éprouvé.
   - Xia et al. (cité par H&R) — labels de similarité **TM-align**.
   → fidèle mais **fuite** si on évalue ensuite contre la structure (cf. nos échanges).
4. **Ce qu'on fait** (jitter 0.3–0.5 Å + masquage de features sur la protéine entière)
   **n'appartient pas** au répertoire éprouvé du CL structural protéique. Le bruit de
   coordonnées vient du monde moléculaire/**denoising** (cf. B.3), pas du CL de fold.

## B. Objectif (la fonction de perte)

1. **InfoNCE / NT-Xent** — SimCLR (2002.05709), MoCo (1911.05722). Éprouvé, mais :
   gourmand en négatifs (grand batch, ou **file de négatifs MoCo** — utile quand le
   batch est petit comme chez nous), **tête de projection** quasi obligatoire, et
   **sensible au collapse** si la tâche est triviale (notre cas).
2. **Sans négatifs, anti-collapse explicite** (recommandés quand peu de données/négatifs)
   - **VICReg** (2105.04906) — termes **variance** (empêche le collapse dimensionnel) +
     **invariance** + **covariance** (décorrèle les dimensions). Pas de grands batches,
     pas de négatifs, pas de stop-grad. Éprouvé.
   - **Barlow Twins** (2103.03230) — rend la **matrice de cross-corrélation** des deux
     vues ≈ identité (invariance + réduction de redondance). Pas de négatifs, robuste,
     passe à l'échelle. Éprouvé.
   → ces deux **évitent par construction** le collapse observé sur nos runs hard-neg.
3. **Débruitage / prédictif (non contrastif, sans paires ni négatifs, sans fuite)**
   - **Pre-training via Denoising** (Zaidi 2206.00133) — prédire le bruit ≈ apprendre
     un champ de forces ; petit σ autour de l'équilibre. Éprouvé (propriétés moléc.).
   - **GeoSSL — SE(3)-Invariant Denoising Distance Matching** (2206.13602). Éprouvé.
   → signal physique, leakage-free ; alternative crédible au contrastif.
4. **Garde-fous éprouvés**
   - **Tête de projection** (dimcol, 2110.09348) — prévient le collapse dimensionnel.
   - **Alignement & uniformité** (Wang & Isola, 2005.10242) — métriques de diagnostic.
   - **Débiaisage des faux négatifs** (Chuang, 2007.00224) ; **négatifs durs avec
     prudence** (Robinson, 2010.04592) — pertinents vu que notre hard-neg crée des faux
     négatifs.

## Ce que ça implique pour nous (sans encore trancher)

- Notre échec (positifs triviaux) est **exactement** ce que résout la création de
  paires par **sous-structure (H&R)** ou **crop (GearNet)**.
- Deux familles éprouvées et leakage-free à considérer :
  1. **Contrastif réparé** : positifs par sous-structure/crop + objectif anti-collapse
     (VICReg ou Barlow) + retrait du hard-neg + tête de projection.
  2. **Non contrastif** : denoising / GeoSSL (pas de paires, signal physique).
- Les « vraies paires TM/CATH » restent une option forte mais à manier pour la fuite.

## Références (liens)
- Hermosilla, Ropinski, *Contrastive Representation Learning for 3D Protein Structures*,
  2022 — https://arxiv.org/abs/2205.15675
- Zhang et al., *GearNet / Multiview Contrast*, ICLR 2023 — https://arxiv.org/abs/2203.06125
- Heinzinger et al., *ProtTucker*, NAR Genom. Bioinform. 2022 — doi:10.1093/nargab/lqac043
- Hamamsy et al., *TM-Vec*, Nat. Biotechnol. 2024 — https://www.nature.com/articles/s41587-023-01917-2
- Bardes, Ponce, LeCun, *VICReg*, ICLR 2022 — https://arxiv.org/abs/2105.04906
- Zbontar et al., *Barlow Twins*, ICML 2021 — https://arxiv.org/abs/2103.03230
- Chen et al., *SimCLR*, 2020 — https://arxiv.org/abs/2002.05709 ; He et al., *MoCo*, 2020 — https://arxiv.org/abs/1911.05722
- Jing et al., *Dimensional Collapse*, ICLR 2022 — https://arxiv.org/abs/2110.09348
- Zaidi et al., *Pre-training via Denoising*, ICLR 2023 — https://arxiv.org/abs/2206.00133
- Liu et al., *GeoSSL denoising distance matching*, 2022 — https://arxiv.org/abs/2206.13602
- Chuang et al., *Debiased CL*, 2020 — https://arxiv.org/abs/2007.00224 ; Robinson et al., *Hard Negatives*, 2021 — https://arxiv.org/abs/2010.04592
- Wang, Isola, *Alignment & Uniformity*, 2020 — https://arxiv.org/abs/2005.10242
