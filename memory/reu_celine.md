
Que représente le pLDDT ? Peu de protéines à pLDDT élevé → un filtrage par pLDDT pourrait réduire le temps de calcul.
Assignation de structure secondaire via biotite / PSEA.
Le choix « canonique » dans la construction du vecteur associé à la distance (et l'ordre croissant des centres).
Clarifier et faire un schéma ; expliciter le choix de construction des protéines « similaires » ; chercher des articles utilisant ce type d'approche.

Discuter le clustering Foldseek dans la construction des paires.
Différence entre l'augmentation des paires et des paires positives issues des clusters Foldseek.
Foldseek ou MMseqs2 pour regrouper les protéines très proches et bâtir les couples d'entraînement ?
Témoin : faire un clustering aléatoire en conservant les mêmes tailles de clusters, puis construire les paires à partir de là.

Un ARI directionnel : le pourcentage de protéines co-clusterisées par Foldseek qui se retrouvent dans le même cluster appris (dans un seul sens).
Degré d'atomisation (sensibilité) : chaque cluster Foldseek est éclaté dans combien de clusters du modèle ?
Le nombre de clusters produits par le modèle.
Éventuellement contraindre le nombre de clusters.

Validation du clustering : préparer quelques clusters en fichiers FASTA pour les logiciels d'alignement.
Semaine prochaine : un plan d'expérimentation.

Mes réponses

Création des paires (arXiv:2205.15675). Au lieu du jitter, on donne à l'encodeur deux sous-structures connexes de la même protéine (≈ 40 %/60 %) et on force leurs représentations à être proches ; les paires négatives sont formées aléatoirement, sous l'hypothèse qu'échantillonner deux protéines réellement proches n'arrive quasiment jamais.
Pourquoi ne pas utiliser Foldseek pour construire les augmentations. Foldseek est notre métrique de validation (on compare nos clusters aux siens) ; on ne doit donc pas l'utiliser en entrée de l'entraînement, sous peine de circularité.
Sur le témoin « clusters Foldseek mélangés ». C'est un bon témoin pour un entraînement supervisé (SupCon), mais pas pour l'apprentissage non supervisé.
Sur l'export FASTA. Tout l'objectif est de trouver des homologues lointains, à faible identité de séquence — un alignement de séquences ne risque-t-il pas d'être trompeur ? Quel intérêt par rapport à une superposition structurale (PyMOL / ChimeraX, TM-align / Foldseek)


Maintenant qu'il n'y a plus de fuite du TM-score dans les données d'apprentissage, les résultats sont moins bons. Comment distinguer un problème de pipeline d'apprentissage d'un problème d'expressivité du modèle ?

ssh -X pnardi@deltafold.univ-lyon1.fr
Rednote440

