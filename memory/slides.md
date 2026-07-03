# Slide 1: Introduction and Contextualization
Bonjour à tous. Aujourd'hui, je vous propose d'explorer un cadre théorique et computationnel qui va être structurant pour la suite de nos travaux : le Deep Learning Topologique (TDL).
Cette présentation s'appuie sur une synthèse rigoureuse de l'article de Mathilde Papillon et ses collaborateurs, intitulé “Architectures of Topological Deep Learning: A Survey of Message-Passing Topological Neural Networks”.

Les graphes se limitent par définition à des relations purement binaires, de noeud à noeud. Or, en biologie structurale, qu’il s’agisse d'interactions électrostatiques multi-corps ou d’agencements de structures secondaires (hélices α, feuillets β), les relations sont intrinsèquement d'ordre supérieur.

Ce papier pose les fondations d'un langage mathématique et graphique unifié pour toutes les architectures de réseaux de neurones topologiques (TNN) à passage de messages. L'objectif d'aujourd'hui est de décortiquer ces concepts clés — domaines, voisinages, équivariance et passage de messages — pour voir comment les appliquer concrètement à nos problématiques de classification structurale et de repliement.

# Slide 2 : Principe Général des TNN
Passons maintenant au principe général de fonctionnement d'un réseau de neurones topologique, ou TNN, schématisé sur cette diapositive. Le processus se décompose en trois grandes étapes.

Tout d'abord, nous partons du domaine de données (Data domain). Dans notre contexte, il s'agit d'une protéine modélisée initialement sous forme de graphe, où les nœuds représentent les résidus d'acides aminés et les arêtes codent les liaisons ou la proximité géométrique.
La deuxième étape est le prétraitement (Preprocessing), également appelé étape de surélévation ou de lifting. C'est ici que réside la rupture conceptuelle avec les approches classiques : nous transformons ce graphe brut en un domaine computationnel (Computational domain). Ce domaine ne se contente plus de lier les résidus deux à deux ; il construit explicitement des structures géométriques et topologiques d'ordre supérieur — comme des faces ou des cavités — capables d'encoder des voisinages complexes et des interactions multi-corps.

Enfin, la troisième étape est le cœur du modèle : le réseau de neurones topologique à couches. Ici, l'exemple montre une architecture à 3 couches. À chaque niveau, le réseau utilise le mécanisme de passage de messages (Message passing) pour faire circuler l'information non seulement entre nœuds adjacents, mais aussi verticalement entre les différents rangs topologiques (des nœuds vers les arêtes, des arêtes vers les faces, et inversement).

Au fil des couches, les caractéristiques (features) de chaque entité topologique sont successivement mises à jour. Ce calcul itératif permet d'extraire une représentation latente globale très riche pour aboutir à la prédiction finale, qu'il s'agisse d'une classification fonctionnelle ou de la prédiction d'une propriété structurale.


# Slide 3 : Les domaines du Deep Learning Topologique (TDL)
Entrons maintenant dans le vif du sujet avec la taxonomie des différents domaines de données exploitables en TDL, illustrée par ce schéma. Pour bien comprendre l’apport de la topologie, il est utile de repartir des domaines discrets traditionnels.

À gauche, vous retrouvez les structures classiques : les Ensembles (Sets), caractérisés par une absence totale de relations explicites entre les entités, et les Graphes (Graphs), qui modélisent uniquement des relations binaires, de pair à pair. Comme nous l'avons évoqué, un graphe associe des nœuds (en bleu) via des arêtes (en rose).

Le Deep Learning Topologique, quant à lui, introduit quatre grandes familles de domaines computationnels pour capturer des relations d'ordre supérieur :

Les complexes simpliciaux (Simplicial complexes) : Ils étendent les graphes en ajoutant des "faces" pleines (en rouge). Une règle stricte régit ce domaine : pour qu'une face ou un triangle supérieur existe, toutes ses sous-parties (ses arêtes et ses nœuds) doivent obligatoirement être présentes dans le domaine. C'est une relation de type "Partie-Tout" (Part-Whole).

Les complexes cellulaires (Cellular complexes) : Ils sont plus flexibles. Contrairement aux triangles rigides des complexes simpliciaux, les faces d'un complexe cellulaire peuvent être des polygones fermés arbitraires à n côtés (comme un cycle de carbone ou un anneau aromatique). Ils obéissent également à cette contrainte "Partie-Tout" où les frontières de la cellule doivent exister.

Les hypergraphes (Hypergraphs) : Ici, nous basculons sur des relations de type "Ensemble" (Set-Type). Une hyperarête peut lier un nombre arbitraire de nœuds simultanément. En revanche, la notion de hiérarchie s'efface : l'existence d'une hyperarête reliant quatre nœuds n'implique pas nécessairement la présence d'hyperarêtes reliant des sous-groupes de ces nœuds.

Les complexes combinatoires (Combinatorial complexes) : C'est l'extension la plus générale et la plus puissante. Ils combinent la flexibilité des hypergraphes et la structure hiérarchique des complexes cellulaires. Les cellules peuvent y avoir une taille arbitraire et, point crucial, une cellule de rang supérieur peut contenir des entités de n'importe quel rang inférieur, sans restriction de frontière stricte.

# Slide 4 : Terminologie unifiée des structures d'ordre supérieur

L'article introduit un effort de clarification conceptuelle essentiel : une terminologie unifiée permettant de comparer rigoureusement ces quatre domaines topologiques.

Le concept fondamental ici est celui de cellule (Cell). Une cellule est définie par deux caractéristiques majeures :

- Son rang (rank), ou sa dimensionnalité.
- Sa taille (size), qui correspond au nombre de sous-cellules qu'elle contient.

Le papier montre alors comment chaque domaine restreint mathématiquement ces deux notions :

Dans un complexe simplicial, les règles sont les plus strictes : une cellule de rang n (comme un triangle plein, de rang 2) possède obligatoirement une taille fixe de n+1 cellules de rang n−1 (c'est-à-dire ses 3 arêtes limitrophes). C'est ce qui garantit la structure purement simpliciale.

Le complexe cellulaire assouplit cette contrainte : une cellule de rang n doit contenir au moins n+1 cellules de rang inférieur. Une face de rang 2 peut ainsi être un triangle, un carré ou un polygone à 12 côtés, ce qui est idéal pour modéliser des cycles chimiques.
L'hypergraphe, quant à lui, aplatit la hiérarchie. Il ne possède par définition que des cellules de rang 0 (les nœuds) et de rang 1 (les hyperarêtes). En revanche, la taille de ces hyperarêtes est totalement arbitraire : une seule hyperarête peut englober 3, 5 ou 50 nœuds à la fois.

Enfin, le complexe combinatoire lève toutes les barrières. Les cellules ont une taille arbitraire et, surtout, une cellule de rang n peut directement contenir et lier des entités de n'importe quel rang strictement inférieur (par exemple, lier directement un nœud de rang 0 à une face de rang 2), sans passer par l'intermédiaire obligatoire des arêtes.

# Slide 5 : Slide 5 : Exemples de domaines appliqués aux molécules
« Pour rendre ces concepts mathématiques très concrets, cette cinquième diapositive illustre comment une même structure chimique peut être encodée différemment selon le domaine topologique choisi. L'article utilise ici l'exemple du cyclopropane d'une part, et d'un acide aminé, la phénylalanine, d'autre part.

Regardons d'abord le cyclopropane, qui forme un anneau à trois carbones :

En complexe simplicial (en haut à gauche, figure a), l'anneau est naturellement modélisé par une face triangulaire pleine (en rouge). C'est parfait car la face possède exactement 3 arêtes sous-jacentes (n+1 pour le rang n).

En complexe cellulaire (figure b), la représentation est identique au complexe simplicial pour cette molécule spécifique, car le cycle comporte trois membres, respectant la contrainte des sous-cellules de frontière.

La distinction devient flagrante lorsque l'on passe à une molécule plus complexe comme la phénylalanine, qui possède un cycle aromatique à 6 carbones (un noyau benzénique) :

Si on voulait utiliser un complexe simplicial, on serait obligé de "trianguler" artificiellement ce cycle à 6 éléments en y insérant des arêtes internes fictives pour créer des faces triangulaires. Cela détruirait la symétrie réelle de la molécule.

En complexe cellulaire (en haut à droite, figure d), le problème disparaît : le cycle à 6 carbones est modélisé par une unique cellule polygonale à 6 côtés (en rouge). C'est une représentation chimiquement et topologiquement fidèle.

Voyons maintenant les approches basées sur les ensembles, en bas de la diapositive :

L'hypergraphe (figure f) permet d'englober tout le groupe aromatique de la phénylalanine dans une seule et unique hyperarête (la zone rose). C'est très utile pour capturer une propriété globale de ce groupe d'atomes, mais au détriment de la hiérarchie géométrique fine du cycle.

Enfin, le complexe combinatoire (en bas à droite) offre le meilleur des deux mondes pour l'acide aminé complet. Il permet de définir une cellule globale pour l'acide aminé (représentée par la grande enveloppe rouge), qui encapsule à la fois les structures locales, les sous-groupes chimiques et les liaisons, sans aucune contrainte de géométrie rigide.

### **Slide 6 : Application aux protéines – Le choix du Complexe Combinatoire (CC)**

[cite_start]« Après avoir passé en revue les différents domaines topologiques, passons à l'application concrète qui nous intéresse au plus haut point pour notre projet de recherche : la modélisation des protéines[cite: 86]. [cite_start]Pour cela, nous allons nous appuyer sur un papier récent et majeur intitulé *“Topotein: Topological Deep Learning for Protein Representation Learning”*[cite: 94]. 

[cite_start]Le constat de départ de Topotein est simple : pour capturer fidèlement la hiérarchie biologique d'une protéine, le **Complexe Combinatoire (CC)** est le domaine le plus adapté, surpassant les graphes ou les complexes simpliciaux[cite: 86]. [cite_start]Regardons le schéma de gauche pour comprendre comment une protéine est ici transposée en un Complexe Combinatoire de Protéine (PCC)[cite: 92]:

* [cite_start]**Le Rang 0 (0-cells)** représente les acides aminés, matérialisés par les nœuds noirs[cite: 92].
* [cite_start]**Le Rang 1 (1-cells)** code les interactions physiques et chimiques locales ou de contact entre ces acides aminés[cite: 92].
* [cite_start]**Le Rang 2 (2-cells)** modélise les **Structures Secondaires** (ou SSE pour *Secondary Structure Elements*), comme les hélices $\alpha$ et les feuillets $\beta$, représentées ici par les larges flèches bleues[cite: 92].
* [cite_start]**Le Rang 3 (3-cell)** englobe enfin la protéine entière dans sa globalité, représentée par la boîte en pointillés[cite: 92].

Quelle est la rupture algorithmique majeure apportée par cette modélisation ? Regardez le schéma de droite. Dans un GNN classique, l'information ne voyage qu'à travers le squelette linéaire de la protéine. En TDL combinatoire, Topotein introduit deux types d'arêtes pour enrichir drastiquement le passage de messages :
1. [cite_start]**Les arêtes internes** (*inner edges*, en violet), qui lient directement les acides aminés aux structures secondaires de rang 2 auxquelles ils appartiennent[cite: 93].
2. [cite_start]**Les arêtes externes** (*outer edges*, en rouge), qui connectent directement les structures secondaires (SSE) entre elles dans l'espace[cite: 93]. 

Cela permet un passage de messages à haute granularité : l'information peut circuler de manière ultra-rapide et pertinente entre deux feuillets $\beta$ distants dans la séquence mais proches dans l'espace tridimensionnel, sans souffrir des phénomènes d'atténuation (*oversmoothing*) propres aux graphes classiques. »

---

### **Slide 7 : Le framework Topotein – Représentation hiérarchique des caractéristiques**

« Maintenant que la structure du domaine computationnel est posée, comment valorise-t-on concrètement l'information biologique à chaque niveau de la hiérarchie ? [cite_start]C'est ce que détaille cette matrice de featurisation de Topotein[cite: 95]. [cite_start]Contrairement aux approches classiques qui n'injectent des caractéristiques qu'au niveau des résidus, Topotein enrichit mathématiquement chaque rang topologique[cite: 96]:

* [cite_start]**Aux Rangs 0 et 1 (Nœuds et Arêtes du squelette) :** On extrait les propriétés physico-chimiques fondamentales de la chaîne carbonée[cite: 96]. [cite_start]On y injecte un encodage *one-hot* à 23 dimensions de l'acide aminé, un alphabet structurel à 21 dimensions (le code *3Di* issu de *Foldseek*), ainsi que 16 dimensions de descripteurs géométriques purs (les angles de torsion de la chaîne principale : $\alpha$, $\kappa$, $\phi$, $\psi$ et $\omega$)[cite: 96].
* [cite_start]**Au Rang 2 (Niveau SSE - Structures Secondaires) :** C'est ici que le modèle prend de la hauteur[cite: 96]. [cite_start]Au lieu de redécouvrir laborieusement ces motifs, on fournit explicitement le type de structure secondaire déterminé par l'annotation DSSP (hélice, feuillet ou pelote), la taille de la SSE, son encodage positionnel, ainsi que des descripteurs de forme avancés par Analyse en Composantes Principales (linéarité, planarité, anisotropie, densité de contact, etc.)[cite: 96]. [cite_start]On ajoute aussi des vecteurs de déplacement géométriques précis (centre de masse, début, milieu, fin de la structure)[cite: 96].
* [cite_start]**Au Rang 3 (Niveau Protéine globale) :** À l'échelle de la macromolécule entière, on intègre des caractéristiques macroscopiques : la distribution globale des SSE, la taille totale, la fréquence de composition en acides aminés, la fréquence des types de SSE, et des mesures physiques globales comme le rayon de gyration[cite: 96].

[cite_start]Cette featurisation hiérarchique multi-échelle est unique[cite: 95]. [cite_start]Elle garantit que le réseau de neurones topologiques dispose, dès la première couche, des contraintes géométriques locales et des abstractions structurales globales[cite: 96]. C’est ce qui permet d’obtenir des représentations latentes extrêmement discriminantes pour la classification de repliements ou la prédiction de fonctions virales. »

### **Slide 8 : Les étapes du passage de messages (Message Passing Steps)**

« Entrons maintenant dans le cœur algorithmique du Deep Learning Topologique : le mécanisme de **passage de messages** à travers une cellule cible, notée ici $x$. Cette diapositive schématise de manière limpide les 4 étapes successives qui permettent de mettre à jour les caractéristiques de cette cellule.

Regardons le diagramme pas à pas :

1. **Étape 1 : Le calcul des messages locaux (Message)** Tout commence en haut du schéma. La cellule cible $x$ (représentée par le point orange au centre) commence par recevoir des signaux bruts provenant de ses différents voisinages. La force du TDL réside dans sa capacité à traiter simultanément plusieurs types de voisinages, notés ici $\mathcal{N}_1$, $\mathcal{N}_2$ et $\mathcal{N}_3$. Par exemple, pour un nœud de protéine, $\mathcal{N}_1$ peut être ses arêtes adjacentes (relations binaires), $\mathcal{N}_2$ les structures secondaires de rang supérieur auxquelles il appartient, et $\mathcal{N}_3$ ses nœuds voisins dans l'espace 3D. Pour chaque type de voisinage, un message spécifique est généré via une fonction dédiée.

2. **Étape 2 : L'agrégation intra-voisinage (Intra-neighborhood Aggregation)** Une fois les messages locaux générés, le réseau doit les condenser. C'est l'étape d'agrégation intra-voisinage. Pour chaque canal de voisinage pris individuellement, le modèle applique un opérateur de réduction (comme une somme, une moyenne ou un mécanisme d'attention). À la fin de cette étape, comme on le voit au niveau 2 du schéma, la cellule cible dispose de trois messages distincts et bien ordonnés, chacun résumant parfaitement les informations d'un type de structure topologique spécifique.

3. **Étape 3 : L'agrégation inter-voisinage (Inter-neighborhood Aggregation)** C'est l'une des grandes innovations présentées dans ce survey. Au niveau 3, le modèle fusionne ces différents canaux. Il applique une seconde fonction d'agrégation qui va combiner les messages issus de $\mathcal{N}_1$, $\mathcal{N}_2$ et $\mathcal{N}_3$ pour n'en former qu'un seul, global et multi-échelle. Cette flexibilité dans la combinaison des voisinages constitue un levier architectural majeur pour s'adapter à la complexité des données.

4. **Étape 4 : La mise à jour des caractéristiques (Feature Update)** Enfin, la quatrième et dernière étape est la mise à jour. Le réseau prend l'état de la cellule $x$ à l'instant t, y associe le message global multi-échelle tout juste calculé, et passe le tout dans une fonction de mise à jour (notée $U$). On obtient ainsi le nouvel état de la cellule pour la couche suivante, t+1.

En résumé, ce processus en 4 étapes permet à chaque entité de notre modèle — qu'il s'agisse d'un acide aminé, d'une interaction chimique ou d'une hélice $\alpha$ — d'assimiler de manière structurée et hiérarchique l'ensemble des informations de son environnement macromoléculaire. »

### **Slide 9 : Formalisation mathématique du passage de messages (Unraveling Message Passing)**

« Sur cette neuvième diapositive, nous allons traduire mathématiquement les 4 étapes conceptuelles que nous venons de décrire. L'article propose une équation générale unifiée qui régit la mise à jour des caractéristiques d'une cellule $x$ de rang $r$ à la couche $t+1$. 

Regardons comment cette formule unique encapsule parfaitement tout le processus :

1. **L'étape du Message (Etape 1) :**
   Dans la formule, elle correspond à la fonction $\psi_k$. Pour un voisinage spécifique indexé par $k$, cette fonction prend trois arguments : les caractéristiques actuelles de notre cellule cible $h_x^t$, les caractéristiques d'une cellule voisine $h_y^t$, et potentiellement des attributs structurels ou géométriques liés à leur connexion, notés $\Theta_{x,y}$. Cela permet d'évaluer l'influence locale de chaque voisin $y$.

2. **L'agrégation Intra-voisinage (Etape 2) :**
   Elle est représentée par le symbole de sommation ou d'agrégation externe $\bigoplus_{y \in \mathcal{N}_k(x)}$. Cet opérateur parcourt l'ensemble des cellules $y$ appartenant au voisinage spécifique $\mathcal{N}_k$ de notre cellule $x$. C'est ici qu'on réduit les messages individuels en un unique vecteur propre à ce canal de voisinage (par exemple, la somme ou la moyenne des messages des arêtes incidentes).

3. **L'agrégation Inter-voisinage (Etape 3) :**
   Elle se traduit par l'opérateur $\bigtriangleup_{k=1}^K$. Cet opérateur prend les $K$ vecteurs consolidés (un par type de voisinage $\mathcal{N}_k$) et les fusionne. C'est le moment clé où le réseau combine verticalement et horizontalement les informations : il fusionne par exemple le message venant des résidus adjacents (rang inférieur) et celui venant de l'hélice $\alpha$ englobante (rang supérieur).

4. **La mise à jour (Etape 4) :**
   Enfin, tout à gauche de l'équation, la fonction globale $U$ (pour *Update*) prend ce message fusionné multi-échelle et l'état précédent de la cellule $h_x^t$ pour calculer la nouvelle représentation $h_x^{t+1}$.

Cette formalisation est extrêmement puissante car elle est universelle. Les auteurs du survey démontrent que la quasi-totalité des architectures de réseaux topologiques de la littérature ne sont que des cas particuliers de cette équation, où l'on choisit des voisinages $\mathcal{N}_k$ et des fonctions $\psi$ ou $U$ spécifiques. Pour notre travail sur les protéines, cette abstraction mathématique nous donne un cadre parfait pour tester différentes combinaisons de voisinages sans avoir à réinventer l'algorithme à chaque fois. »

### **Slide 10 : La fonction de mise à jour des caractéristiques (Step 4: Update function)**

« Après avoir calculé et fusionné l'ensemble des messages multi-échelles provenant des différents voisinages topologiques, nous arrivons à la quatrième et dernière étape du processus : **la fonction de mise à jour** (notée $U$). Cette diapositive présente les grandes stratégies mathématiques proposées dans la littérature pour réaliser cette opération cruciale.

L'objectif de cette étape est de combiner intelligemment deux sources d'informations : la représentation actuelle de la cellule à la couche $t$, notée $h_x$, et le message global consolidé qu'elle vient de recevoir, noté $m_x$. Les auteurs du survey recensent trois approches principales :

1. **La stratégie par concaténation (Concatenated stream) :**
   C'est la première équation généraliste que vous voyez à l'écran. Ici, le réseau prend le vecteur de caractéristiques de la cellule $h_x$ et le vecteur du message $m_x$, puis les concatène (représenté par le symbole double barre $||$). Ce long vecteur combiné est ensuite multiplié par une matrice de poids apprenables $W$, auquel on ajoute un biais $b$, avant d'appliquer une fonction d'activation non linéaire $\sigma$ (comme une ReLU). Cette méthode a l'avantage de préserver l'intégrité brute des deux signaux avant leur transformation linéaire.

2. **La stratégie par flux séparés (Separate streamlines) :**
   La deuxième équation illustre une approche alternative où la cellule et le message sont traités de manière indépendante. On applique une matrice de poids dédiée $W_1$ sur les caractéristiques intrinsèques de la cellule, et une autre matrice de poids $W_2$ sur le message reçu. On fait ensuite la somme de ces deux projections linéaires avant d'appliquer la non-linéarité $\sigma$. C'est une méthode extrêmement courante qui permet au réseau d'apprendre à accorder une importance ou un "poids" différent à l'historique de la cellule par rapport aux informations provenant des voisinages.

3. **L'approche récurrente (Gated Recurrent Unit - GRU) :**
   Enfin, pour les architectures plus profondes ou complexes, le survey met en avant l'utilisation d'une cellule de type GRU (*Gated Recurrent Unit*). Au lieu d'une simple combinaison linéaire, la mise à jour de $h_x^{t+1}$ est gérée par des mécanismes de portes (portes de mise à jour et de réinitialisation) via une fonction `GRUCell`. Le message $m_x^{t+1}$ fait office de nouvelle entrée (*input*) tandis que $h_x^t$ sert d'état caché (*hidden state*). Cette approche est particulièrement puissante pour réguler le flux d'information à travers de nombreuses couches et éviter l'explosion ou la disparition du gradient.

Pour notre framework sur les protéines, le choix de cette fonction de mise à jour est loin d'être anodin. Utiliser des flux séparés ou un mécanisme récurrent comme une GRU permettra à nos représentations de résidus ou de structures secondaires de conserver leur identité géométrique locale tout en assimilant progressivement le contexte macromoléculaire global au fil des couches du réseau. »

### **Slide 11 : Visualisation du passage de messages par les diagrammes de tenseurs (Tensor Diagrams)**

« Pour formaliser visuellement et de manière rigoureuse ces différentes opérations de passage de messages, les auteurs du survey introduisent un outil de notation graphique extrêmement élégant : les **diagrammes de tenseurs** (*Tensor diagrams*). 

L'objectif de ces diagrammes est de suivre la trace d'une caractéristique mathématique depuis une cellule $y$ de rang initial à la couche $t$, jusqu'à une cellule cible $x$ de rang final à la couche $t+1$. Le tableau sur cette diapositive détaille comment chaque étape de notre algorithme correspond à un élément graphique et à une transformation de tenseurs précise :

1. **La définition du message envoyé ($m_{y\rightarrow x}^{(r'\rightarrow r)}$) :**
   Cette étape définit mathématiquement le message transmis d'une cellule de rang $r'$ à une cellule de rang $r$. Sur le diagramme, cela se traduit par l'utilisation de matrices de voisinage spécifiques (notées par exemple $B_1$, $B_1^T$, ou des matrices de Laplaciens $L$). Graphiquement, on trace une flèche orientée qui montre le flux de caractéristiques d'un rang vers un autre. Le survey distingue ici les approches purement convolutionnelles (sans attention) des approches avec attention.

2. **L'agrégation par type de voisinage ($m_{x}^{(r'\rightarrow r)}$) :**
   Cette ligne spécifie comment les messages reçus au sein d'un même voisinage $k$ sont condensés. Dans une architecture classique de type convolutionnelle standard, cela revient tout simplement à effectuer une somme de toutes les flèches ou messages incidents provenant de ce voisinage. Pour des modèles plus avancés, on applique une fonction générale de combinaison pour préserver la structure.

3. **La fusion inter-voisinages ($m_{x}^{(r)}$) :**
   Ici, le diagramme montre comment le modèle rassemble les informations issues de voisinages de natures différentes (par exemple, un voisinage horizontal au même rang et un voisinage vertical provenant d'un rang supérieur). La notation graphique fusionne ces flux, soit par une simple sommation, soit par une fonction de combinaison non linéaire plus complexe.

4. **La mise à jour de la cellule cible ($h_{x}^{t+1,(r)}$) :**
   Enfin, tout en bas du diagramme, l'état final de la cellule de rang $r$ à la couche $t+1$ est mis à jour en appliquant la fonction $U$, qui dépend de l'historique de la cellule à la couche $t$ et du message fusionné.

Ces diagrammes de tenseurs ne sont pas de simples illustrations : ils constituent une véritable syntaxe graphique. Ils permettent de concevoir visuellement de nouvelles architectures de réseaux de neurones topologiques complexes, en dessinant simplement des chemins de tenseurs entre différents rangs, avant même d'avoir à écrire la moindre ligne de code. »

### **Slide 12 : Exemples concrets de diagrammes de tenseurs (Example of tensor diagram)**

« Pour bien appréhender la puissance opérationnelle de cette syntaxe graphique, cette douzième diapositive met en opposition trois grandes familles de mécanismes de passage de messages à travers leurs diagrammes de tenseurs respectifs. Chaque colonne illustre comment une cellule cible $x_i$ (au centre, en violet) reçoit et fusionne les informations de son voisinage.

1. **La convolution standard (Standard Convolutional) :**
   C'est le schéma situé tout à gauche. Dans ce modèle classique, chaque cellule voisine $y_a, y_b, y_c, y_d$ transmet un message linéaire vers la cellule centrale $x_i$. Ces messages sont simplement pondérés par des coefficients fixes ou purement structurels, notés $c_{ia}, c_{ib}$, etc., qui dépendent uniquement de la connectivité topologique brute (comme les éléments d'une matrice d'adjacence ou d'un opérateur laplacien). Comme le montre le petit diagramme de tenseur en bas de la colonne, le flux est direct, uniforme, et les messages de même nature sont fusionnés par une simple somme.

2. **La convolution avec attention (Attentional Convolutional) :**
   Le schéma central introduit une couche d'intelligence algorithmique supplémentaire. Ici, la transmission n'est plus statique : le réseau calcule dynamiquement des coefficients d'attention (notés par les flèches courbes orange). L'importance du message envoyé par le voisin $y_b$ vers $x_i$ dépend explicitement de la corrélation entre les caractéristiques actuelles de $y_b$ et celles de $x_i$. Le diagramme de tenseur en bas reflète cette complexification en montrant un routage où les flèches de messages (en rouge) intègrent une fonction d'alignement dynamique avant l'agrégation.

3. **Le mécanisme général (General) :**
   Enfin, la colonne de droite représente le niveau d'abstraction le plus élevé et le plus flexible du survey. Dans le cas général, la fonction de message (notée $g^{(a,i)}$) est une fonction non linéaire arbitraire (comme un perceptron multicouche dédié) qui prend en entrée la totalité des informations combinées du nœud source, du nœud cible et de leur relation topologique. Le diagramme de tenseur associé en bas montre que les flux provenant de voisinages de dimensions ou de rangs totalement hétérogènes peuvent être traités par des fonctions de transformation indépendantes avant d'être fusionnés de manière non linéaire.

Pour notre architecture de traitement des protéines, cette décomposition est fondamentale. Elle nous montre que nous pouvons commencer par valider nos modèles avec une approche convolutionnelle standard stable, puis faire évoluer le réseau vers des mécanismes attentionnels ou généraux pour capturer des motifs biophysiques ultra-fins, sans jamais modifier la structure topologique sous-jacente de notre complexe combinatoire. »

### **Slide 13 : Revue de la littérature – Les architectures sur Hypergraphes (Literature review: Hypergraphs)**

« Avec cette treizième diapositive, nous entamons la section du survey dédiée à la classification des modèles existants dans la littérature. Les auteurs cartographient ici l'ensemble des architectures de réseaux de neurones sur hypergraphes en les classant selon la taxonomie unifiée et les diagrammes de tenseurs que nous venons d'étudier.

Comme vous pouvez le voir sur la matrice à l'écran, la littérature est segmentée en deux grandes familles de passage de messages :

1. **Les approches convolutionnelles standards (Standard Convolutional) :**
   À gauche, on retrouve des modèles pionniers comme *HyperSAGE*, *HGC-RNN*, *DHGCN* ou encore *AllSet*. Si vous observez leurs diagrammes de tenseurs respectifs, vous remarquerez qu'ils effectuent des projections linéaires successives très structurées. L'information part généralement des caractéristiques des nœuds (en bleu), passe par une matrice d'incidence (notée $B_1^T$) pour caractériser les hyperarêtes (en rose), y applique parfois des opérateurs de normalisation ou des fonctions d'activation $\sigma$, puis redescend vers les nœuds via la matrice transposée $B_1$. Ce sont des modèles rigides, mathématiquement stables, très efficaces pour propager l'information de manière uniforme à l'intérieur d'un groupe d'entités.

2. **Les approches attentionnelles et générales (Attentional Convolutional/General) :**
   À droite, la matrice liste les modèles qui intègrent une modulation dynamique des messages, tels que *DHGNN*, *HyperGAT*, *SHARE*, ou *HTNN*. Graphiquement, cela se traduit par les arcs ou demi-cercles rouges et noirs entourant les cellules. Ces symboles indiquent que le passage de messages entre un nœud et son hyperarête est pondéré dynamiquement par un mécanisme d'attention. Par exemple, dans *HyperGAT*, le réseau apprend si un acide aminé particulier a plus d'importance qu'un autre au sein d'une même poche d'interaction fonctionnelle.

Ce qu'il faut retenir de cette classification, c'est l'immense valeur du travail d'unification du survey : des dizaines de papiers publiés de manière éparpillée ces dernières années, avec des notations mathématiques parfois contradictoires, sont ici résumés et comparés en un coup d'œil grâce à la syntaxe visuelle des diagrammes de tenseurs. 

Pour notre projet, l'analyse de cette diapositive nous montre que si nous décidons d'isoler certaines propriétés globales sous forme d'hyperarêtes (comme un regroupement de résidus hydrophobes), nous disposons déjà d'un catalogue d'outils algorithmiques très mûr, allant de la convolution stable à l'attention dynamique, pour faire circuler l'information. »

### **Slide 14 : Revue de la littérature – Les architectures sur Complexes Simpliciaux (Literature review: simplicial complexes)**

« Sur cette quatorzième diapositive, nous analysons la deuxième grande famille de la revue de la littérature, qui est historiquement l'une des plus riches du Deep Learning Topologique : les modèles appliqués aux **complexes simpliciaux**. 

Les auteurs ont répertorié une quantité impressionnante de contributions scientifiques. Contrairement aux hypergraphes qui n'ont que deux rangs, les complexes simpliciaux manipulent une hiérarchie stricte de rangs $r, r-1, r+1$, ce qui enrichit considérablement les possibilités algorithmiques. Les modèles sont ici encore séparés en deux philosophies :

1. **Les modèles convolutionnels standards (Standard Convolutional) :**
   À gauche de la matrice, on retrouve des architectures fondatrices comme *SNN (Ebli et al.)*, *SCCONV (Bunch et al.)*, ou *SCN (Yang et al.)*. Si vous observez attentivement leurs diagrammes de tenseurs, le passage de messages y est guidé de manière rigoureuse par des opérateurs algébriques profonds. On y voit l'utilisation fréquente des matrices de frontières (notées $B_r, B_{r+1}$ et leurs transposées) ainsi que des opérateurs Laplaciens simpliciaux combinatoires, qu'il s'agisse du Laplacien inférieur ($L_{\downarrow,r}$) ou du Laplacien supérieur ($L_{\uparrow,r}$). Ces réseaux permettent de propager l'information "vers le haut" (des nœuds vers les arêtes puis vers les triangles) et "vers le bas" de manière parfaitement symétrique.

2. **Les modèles attentionnels et généraux (Attentional Convolutional/General) :**
   À droite de la matrice, le survey répertorie des modèles capables de moduler l'intensité de ces flux de messages algébriques, tels que *SAN*, *SAT*, *SGAT*, ou encore *MPSN (Bodnar et al.)*. Ces architectures intègrent des mécanismes d'attention (indiqués par les arcs rouges sur les diagrammes) appliqués directement sur les Laplaciens ou sur les structures adjacentes. L'objectif est d'apprendre quelles sous-faces ou quelles arêtes limitrophes contiennent l'information la plus critique pour la tâche globale.

Ce qu'il est capital de percevoir sur cette diapositive, c'est la complexité des chemins de tenseurs que le survey parvient à unifier. Un modèle comme *MPSN* ou *SCoNe* montre des flux croisés complexes où une entité de rang $r$ met à jour ses caractéristiques en combinant simultanément les signaux de ses frontières inférieures et de ses co-frontières supérieures.

Pour notre projet, cette partie de la littérature nous montre à quel point le traitement des structures géométriques triangulées est mathématiquement mûr. Si nous choisissons d'extraire des motifs structuraux rigides et purement simpliciaux dans nos protéines (comme des triplets d'atomes), nous disposons ici d'un arsenal d'opérateurs laplaciens très puissants pour propager les informations géométriques. »

### **Slide 15 : Revue de la littérature – Les architectures Cellulaires et Combinatoires (Literature review: cellular and combinatorial)**

« Pour clore cette vue d’ensemble de la littérature, nous arrivons à la quinzième diapositive, qui synthétise les travaux portant sur les deux domaines les plus flexibles et les plus riches en expressions topologiques : les **complexes cellulaires** et les **complexes combinatoires**. C'est précisément dans cette catégorie que se positionne notre stratégie de modélisation pour les structures de protéines.

Le survey sépare ces contributions en deux sections horizontales :

1. **Les architectures sur Complexes Cellulaires (Cellular) :**
   En haut de la matrice, on retrouve des modèles phares comme *CXN (Hajij et al.)*, *CWN (Bodnar et al.)*, ou *CAN (Giusti et al.)*. Les diagrammes de tenseurs montrent que ces réseaux tirent parti de la flexibilité des cellules polygonales. On y retrouve l'utilisation d'opérateurs laplaciens et de matrices de frontières adaptés à des dimensions arbitraires ($L_{\perp,r}$, $L_{\dagger,r}$). Les modèles comme *CAN* (*Cell Attention Networks*) y introduisent des mécanismes d'attention sophistiqués pour pondérer l'importance d'une cellule par rapport à ses voisines directes, ce qui s'avère excellent pour capturer la géométrie de cycles chimiques fermés non triangulaires.

2. **Les architectures sur Complexes Combinatoires (Combinatorial) :**
   En bas de la matrice, nous entrons dans le domaine le plus généralisé, illustré notamment par le modèle *HOAN* (*Higher-Order Attention Networks*). Si vous observez attentivement les diagrammes de tenseurs de cette section, vous remarquerez une rupture visuelle majeure : les flèches de messages peuvent sauter directement d'un rang 0 à un rang 2 ou d'un rang 2 à un rang 0 (noté par exemple via des opérateurs de transfert direct comme $B_{2\rightarrow0}^T$), s'affranchissant de l'obligation de passer par le rang intermédiaire 1. C'est l'incarnation même des relations de type "Ensemble" où la hiérarchie n'est plus une contrainte géométrique stricte, mais une flexibilité logique.

**Pourquoi cette diapositive valide-t-elle nos choix technologiques ?**
Elle démontre que le cadre théorique du passage de messages topologiques possède la plasticité nécessaire pour soutenir notre framework. En adoptant une structure de complexe combinatoire inspirée de *Topotein* et des formalismes généraux comme *HOAN*, nous pouvons légitimement faire communiquer nos acides aminés (rang 0) directement avec leurs structures secondaires (rang 2), tout en régulant ces flux par les mécanismes d'attention recensés dans ce tableau. 

Nous disposons ainsi d'un cadre unifié, mathématiquement validé par la littérature, pour propager l'information biologique à toutes les échelles de la macromolécule. »