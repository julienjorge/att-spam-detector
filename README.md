# 📱 AT&T Spam Detector
### Détection automatique des SMS indésirables

> Un modèle lit le texte d'un SMS et décide s'il s'agit d'un spam. Deux approches
> sont comparées : un réseau de neurones construit de zéro et un modèle de langage
> pré-entraîné affiné sur nos données.

## Le projet

Les clients d'un opérateur reçoivent des SMS indésirables. L'opérateur veut les
filtrer automatiquement, à partir du seul contenu du message, sans bloquer les
messages légitimes.

Nous avons construit et comparé deux classifieurs de SMS, en deep learning :

- **un réseau construit de zéro**, qui apprend sa propre représentation des mots sur
  nos seuls SMS ;
- **DistilBERT affiné**, un modèle de langage pré-entraîné sur un très grand corpus
  anglais, adapté à notre tâche.

Le livrable est un notebook qui prépare les données, entraîne les deux modèles et
mesure leurs performances sur un jeu de test.

## Les données

- **Un fichier de 5 572 SMS en anglais**, fourni dans l'énoncé, chacun étiqueté
  spam ou ham (message légitime). Quelques messages mal découpés par le format CSV
  sont recollés, et 414 doublons sont retirés : il reste 5 158 SMS, dont 642 spams
  (12,4 %).

## Comment ça marche

1. **Exploration** : structure du fichier, réparation des messages coupés,
   suppression des doublons, longueur des messages par classe.
2. **Découpage en trois jeux** : entraînement (70 %), validation (15 %) et test
   (15 %), en conservant la proportion de spams dans chacun.
3. **Tokenisation** : chaque SMS est découpé en tokens (morceaux de mots) convertis
   en numéros, puis complété ou tronqué à une longueur commune choisie pour ne
   tronquer aucun spam.
4. **Premier modèle** : une couche d'embedding, qui donne un vecteur à chaque token,
   une moyenne des vecteurs du message, puis deux couches denses. Entraîné sur
   12 epochs, le nombre lu sur la courbe de validation.
5. **Second modèle** : DistilBERT, avec son propre tokenizer, auquel nous ajoutons
   une tête de classification. Affiné sur 3 epochs, lues de la même façon.
6. **Seuil de décision** : pour chaque modèle, le seuil qui transforme la
   probabilité de spam en décision est choisi sur le jeu de validation.
7. **Évaluation** : les deux modèles sont mesurés une seule fois sur le jeu de test.

Trois choix importants :

- **Ne jamais tronquer un spam.** Les spams sont plus longs que les hams, et leur
  longueur est un indice ; la longueur commune est fixée au-dessus du spam le plus
  long du jeu d'entraînement.
- **Un seuil de 0,80 plutôt que 0,50.** Bloquer un message légitime est plus grave
  que laisser passer un spam ; un seuil élevé privilégie la précision. Il est choisi
  sur la validation et jamais revu après le test.
- **La même démarche pour les deux modèles.** Même découpage, même fonction
  d'entraînement, même choix du seuil : la comparaison porte sur les modèles, pas
  sur la méthode.

## Les résultats

Sur le jeu de test (774 SMS, dont 96 spams), au seuil de 0,80 :

| | Réseau construit de zéro | DistilBERT affiné |
|---|---|---|
| Paramètres | 3,2 millions | 67 millions |
| Hams bloqués à tort | 3 | 0 |
| Spams non détectés | 12 | 1 |
| Précision (spam) | 96,6 % | 100 % |
| Rappel (spam) | 87,5 % | 99,0 % |
| F1 (spam) | 0,918 | 0,995 |
| Accuracy | 98,1 % | 99,9 % |

Un modèle qui classerait tout en ham obtiendrait 87,6 % d'accuracy. Le premier
modèle fait déjà beaucoup mieux ; le second ne commet qu'une erreur sur 774 messages.

## Exécuter le projet

Arborescence :

- 01-AT&T_spam_detector.ipynb : le notebook complet
- data/raw/spam.csv : les SMS d'origine, non modifiés

Prérequis : Python 3.13 avec pandas, numpy, plotly, scikit-learn, torch, tiktoken et
transformers. Aucune clé ni fichier de configuration.

Lancement : ouvrir le notebook et exécuter toutes les cellules depuis la racine du
projet. Le premier lancement télécharge DistilBERT (environ 270 Mo). L'exécution
complète prend une vingtaine de minutes avec un accélérateur de calcul (puce
graphique), davantage sur processeur seul.

## Limites et suites possibles

- **Des jeux de validation et de test petits** (97 et 96 spams) : quelques messages
  suffisent à faire varier les métriques, et le seuil ne peut pas être réglé
  finement. Une évaluation sur un volume plus important serait nécessaire avant une
  mise en production.
- **Un corpus en anglais**, d'un seul type de messages : le modèle n'est pas
  transposable tel quel à d'autres langues ou à d'autres canaux.
- **Pas de sauvegarde des modèles entraînés** : chaque exécution réentraîne tout.
  Enregistrer les poids permettrait de prédire sans réentraîner.
