# Projet Walmart Sales : prédiction des ventes hebdomadaires

Certification CDSD RNCP35288 - Bloc 3 : Analyse prédictive de données structurées par l'intelligence artificielle

## Contexte

Le service marketing de Walmart veut comprendre comment les ventes
hebdomadaires de ses magasins dépendent d'indicateurs économiques
(température, prix du carburant, CPI, chômage) et du calendrier (jours fériés),
afin de mieux planifier ses campagnes.

## Objectifs

1. Explorer le dataset fourni et construire un pipeline de preprocessing
   avec scikit-learn.
2. Entraîner un modèle de **régression linéaire** et le confronter à des
   baselines simples.
3. Limiter le surapprentissage avec une **régression régularisée** (Ridge et
   Lasso), dont le paramètre `alpha` est réglé par validation croisée.
4. Identifier les variables qui comptent vraiment, en mesurant la fiabilité des
   coefficients par bootstrap.

## Données

`data/Walmart_Store_sales.csv` : 150 lignes, 8 colonnes (jeu custom
fourni pour le projet par Jedha, dérivé d'une compétition Kaggle).

lien : https://julie-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/projects-supervised-machine-learning-ft/walmart-sales-ft/Walmart_Store_sales.csv

| Colonne | Description |
|---|---|
| `Store` | Numéro du magasin (20 magasins) |
| `Date` | Semaine (format `jj-mm-aaaa`, tous des vendredis) |
| `Weekly_Sales` | **Cible** : ventes hebdomadaires en dollars |
| `Holiday_Flag` | Semaine fériée (1) ou non (0) |
| `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` | Indicateurs locaux |

Toutes les colonnes, sauf `Store`, contiennent des valeurs manquantes (12 à 18
sur 150 lignes, dont 14 sur la cible et 18 sur la date). Après nettoyage
(suppression des cibles manquantes, puis application de la règle des outliers
±3σ), il reste 131 lignes exploitables.

## Etapes du projet

- **Préparation** : suppression des lignes sans cible (150 → 136), variables
  `Year`, `Month`, `Day` issues de la date, suppression des outliers à ±3σ
  (136 → 131), découpage train/test 80/20 (104 / 27 lignes).
- **Pipeline** : imputation (médiane / modalité la plus fréquente),
  normalisation, encodage one-hot avec modalité de référence. Il est intégré au
  modèle et ré-ajusté à chaque pli de validation croisée, sans fuite
  d'information.
- **Évaluation** : R² et RMSE (train, test et validation croisée), MAPE sur le
  test. La validation croisée répétée (5 plis × 10 répétitions) est plus
  fiable que le seul test (27 lignes).
- **Modèles de référence** : moyenne globale sur jeu train, magasin seul (régression sur la
  seule variable `Store`, équivalente à la moyenne par magasin) et toutes les
  variables sauf le magasin.
- **Fiabilité du jeu de test** : la comparaison régression / magasin seul est
  répétée sur 50 découpages train/test différents, et on mesure le poids de la
  plus grosse erreur du test.
- **Diagnostic et coefficients** : graphique réel vs prédit, biais sur le test,
  puis intervalles à 95 % des coefficients par bootstrap (300 rééchantillons
  du jeu d'entraînement).
- **Réglage de Ridge et Lasso** : `GridSearchCV` (40 valeurs de `alpha`) avec le
  même découpage de validation croisée que pour l'évaluation, ce qui rend les
  scores comparables.

## Résultats

| Modèle | R² test | RMSE test | R² CV | RMSE CV | MAPE test |
|---|---|---|---|---|---|
| Moyenne globale | −0,032 | 598 753 $ | −0,069 | 674 237 $ | 80,0 % |
| Magasin seul (référence) | 0,925 | 161 621 $ | 0,927 | 170 920 $ | 10,9 % |
| Sans le magasin | −0,147 | 631 112 $ | −0,172 | 701 443 $ | 84,1 % |
| Régression linéaire | 0,898 | 188 478 $ | 0,949 | 144 608 $ | 15,1 % |
| Ridge | 0,899 | 187 207 $ | 0,950 | 143 516 $ | 15,0 % |
| Lasso | 0,902 | 184 476 $ | 0,950 | 142 777 $ | 15,2 % |

**Interprétations et résultats**

- **Le magasin explique l'essentiel des ventes** : une simple moyenne par
  magasin est déjà un baseline solide (R² de 0,93). À l'inverse, un modèle
  sans la variable magasin est moins bon que la moyenne (R² CV de −0,17).
- La régression linéaire améliore le RMSE de 15 % en validation croisée, mais
  pas sur le jeu de test (27 lignes). Ce test est trop petit pour trancher :
  sur 50 découpages train/test différents, la régression bat le magasin seul
  37 fois (gain médian d'environ 10 %, de −37 % à +30 % selon le découpage), et
  une seule observation du test représente 36 % de son erreur quadratique.
  Sans elle, les deux modèles sont à égalité (autour de 155 k$).
- Le modèle surestime en moyenne les ventes sur le test (résidu moyen de
  −98 k$). L'observation extrême (−589 k$) n'explique qu'une petite part de ce
  biais : sans elle, il reste de −79 k$.
- Régression linéaire, Ridge et Lasso sont à 1,3 % d'écart : **la
  régularisation n'apporte presque rien** ici, il n'y a pas de gagnant net.
  Le Ridge retient un `alpha` très faible (0,027) et le Lasso n'annule qu'un
  seul coefficient sur 26 (celui du magasin 19). L'écart de R² entre train et
  test (0,978 / 0,898 pour la régression) n'est pas réduit.
- Le CPI et le chômage sont expliqués à 99 % et 91 % par le magasin : ce sont
  surtout des identifiants de magasin. Seuls la température (−), le chômage (−)
  et le mois (+) ont un effet dont l'intervalle de confiance à 95 % (bootstrap)
  exclut 0 ; ceux du CPI, du carburant, du jour férié, de l'année et du jour
  ne sont pas distinguables de 0. Ces effets restent modestes (quelques
  dizaines de milliers de dollars) face aux écarts entre magasins (de −1,36 M$
  à +0,61 M$ par rapport au magasin 1). Un effet stable n'est pas pour autant
  un effet causal.

**Point d'attention.** La règle des ±3σ de l'énoncé supprime 5 lignes, toutes
du **magasin 12** (chômage de 13,5 à 14,3 %), qui disparaît donc entièrement du
jeu de données. Son chômage est structurellement élevé : ce n'est pas du bruit
de mesure.

## Limites

- Très peu d'observations (131 lignes, 27 en test) : toutes les métriques sont
  incertaines.
- 12 % de dates manquantes (imputées) ; le jour du mois est peu pertinent.
- Les observations d'un même magasin dans le temps ne sont pas indépendantes ;
  un découpage temporel serait plus réaliste pour une vraie prévision.
- Une validation croisée imbriquée serait plus rigoureuse pour comparer les
  modèles réglés.
- Le magasin 12 est retiré par la règle de l'énoncé ; à discuter avec le métier.
- Pistes : plus de données, modèles non linéaires (arbres, forêts aléatoires),
  variables propres à chaque magasin.

## Structure du repo

```
cdsd_bloc3_projet_walmart_sales/
├── assets
│   └── images
│       ├── fig_01_valeurs_manquantes.png
│       ├── fig_02_distribution_ventes.png
│       ├── fig_03_ventes_magasin_ferie.png
│       ├── fig_04_ventes_vs_variables.png
│       ├── fig_05_correlations.png
│       ├── fig_06_cpi_par_magasin.png
│       ├── fig_07_outliers.png
│       ├── fig_08_r2_modeles_base.png
│       ├── fig_09_reel_vs_predit.png
│       ├── fig_10_effets_bootstrap.png
│       ├── fig_11_courbes_alpha.png
│       └── fig_12_comparaison_modeles.png
├── data
│   └── Walmart_Store_sales.csv
├── notebook
│   └── walmart_weekly_sales.ipynb
├── README.md
└── requirements.txt
```

## Installation et exécution

```powershell
conda create -n cdsd_b3_walmart_sales python=3.11.17 pip -y
conda activate cdsd_b3_walmart_sales
python -m pip install -r requirements.txt    
```

Le notebook utilise des chemins relatifs et une graine fixe
(`RANDOM_STATE = 42`). Les figures sont enregistrées en PNG dans
`assets/images/` (cela nécessite le paquet `kaleido` ; sans lui, elles
s'affichent simplement dans le notebook).
