# CLAUDE.md — Modélisation de la corrélation spatiale du puceron de la laitue

## 1. Contexte

Ce projet s'inscrit dans la thèse de doctorat d'Emma Dubrûle (Université de Sherbrooke,
Département de géomatique appliquée) portant sur la détection en temps réel du puceron
de la laitue (*Nasonovia ribisnigri*) à l'aide d'IA en périphérie.

Le présent module implémente la composante **modélisation géospatiale** du projet :
on dispose de ~20 capteurs fixes (déplaçables manuellement) répartis sur un champ
contenant des milliers de plants. Chaque capteur retourne une **probabilité de
présence** dans `[0, 1]` (fraction d'observations positives sur une fenêtre
temporelle), et l'on veut estimer la probabilité de présence des pucerons sur
l'ensemble du champ (cartographie probabiliste) en exploitant la corrélation
spatiale.

Comme on ne dispose pas encore de données terrain, ce module commence par une
**simulation contrôlée** : on génère un champ synthétique avec une vérité terrain
connue (probabilité et présence/absence de pucerons à chaque plant), on y place des
capteurs, puis on applique différentes méthodes pour reconstruire la carte
probabiliste à partir des observations partielles. La vérité terrain simulée sert à
évaluer et comparer les méthodes.

## 2. Objectifs

1. Simuler un champ de laitue de 100 × 1000 plants avec une distribution réaliste
   de la présence des pucerons (effet de bordure, foyers d'infestation, autocorrélation
   spatiale).
2. Implémenter plusieurs schémas de placement des capteurs (aléatoire, régulier,
   stratifié, biaisé bordure).
3. Implémenter le maximum de méthodes pertinentes pour estimer la probabilité de
   présence en tout point du champ à partir des observations des capteurs.
4. Évaluer et comparer les méthodes sur des métriques rigoureuses (AUC, log-loss,
   Brier score, calibration, CRPS).
5. Produire des visualisations claires (cartes vérité/estimation/incertitude).

## 3. Structure du projet

```
aphid_spatial/
├── CLAUDE.md                          # ce fichier
├── pyproject.toml                     # config projet + dépendances
├── README.md                          # documentation utilisateur
├── src/
│   └── aphid_spatial/
│       ├── __init__.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── field.py               # génération de la vérité terrain
│       │   └── sensors.py             # schémas de placement des capteurs
│       ├── methods/
│       │   ├── __init__.py
│       │   ├── exploration.py         # statistiques descriptives, cartes
│       │   ├── autocorrelation.py     # Moran, Geary, Getis-Ord, LISA
│       │   ├── point_process.py       # Ripley K/L/g, CSR
│       │   ├── geostatistics.py       # variogrammes, krigeage
│       │   ├── lattice.py             # CAR, SAR, MRF/Ising
│       │   ├── hierarchical.py        # GLMM bayésien spatial
│       │   ├── gp.py                  # processus gaussien
│       │   ├── ml.py                  # forêts aléatoires spatiales
│       │   └── sadie.py               # SADIE (entomologie)
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # AUC, log-loss, Brier, CRPS
│       │   └── cv.py                  # validation croisée spatiale
│       └── visualization/
│           ├── __init__.py
│           ├── maps.py                # cartes 2D de probabilité/incertitude
│           └── diagnostics.py         # variogrammes, courbes de calibration
├── notebooks/
│   ├── 01_simulation.ipynb
│   ├── 02_exploration.ipynb
│   ├── 03_geostatistics.ipynb
│   ├── 04_lattice_mrf.ipynb
│   ├── 05_hierarchical_bayes.ipynb
│   ├── 06_ml_methods.ipynb
│   └── 07_comparison.ipynb
├── tests/
│   ├── test_simulation.py
│   ├── test_methods.py
│   └── test_metrics.py
├── data/
│   └── simulated/                     # champs synthétiques (npz)
└── outputs/
    ├── figures/
    └── results/                       # tables de métriques (csv)
```

## 4. Environnement et dépendances

**Python ≥ 3.11**, environnement géré par `uv` ou `conda`.

### Dépendances principales

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "matplotlib>=3.8",
    "seaborn>=0.13",

    # Géostatistique
    "gstools>=1.5",          # variogrammes, champs gaussiens, krigeage
    "scikit-gstat>=1.0",     # variogrammes empiriques
    "pykrige>=1.7",          # krigeage ordinaire/universel/indicateur

    # Statistiques spatiales
    "libpysal>=4.10",        # poids spatiaux
    "esda>=2.5",             # Moran, Geary, Getis-Ord, LISA
    "spreg>=1.4",            # régression spatiale (SAR)
    "pointpats>=2.4",        # processus ponctuels (Ripley)

    # Bayésien
    "pymc>=5.10",            # GLMM, MRF, hiérarchique
    "arviz>=0.17",           # diagnostic MCMC

    # ML
    "scikit-learn>=1.4",     # GP, Random Forest, métriques
    "gpy>=1.13",             # processus gaussiens flexibles

    # Utilitaires
    "tqdm>=4.66",
    "joblib>=1.3",
    "rasterio>=1.3",         # I/O raster pour exports
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "jupyter"]
```

### Note sur INLA

L'approche INLA-SPDE (référence en bayésien spatial) n'a pas d'équivalent natif en
Python. Deux options : (a) PyMC avec covariance Matérn — plus lent mais 100% Python ;
(b) `rpy2` + `R-INLA` si on veut la version « gold standard ». Commencer avec (a),
documenter (b) en option avancée.

## 5. Phase 1 — Simulation du champ

### 5.1 Spécifications

- **Grille** : 100 lignes × 1000 colonnes (100 000 plants), espacement régulier de
  30 cm (paramétrable). Coordonnées en mètres.
- **Sortie** : pour chaque plant, une probabilité de présence `p ∈ [0,1]` et un
  tirage Bernoulli `y ∈ {0,1}`.

### 5.2 Modèle générateur

Combine trois composantes pour `logit(p) = μ + f_edge + f_GRF + f_hotspots` :

1. **Niveau de base** : `μ = logit(0.05)` (prévalence moyenne ~5 %).
2. **Effet de bordure** : la littérature (Mackenzie & Vernon 1988, Severtson 2015)
   indique une infestation jusqu'à 25 % plus élevée dans les 20-30 m du bord.
   Modéliser par `f_edge = β_edge * exp(-d_bord / λ_edge)` avec `λ_edge ≈ 20 m`.
3. **Champ gaussien (GRF)** : générer un champ gaussien stationnaire avec covariance
   Matérn (ν=1.5, portée 5-15 m), variance σ² = 1.5. Utiliser `gstools.SRF`.
4. **Foyers (hotspots)** : tirer 3-8 centres aléatoires (préférentiellement près des
   bords) ; ajouter une bosse gaussienne 2D autour de chaque centre.

Tirer ensuite `y ~ Bernoulli(p)` pour obtenir la vérité terrain binaire.

### 5.3 API à implémenter (`simulation/field.py`)

```python
@dataclass
class FieldConfig:
    n_rows: int = 100
    n_cols: int = 1000
    spacing_m: float = 0.30
    base_prevalence: float = 0.05
    edge_strength: float = 1.5
    edge_lambda_m: float = 20.0
    matern_range_m: float = 10.0
    matern_nu: float = 1.5
    matern_var: float = 1.5
    n_hotspots: tuple[int, int] = (3, 8)
    hotspot_strength: float = 2.0
    hotspot_radius_m: float = 8.0
    seed: int = 42

@dataclass
class Field:
    coords: np.ndarray    # (N, 2), en mètres
    prob: np.ndarray      # (N,), probabilité vraie
    presence: np.ndarray  # (N,), {0,1}
    config: FieldConfig

    def to_grid(self, what: Literal["prob", "presence"]) -> np.ndarray:
        """Retourne un tableau 2D (n_rows, n_cols)."""

def simulate_field(config: FieldConfig) -> Field:
    """Génère un champ synthétique."""
```

### 5.4 Critères d'acceptation

- Reproductibilité totale via `seed`.
- Sauvegarde/chargement en `.npz`.
- Variogramme empirique du `logit(p)` simulé doit retrouver approximativement la
  portée Matérn paramétrée (test unitaire).
- Visualisation : carte 2D de la probabilité vraie + carte 2D de la présence.

## 6. Phase 2 — Placement des capteurs

### 6.1 Schémas à implémenter (`simulation/sensors.py`)

1. **Uniforme aléatoire** : `n` capteurs tirés sans remise sur la grille.
2. **Grille régulière** : `n` capteurs sur une sous-grille (le plus uniforme possible).
3. **Aléatoire stratifié** : diviser le champ en blocs, 1 capteur par bloc.
4. **Biaisé bordure** : tirer avec probabilité décroissante avec la distance au bord
   (test du « si on suit la littérature »).
5. **Aléatoire avec contrainte de distance min** : Poisson-disk sampling.

### 6.2 Modèle d'observation

Chaque capteur retourne une **probabilité de présence** dans `[0, 1]`, pas une
détection binaire. Concrètement, le capteur en position `s_i` observe le plant
sous lui (`sensor_radius = 0`) ou un voisinage `(2r+1) × (2r+1)`
(`sensor_radius ≥ 1`, simule le champ de vue de la caméra) ; la probabilité
locale `prob_local` est la **moyenne** de la probabilité vraie `prob` sur cette
fenêtre. Le capteur effectue ensuite `n_observations` mesures temporelles
indépendantes, chacune Bernoulli de `prob_local`, et retourne la fraction de
positifs `k / n_observations`. Avec `n_observations = None`, le capteur lit
`prob_local` exactement (limite `K → ∞`, baseline théorique sans bruit
d'estimation).

Justification : les capteurs ne fournissent pas une détection ponctuelle mais
une mesure intégrée dans le temps (durée d'exposition à l'infestation). Cette
information graduée est plus riche qu'un simple 0/1 pour les méthodes
d'interpolation spatiale.

### 6.3 API

```python
@dataclass
class SensorConfig:
    n_sensors: int = 20
    placement: Literal["uniform", "grid", "stratified", "edge_biased", "poisson_disk"] = "uniform"
    sensor_radius: int = 0           # 0 = un plant ; 1 = 3x3 (moyenne de prob)
    n_observations: int | None = 50  # K mesures temporelles ; None = exact
    seed: int = 0
    edge_lambda_m: float = 20.0      # utilisé par placement="edge_biased"

@dataclass
class SensorReadings:
    sensor_idx: np.ndarray      # indices dans la grille
    coords: np.ndarray          # (n_sensors, 2)
    obs: np.ndarray             # (n_sensors,), float64 dans [0, 1]
    prob_local: np.ndarray      # (n_sensors,), probabilité réelle locale
                                # (référence pour analyse, à ne pas utiliser
                                # par les méthodes d'estimation)

def place_sensors(field: Field, config: SensorConfig) -> SensorReadings: ...
```

### 6.4 Critères d'acceptation

- Plusieurs schémas reproductibles.
- Tests : exactement `n_sensors` retournés ; aucun doublon ; coordonnées dans la
  grille.

## 7. Phase 3 — Méthodes d'estimation

Chaque méthode expose une interface uniforme :

```python
class SpatialMethod(Protocol):
    def fit(self, readings: SensorReadings, field_meta: FieldMeta) -> None: ...
    def predict_proba(self, query_coords: np.ndarray) -> np.ndarray: ...
    def predict_uncertainty(self, query_coords: np.ndarray) -> np.ndarray | None: ...
    name: str
```

### 7.1 Exploration et statistiques descriptives (`exploration.py`)

- Carte des observations sur les capteurs (gérée par `visualization.maps.plot_sensors`).
- `nearest_sensor_distance(readings, query_coords)` : distance au capteur le plus
  proche en tout point.
- `nearest_sensor_value(readings, query_coords)` : valeur ``obs`` du capteur le
  plus proche (utilisé comme feature par le RF).
- `inter_sensor_distances(readings)` : toutes les distances inter-capteurs
  (pour histogramme).
- `edge_distance(field_meta)` : distance au bord pour chaque cellule (utilisé
  comme covariable par UK et RF).
- `descriptive_summary(readings, field_meta)` : résumé numérique en une passe.
- `BaselineConstant` (classe `SpatialMethod`) : prédit la prévalence empirique
  partout — borne inférieure de comparaison.

### 7.2 Autocorrélation spatiale (`autocorrelation.py`)

Avec ~20 capteurs c'est limite, mais utile pour caractériser. Utiliser `esda` et
`libpysal`.

- **Matrices de poids** : K plus proches voisins (K=3,5,8), distance avec seuil,
  Gaussien-décroissant.
- **I de Moran global** : test de l'autocorrélation globale ; permutations pour le
  p-value.
- **c de Geary global**.
- **Getis-Ord G global** (pour tester clustering de valeurs hautes vs basses).
- **LISA (Moran local)** : identifier capteurs en *cluster* HH/LL et *outliers*
  HL/LH.
- **Getis-Ord Gᵢ\*** : points chauds/froids locaux.

Sortie : `dict` avec statistique, p-value, et carte LISA.

### 7.3 Processus ponctuels (`point_process.py`)

Utile si on traite les capteurs « positifs » comme un semis ponctuel sous la
référence de tous les capteurs.

- **Fonction K de Ripley** avec correction des effets de bord (Ripley, isotropic).
- **Fonction L** = √(K/π) - r.
- **Fonction g de pair-correlation**.
- **Test de complète aléa spatial (CSR)** par enveloppes Monte Carlo.
- **Lambda(s) — intensité par lissage** (KDE 2D).

### 7.4 Géostatistique (`geostatistics.py`)

Cœur des méthodes pour produire la carte probabiliste continue.

- **Variogramme empirique** (variable = `obs ∈ [0, 1]`, observation
  probabiliste du capteur) : avec `scikit-gstat` ou `gstools`. Tracer +
  ajuster modèles sphérique / exponentiel / gaussien / Matérn. Le passage
  au continu rend le variogramme plus stable qu'avec une variable binaire.
- **Krigeage ordinaire** sur les observations probabilistes — donne une
  probabilité (clipping `[0, 1]` en cas de léger débordement numérique) et
  une variance de krigeage. `pykrige.OrdinaryKriging` ou
  `gstools.krige.Ordinary`.
- **Krigeage indicateur** : conserver pour comparaison historique en
  binarisant les observations à un seuil (par ex. `obs > 0.5`).
- **Krigeage universel** avec dérive = distance au bord (pour intégrer
  explicitement l'effet de bordure documenté en littérature).
- **Co-krigeage** (optionnel) : si une covariable auxiliaire est disponible (par
  ex. NDVI simulé corrélé), exploiter la corrélation croisée.

Sortie : carte 2D de probabilité + carte 2D de variance/écart-type.

### 7.5 Modèles de treillis et MRF (`lattice.py`) — **méthode-clé du projet**

C'est l'« approche contextuelle markovienne » mentionnée dans la thèse.

- **Construction du graphe** : grille (rook = 4-voisins, queen = 8-voisins), ou
  KNN sur le pas spatial.
- **CAR (Conditional Autoregressive)** : implémenter via PyMC (`pm.CAR` n'existe
  pas natif → coder via prior gaussien à précision creuse). Variante Besag,
  Besag-York-Mollié (BYM).
- **SAR (Simultaneous Autoregressive)** : via `spreg` pour la version fréquentiste.
- **Modèle d'Ising binaire** : le champ latent `y_i ∈ {0, 1}` reste binaire
  (présence/absence par plant) ; les observations probabilistes des capteurs
  sont gérées par un modèle de mesure Binomial.
  - Énergie locale a priori : `E(y) = -α Σ y_i - β Σ_{i~j} y_i y_j`.
  - Modèle de mesure aux capteurs : `K * obs_i ~ Binomial(K, p_i)` où `p_i`
    est une fonction de `y_i` (par ex. `p = (1−q_0) * y + q_0 * (1−y)` pour
    intégrer un taux de bruit `q_0`) — alternative simple : utiliser
    `obs_i` comme étant `p_i` directement (capteur idéal).
  - Estimation des paramètres `(α, β, q_0)` par **pseudo-vraisemblance**
    (Besag) sur les capteurs, en traitant les voisins non observés comme
    manquants.
  - **Inférence** : échantillonneur de Gibbs conditionné sur les capteurs
    observés pour produire `P(y_s = 1 | obs)` en chaque cellule.
  - Implémentation custom NumPy/Numba (la grille fait 100k cellules,
    attention à la performance — utiliser des updates vectorisés en
    damier).
- **Modèle de Potts** (généralisation à K classes) — optionnel, utile si on
  étend à classes (aptère/ailé/coccinelle).

Cette section est la plus volumineuse à coder. Prévoir des tests sur petite grille
(20×20) avec paramètres connus pour valider la convergence.

### 7.6 Modèles bayésiens hiérarchiques (`hierarchical.py`)

- **GLMM Binomial avec effet aléatoire spatial Matérn** via PyMC :
  ```
  logit(p_i) = β0 + β1 * d_bord_i + W_i
  W ~ MvNormal(0, K_Matern)
  k_i ~ Binomial(K, p_i)         # k_i = K * obs_i, K = n_observations
  ```
  Si `n_observations is None` (capteur idéal), utiliser une vraisemblance
  Beta ou Normale tronquée sur `obs_i` directement.
- Inférence par NUTS sur les capteurs ; prédiction par krigeage gaussien postérieur
  sur la grille complète.
- Diagnostic via ArviZ (R-hat, ESS, trace plots).
- **Variante SPDE-like** : approximation par éléments finis si la grille fine
  rend le full Matérn trop coûteux. Optionnel, documenter.

### 7.7 Processus gaussien (`gp.py`)

- **GP régression** sur les observations probabilistes (sortie continue dans
  `[0, 1]`) avec noyau Matérn : `sklearn.gaussian_process.GaussianProcessRegressor`
  avec clipping. Avantage : exploite directement la richesse de `obs ∈ [0, 1]`.
- **GP classifier** : `sklearn.gaussian_process.GaussianProcessClassifier` sur
  `obs > 0.5` binarisé — à conserver comme comparaison avec l'ancien pipeline
  binaire.
- Alternative : `GPy` pour plus de flexibilité (noyaux composites, anisotropie,
  vraisemblance Binomial pour traiter `obs` comme `k/K`).
- Sortie : probabilité prédite + variance approchée.

### 7.8 Apprentissage automatique (`ml.py`)

- **Random Forest** avec features `(x, y, distance au bord, distance au capteur le
  plus proche, valeur du capteur le plus proche)`. Baseline non-spatiale-mais-
  géo-aware.
- **Régression Random Forest spatiale** : entraînement sur capteurs, prédiction
  partout. À considérer comme baseline pragmatique.

### 7.9 SADIE (`sadie.py`)

Méthode entomologique classique (Perry 1995) — implémentation simplifiée :

- L'indice d'agrégation original repose sur une **distance de rearrangement**
  (transport optimal vers la régularité). Pour rester en Python pur sans
  dépendance optimal-transport, on substitue une **mesure de concentration au
  barycentre pondéré** : la distance moyenne pondérée des points au barycentre
  des valeurs. La logique reste la même : comparer la statistique observée à
  une distribution sous permutation aléatoire des valeurs sur les positions
  fixes.
- Identifier *clusters* (`v_i > 0`) et *gaps* (`v_i < 0`) localement via un
  z-score (`(obs - mean) / std`) — version la plus simple.
- Plus exploratoire que prédictif ; pour l'interface `SpatialMethod`, on
  fournit en plus une interpolation **inverse-distance-weighted** (IDW) sur
  les capteurs, ce qui donne une carte continue par défaut.
- Les résultats SADIE (statistiques globales et `v_i`) sont accessibles via
  les propriétés `stats` et `v_local` après `fit`.

## 8. Phase 4 — Évaluation et comparaison

### 8.1 Métriques (`evaluation/metrics.py`)

Toutes calculées sur la grille complète (100k plants), en comparant la prédiction
de chaque méthode à la vérité terrain simulée :

- **AUC ROC** (sur `y` vrai vs `p̂`).
- **AUC PR** (mieux adapté au déséquilibre de classes attendu).
- **Brier score** : `mean((p̂ - y)²)`.
- **Log-loss** : `-mean(y log p̂ + (1-y) log(1-p̂))`.
- **CRPS** (continuous ranked probability score) : pour comparer les distributions
  prédictives complètes quand elles sont disponibles.
- **MAE / RMSE** sur la probabilité vraie `p` (pas seulement la présence binaire).
- **Calibration** : courbe de fiabilité (binning des p̂, comparaison à la fréquence
  empirique).
- **Couverture des intervalles de crédibilité** : pour les méthodes bayésiennes.

### 8.2 Validation croisée spatiale (`evaluation/cv.py`)

Comme on a la vérité terrain simulée, on peut évaluer directement. Mais pour une
évaluation comparable au cas réel :

- **Leave-one-sensor-out** : retirer chaque capteur, prédire son observation,
  comparer.
- **Spatial K-fold** : blocs spatiaux pour éviter la fuite d'information.

### 8.3 Plan de comparaison

Pour chaque combinaison `(seed_simulation, schéma_placement, n_sensors, méthode)` :

1. Simuler le champ.
2. Placer les capteurs.
3. Ajuster la méthode.
4. Prédire sur la grille.
5. Calculer les métriques.

Stocker dans un DataFrame long pour analyses agrégées. Répéter sur ~30-50 seeds
pour estimer la variance des performances.

## 9. Phase 5 — Visualisation

### 9.1 Cartes (`visualization/maps.py`)

Pour chaque méthode et chaque exécution :

- Carte de la **probabilité vraie** (référence).
- Carte de la **probabilité prédite**.
- Carte de l'**incertitude** (écart-type, IC95) quand disponible.
- Carte de l'**erreur** `p̂ - p`.
- Position des capteurs et leur observation (cercles colorés).

Utiliser une palette divergente pour l'erreur, séquentielle pour les probabilités,
ColorBrewer-safe (ex. `viridis`, `RdBu_r`).

### 9.2 Diagnostics (`visualization/diagnostics.py`)

- Variogrammes empiriques + modèles ajustés.
- Courbes de calibration (reliability diagram).
- Box-plots des métriques par méthode.
- Heatmap performance par `(méthode, schéma_placement)`.

## 10. Conventions de code

- **Style** : `ruff` (avec règles E, F, I, N, UP, B, SIM, RUF). Les règles
  `RUF001/002/003/046` sont ignorées globalement (faux positifs sur les
  caractères mathématiques `σ`, `μ`, `×` et sur `int(np.sqrt(...))`).
- **Typage** : annotations partout. `mypy --strict` est trop lourd avec les
  stubs NumPy ; on utilise un mode relâché (`check_untyped_defs`,
  `warn_unused_ignores`, `warn_redundant_casts`, `no_implicit_optional`)
  qui garde les annotations comme documentation et attrape les erreurs
  vraies.
- **Docstrings** : style NumPy.
- **Tests** : `pytest`, couverture cible >80% sur `src/aphid_spatial/simulation/`
  et `evaluation/`.
- **Reproductibilité** : tout module qui contient de l'aléa accepte un `seed`
  ou un `numpy.random.Generator`.
- **Performance** : grille de 100k plants → privilégier opérations vectorisées
  NumPy ; `numba` pour les boucles d'inférence MCMC custom (Ising notamment).
- **Logging** : `logging` standard avec niveaux INFO/DEBUG ; pas de `print` en
  bibliothèque.

## 11. Tests

### Tests minimaux

- `test_simulation.py` :
  - dimensions correctes,
  - reproductibilité du seed,
  - prévalence empirique cohérente avec `base_prevalence` (à effets près),
  - variogramme retrouve approximativement la portée Matérn,
  - effet de bordure visible (prévalence bord > centre).
- `test_sensors.py` :
  - nombre exact de capteurs,
  - pas de doublons,
  - schémas distincts produisent des distributions différentes.
- `test_methods.py` :
  - chaque méthode ajuste sans crash sur un mini-champ (10×100, 5 capteurs),
  - sortie dans [0,1],
  - dimensions correctes.
- `test_metrics.py` :
  - cas dégénérés (toutes prédictions = vraies présences),
  - équivalence avec `sklearn` quand applicable.

## 12. Ordre d'implémentation recommandé

Les phases 1 et 2 sont prérequis ; ensuite, attaquer les méthodes par ordre
croissant de complexité d'implémentation :

1. ✅ **Phase 1** — Simulation du champ + visualisation de base.
2. ✅ **Phase 2** — Placement (5 schémas) + observation probabiliste Binomial.
3. ✅ **Évaluation minimale** — métriques et `BaselineConstant`.
4. **Méthodes simples** :
   - ✅ exploration (descriptifs + baseline) ;
   - ⏳ autocorrélation (Moran, LISA) — nécessite `libpysal` + `esda`.
5. ✅ **Géostatistique** — variogramme + `OrdinaryKrigingIndicator`,
   `UniversalKrigingEdge`, `IndicatorKrigingThreshold`.
6. ⏳ **Processus ponctuels** — Ripley K + tests CSR (nécessite `pointpats`).
7. ✅ **GP classifier + GP régression** (baseline ML rapide).
8. ✅ **Random Forest** géo-aware.
9. ⏳ **MRF / Ising** — méthode-clé thèse, prévoir 2 itérations (V1 NumPy,
   V2 optimisée Numba).
10. ⏳ **GLMM bayésien** — prévoir long temps de calcul (PyMC).
11. ⏳ **CAR/SAR/BYM** — variantes lattice.
12. ✅ **SADIE** — version simplifiée (concentration + IDW + permutations).
13. ⏳ **Comparaison globale** — script orchestrateur + figures finales
    (`07_comparison.ipynb`). Le notebook `06_ml_methods.ipynb` couvre déjà
    la comparaison des méthodes implémentées.

Légende : ✅ implémenté · ⏳ à faire dans un round ultérieur.

## 13. Livrables attendus

- Bibliothèque Python installable (`pip install -e .`).
- Suite de tests passants.
- 7 notebooks didactiques (un par grande étape).
- Rapport `outputs/results/comparison.csv` avec toutes les métriques.
- Figures finales en PDF/PNG dans `outputs/figures/`.
- README expliquant comment reproduire les résultats principaux en 1 commande.

## 14. Pièges connus à anticiper

- **Krigeage sur observations probabilistes** : peut produire des prédictions
  légèrement hors `[0, 1]` (effet numérique, beaucoup moins marqué qu'avec une
  variable binaire) → clipper et documenter le pourcentage clippé.
- **Variogramme avec 20 points** : moins instable qu'en binaire grâce à
  l'observation graduée, mais reste sensible. Augmenter en agrégeant plusieurs
  séries temporelles (dates de déplacement des capteurs) si on veut un variogramme
  fiable.
- **MRF / Ising convergence** : tester d'abord sur 20×20 avec paramètres connus.
  Les cycles de Gibbs sur 100k cellules sont coûteux → updates en damier (cells
  noires puis blanches en parallèle).
- **PyMC NUTS sur 20 obs** : si on déclare 100k effets latents, c'est non
  scalable. Astuce : prédire seulement aux capteurs pour fitter, puis krigeage
  postérieur conditionnel pour prédire sur la grille.
- **Effet de bordure en covariable** : ne pas double-compter (si `f_edge` est dans
  la simulation, l'inclure aussi en covariable du modèle d'inférence est légitime
  et attendu, mais documenter le scénario où on l'ignore pour voir la robustesse).
- **Capteurs avec `obs ≈ 0` partout** : dans les régions à très faible prévalence,
  un capteur avec K=50 a une probabilité non négligeable de retourner exactement 0
  même si `prob > 0`. La variance des observations chute alors et le krigeage
  retombe sur la baseline constante. Augmenter K (ou utiliser
  `n_observations=None` pour la baseline théorique idéale) pour tester la
  sensibilité.
- **Choix de `K = n_observations`** : trop petit → bruit binomial domine la
  variabilité spatiale ; trop grand → information saturée et coûteuse à obtenir
  en pratique. Faire varier K dans `{10, 30, 50, 100, 1000}` pour quantifier le
  compromis information/coût.

## 15. Extensions possibles

- **Conception spatiale optimale** : où placer le 21ᵉ capteur ? Maximiser la
  réduction d'entropie postérieure du champ latent (active learning spatial).
- **Données temporelles** : si les capteurs sont déplacés, exploiter la
  dynamique (modèle spatio-temporel, GP avec kernel produit espace×temps).
- **Calibration sur données réelles** : remplacer la simulation par les
  données été 2024 / 2026 quand disponibles.
- **Comparaison avec INLA-SPDE** via `rpy2` pour valider les résultats PyMC.

---

**Démarrage rapide pour Claude Code** : commencer par `Phase 1` (simulation),
implémenter la visualisation de base, puis itérer méthode par méthode dans l'ordre
de la section 12. Chaque méthode doit être livrée avec son notebook d'illustration,
ses tests, et au moins une figure dans `outputs/figures/`.
