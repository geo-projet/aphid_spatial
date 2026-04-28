# aphid_spatial

Modélisation de la corrélation spatiale du puceron de la laitue
(*Nasonovia ribisnigri*) — composante géospatiale de la thèse d'Emma Dubrûle
(Université de Sherbrooke, Département de géomatique appliquée).

À partir des observations de ~20 capteurs fixes répartis sur un champ de
laitue — chaque capteur retournant une **probabilité de présence** dans
`[0, 1]` (fraction d'observations positives sur une fenêtre temporelle) —
on estime la probabilité de présence du puceron sur l'ensemble du champ
(des milliers de plants) en exploitant la corrélation spatiale.

Faute de données terrain, le module commence par une **simulation contrôlée** :
on génère un champ synthétique avec une vérité terrain connue (probabilité
continue + tirage Bernoulli), on y place des capteurs, puis on applique
différentes méthodes pour reconstruire la carte probabiliste à partir des
observations partielles.

## État actuel

**Toutes les méthodes ciblées (géostatistique, autocorrélation, processus
ponctuels, GP, RF, MRF/Ising, GLMM bayésien, CAR/BYM, SAR) sont
implémentées.**

Phases couvertes :

- **Phase 1** : simulation du champ (effet de bordure + GRF Matérn + foyers).
- **Phase 2** : 5 schémas de placement des capteurs (modèle d'observation
  probabiliste binomial).
- **Évaluation** : AUC ROC/PR, Brier, log-loss, MAE/RMSE, calibration.
- **Méthodes prédictives implémentées (13)** :
  - `BaselineConstant` — borne inférieure (prévalence empirique).
  - `OrdinaryKrigingIndicator` — krigeage ordinaire (référence).
  - `UniversalKrigingEdge` — krigeage universel avec dérive distance-au-bord.
  - `IndicatorKrigingThreshold` — krigeage sur ``obs > seuil``.
  - `MaternGPRegressor` / `MaternGPClassifier` — processus gaussiens.
  - `SpatialRandomForest` — RF géo-aware avec features dérivées.
  - `SADIE` — indices d'agrégation (Perry 1995, version simplifiée) +
    interpolation IDW.
  - `IsingMRF` — Markov Random Field V1 NumPy (Gibbs damier conditionné).
  - `MaternGLMM` — GLMM bayésien Binomial avec effet aléatoire spatial
    Matérn (PyMC NUTS + krigeage postérieur).
  - `CARModel` (Besag) / `BYMModel` (Besag-York-Mollié) — Conditional
    Autoregressive avec composante optionnelle iid (PyMC `pm.CAR`).
  - `SARLagModel` — Simultaneous Autoregressive Lag fréquentiste
    (`spreg.GM_Lag`).
- **Modules exploratoires** (statistiques, pas de prédiction directe) :
  - `methods.exploration` — distance au bord, helpers capteur proche,
    distances inter-capteurs.
  - `methods.autocorrelation` — Moran I, Geary c, Getis-Ord G, LISA,
    Gᵢ\* (libpysal + esda).
  - `methods.point_process` — Ripley K/L/g, enveloppe CSR Monte Carlo,
    KDE d'intensité (pointpats).
- **Visualisation** : cartes 2D (vérité, prédiction, incertitude, erreur).
- **Notebooks** : `01_simulation`, `02_exploration` (+ autocorr + Ripley),
  `03_geostatistics`, `04_lattice_mrf`, `05_hierarchical_bayes`,
  `06_ml_methods`, `07_comparison` (orchestrateur 13 méthodes × 5 schémas).

Améliorations possibles : V2 Ising Numba, formulation géostatistique
extensible de CAR/BYM, SAR error model, validation INLA-SPDE via `rpy2`.

## Installation

Python ≥ 3.11.

```bash
python -m venv .venv

# Activation (Windows / PowerShell)
.venv\Scripts\Activate.ps1
# Activation (Unix)
source .venv/bin/activate

pip install -e ".[dev]"
```

## Utilisation rapide

```python
from aphid_spatial.simulation import FieldConfig, simulate_field, SensorConfig, place_sensors
from aphid_spatial.methods.geostatistics import (
    OrdinaryKrigingIndicator,
    UniversalKrigingEdge,
)
from aphid_spatial.methods.gp import MaternGPRegressor
from aphid_spatial.methods.ml import SpatialRandomForest
from aphid_spatial.methods.hierarchical import MaternGLMM
from aphid_spatial.methods.lattice import CARModel, SARLagModel
from aphid_spatial.evaluation.metrics import evaluate_all

field = simulate_field(FieldConfig(seed=42))
# Chaque capteur effectue 50 mesures temporelles ; obs ∈ [0, 1] est k/50.
readings = place_sensors(
    field, SensorConfig(n_sensors=20, placement="uniform", n_observations=50)
)

# Ajuster et comparer plusieurs méthodes
for method in [
    OrdinaryKrigingIndicator(),
    UniversalKrigingEdge(),       # avec dérive = distance au bord
    MaternGPRegressor(),
    SpatialRandomForest(),
    SARLagModel(),
    MaternGLMM(n_draws=200, n_tune=200),  # bayésien (lent)
    CARModel(n_draws=200, n_tune=200),
]:
    method.fit(readings, field)
    p_pred = method.predict_proba(field.coords)
    metrics = evaluate_all(field.presence, p_pred, field.prob)
    print(f"{method.name:35s} AUC={metrics['auc_roc']:.3f} RMSE={metrics['rmse_prob']:.3f}")
```

### Modèle d'observation

Chaque capteur retourne une **probabilité de présence** dans `[0, 1]`, pas
une détection binaire. Concrètement, `n_observations` mesures Bernoulli
indépendantes de la probabilité locale `prob_local` (moyenne de `prob` sur
le voisinage `(2r+1)²`) sont effectuées, et l'observation est la fraction
de positifs `k / n_observations`. Avec `n_observations=None`, le capteur
retourne `prob_local` exact (cas idéal, sans bruit d'estimation).

`SensorReadings` expose deux tableaux distincts :

- `obs` (float64) — l'observation bruitée du capteur, dans `[0, 1]`. C'est
  ce que les méthodes d'estimation utilisent pour l'inférence.
- `prob_local` (float64) — la probabilité réelle locale (référence pour
  analyse) ; ne pas l'utiliser comme entrée des méthodes d'estimation.

`Field.presence` (binaire) reste disponible comme vérité terrain pour
calculer AUC, log-loss, Brier ; `Field.prob` permet de calculer MAE/RMSE
sur la probabilité vraie.

## Reproduire les résultats principaux

```bash
pytest                                       # 86 tests (-m "not slow" pour skip PyMC)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Les figures sont écrites dans `outputs/figures/` et les CSV de métriques
dans `outputs/results/`. Les notebooks `05_hierarchical_bayes.ipynb` et
`07_comparison.ipynb` sont longs (~10 min et ~30 min respectivement à
cause de PyMC NUTS).

## Résultats actuels

Comparaison des 13 méthodes implémentées sur le scénario par défaut
(champ 100×1000, 20 capteurs uniformes, K = 50 mesures temporelles,
seed 2024). Source : `outputs/results/07_comparison.csv`.

| Méthode                       | AUC ROC | AUC PR | Brier | RMSE p̂ | MAE p̂  |
|-------------------------------|--------:|-------:|------:|--------:|-------:|
| `sar_lag_spreg`               |   0.688 |  0.382 | 0.177 |   0.166 |  0.126 |
| `bym_pymc`                    |   0.685 |  0.395 | 0.169 |   0.140 |  0.102 |
| `universal_kriging_edge`      |   0.681 |  0.383 | 0.169 |   0.140 |  0.101 |
| `sadie_simplified`            |   0.680 |  0.397 | 0.169 |   0.139 |  0.104 |
| `car_pymc`                    |   0.675 |  0.396 | 0.169 |   0.141 |  0.105 |
| `matern_glmm_pymc`            |   0.675 |  0.372 | 0.170 |   0.144 |  0.102 |
| `ordinary_kriging_indicator`  |   0.672 |  0.379 | 0.170 |   0.142 |  0.105 |
| `gp_matern_regressor`         |   0.658 |  0.363 | 0.172 |   0.150 |  0.112 |
| `gp_matern_classifier`        |   0.636 |  0.355 | 0.178 |   0.169 |  0.127 |
| `indicator_kriging_threshold` |   0.630 |  0.366 | 0.197 |   0.219 |  0.167 |
| `spatial_random_forest`       |   0.622 |  0.349 | 0.175 |   0.160 |  0.115 |
| `ising_mrf_v1`                |   0.501 |  0.240 | 0.183 |   0.183 |  0.143 |
| `baseline_constant`           |   0.500 |  0.239 | 0.182 |   0.181 |  0.141 |

Les méthodes les mieux classées (SAR Lag, BYM, krigeage universel, SADIE,
CAR, GLMM Matérn) se tiennent dans une fourchette d'AUC très resserrée
(0.675–0.688). Toutes capturent l'autocorrélation spatiale + l'effet de
bordure. SAR Lag est en tête sur l'AUC mais le krigeage universel reste
meilleur sur RMSE/Brier (calibration plus fine). L'**Ising V1** reste
équivalent à la baseline (β ≈ 0 estimé par PL avec 20 capteurs).

Voir `04_lattice_mrf.ipynb` pour le diagnostic Ising,
`05_hierarchical_bayes.ipynb` pour les diagnostics PyMC (R-hat, ESS,
trace plots), `06_ml_methods.ipynb` pour la comparaison ML détaillée, et
`07_comparison.ipynb` pour la robustesse aux schémas de placement
(heatmap + box-plots).

## Tests

```bash
pytest
```

## Notebooks

```bash
jupyter notebook notebooks/
```

## Structure

```
aphid_spatial/
├── pyproject.toml
├── README.md
├── src/aphid_spatial/
│   ├── simulation/         # field.py, sensors.py
│   ├── methods/            # exploration, autocorrelation, point_process,
│   │                       # geostatistics, gp, ml, sadie,
│   │                       # lattice (Ising/CAR/BYM/SAR), hierarchical (GLMM)
│   ├── evaluation/         # metrics.py
│   └── visualization/      # maps.py
├── tests/
├── notebooks/              # 01_simulation, 02_exploration, 03_geostatistics,
│                           # 04_lattice_mrf, 05_hierarchical_bayes,
│                           # 06_ml_methods, 07_comparison
├── data/simulated/         # champs synthétiques (npz, gitignored)
└── outputs/
    ├── figures/            # PNG (gitignored)
    └── results/            # CSV de métriques (gitignored)
```
