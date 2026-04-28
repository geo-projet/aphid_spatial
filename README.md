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

Voir `CLAUDE.md` pour la spécification détaillée.

## État actuel

Phases couvertes :

- **Phase 1** : simulation du champ (effet de bordure + GRF Matérn + foyers).
- **Phase 2** : 5 schémas de placement des capteurs (modèle d'observation
  probabiliste binomial).
- **Évaluation** : AUC ROC/PR, Brier, log-loss, MAE/RMSE, calibration.
- **Méthodes prédictives implémentées** :
  - `BaselineConstant` — borne inférieure (prévalence empirique).
  - `OrdinaryKrigingIndicator` — krigeage ordinaire (référence).
  - `UniversalKrigingEdge` — krigeage universel avec dérive distance-au-bord.
  - `IndicatorKrigingThreshold` — krigeage sur ``obs > seuil``.
  - `MaternGPRegressor` / `MaternGPClassifier` — processus gaussiens.
  - `SpatialRandomForest` — RF géo-aware avec features dérivées.
  - `SADIE` — indices d'agrégation (Perry 1995, version simplifiée) +
    interpolation IDW.
  - `IsingMRF` — Markov Random Field V1 NumPy (Gibbs damier conditionné).
- **Modules exploratoires** (statistiques, pas de prédiction directe) :
  - `methods.exploration` — distance au bord, helpers capteur proche,
    distances inter-capteurs.
  - `methods.autocorrelation` — Moran I, Geary c, Getis-Ord G, LISA,
    Gᵢ\* (libpysal + esda).
  - `methods.point_process` — Ripley K/L/g, enveloppe CSR Monte Carlo,
    KDE d'intensité (pointpats).
- **Visualisation** : cartes 2D (vérité, prédiction, incertitude, erreur).
- **Notebooks** : `01_simulation`, `02_exploration` (avec autocorr + Ripley),
  `03_geostatistics`, `04_lattice_mrf`, `06_ml_methods`,
  `07_comparison` (orchestrateur 9 méthodes × 5 schémas).

Méthodes prévues pour les rounds suivants : GLMM bayésien (PyMC), CAR/BYM,
SAR fréquentiste (spreg), V2 optimisée Numba du MRF.

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
pytest                                    # 79 tests, doivent tous passer
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Les figures sont écrites dans `outputs/figures/` et les CSV de métriques
dans `outputs/results/`.

## Résultats actuels

Comparaison des 9 méthodes implémentées sur le scénario par défaut
(champ 100×1000, 20 capteurs uniformes, K = 50 mesures temporelles,
seed 2024) — extrait de `outputs/results/07_comparison.csv` :

| Méthode                       | AUC ROC | AUC PR | Brier | RMSE p̂ | MAE p̂  |
|-------------------------------|--------:|-------:|------:|--------:|-------:|
| `universal_kriging_edge`      |   0.681 |  0.383 | 0.169 |   0.140 |  0.101 |
| `sadie_simplified`            |   0.680 |  0.397 | 0.169 |   0.139 |  0.104 |
| `ordinary_kriging_indicator`  |   0.672 |  0.379 | 0.170 |   0.142 |  0.105 |
| `gp_matern_regressor`         |   0.658 |  0.363 | 0.172 |   0.150 |  0.112 |
| `gp_matern_classifier`        |   0.636 |  0.355 | 0.178 |   0.169 |  0.127 |
| `indicator_kriging_threshold` |   0.630 |  0.366 | 0.197 |   0.219 |  0.167 |
| `spatial_random_forest`       |   0.622 |  0.349 | 0.175 |   0.160 |  0.115 |
| `ising_mrf_v1`                |   0.501 |  0.240 | 0.183 |   0.183 |  0.143 |
| `baseline_constant`           |   0.500 |  0.239 | 0.182 |   0.181 |  0.141 |

Le **krigeage universel avec dérive distance-au-bord** arrive en tête —
l'effet de bordure documenté en littérature est bien capturé en
l'incorporant comme covariable. L'**Ising MRF V1** est presque équivalent
à la baseline constante : avec 20 capteurs sparses, la pseudo-vraisemblance
estime ``β ≈ 0`` (manque d'identifiabilité). Une V2 EM + Numba est prévue.

Voir `04_lattice_mrf.ipynb` pour la validation Ising sur petite grille,
`06_ml_methods.ipynb` pour la comparaison ML, et `07_comparison.ipynb`
pour la robustesse aux schémas de placement (heatmap + box-plots).

## Tests

```bash
pytest
```

## Notebooks

```bash
jupyter notebook notebooks/
```

## Structure

Voir `CLAUDE.md` section 3 pour l'arborescence complète.
