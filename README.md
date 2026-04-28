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
- **Méthodes implémentées** :
  - `BaselineConstant` — borne inférieure (prévalence empirique).
  - `OrdinaryKrigingIndicator` — krigeage ordinaire (référence).
  - `UniversalKrigingEdge` — krigeage universel avec dérive distance-au-bord.
  - `IndicatorKrigingThreshold` — krigeage sur ``obs > seuil``.
  - `MaternGPRegressor` / `MaternGPClassifier` — processus gaussiens.
  - `SpatialRandomForest` — RF géo-aware avec features dérivées.
  - `SADIE` — indices d'agrégation (Perry 1995, version simplifiée) +
    interpolation IDW.
- **Helpers exploratoires** (``methods.exploration``) : distance au bord,
  distance/valeur du capteur le plus proche, distances inter-capteurs.
- **Visualisation** : cartes 2D (vérité, prédiction, incertitude, erreur).
- **Notebooks** : `01_simulation`, `02_exploration`, `03_geostatistics`,
  `06_ml_methods` (comparaison de toutes les méthodes).

Méthodes prévues pour les rounds suivants : autocorrélation (Moran/LISA via
``libpysal``/``esda``), processus ponctuels (Ripley K via ``pointpats``),
MRF/Ising (NumPy/Numba custom), GLMM bayésien (PyMC), CAR/SAR/BYM.

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
pytest                                    # 49 tests, doivent tous passer
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Les figures sont écrites dans `outputs/figures/` et un CSV de métriques
dans `outputs/results/`.

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
