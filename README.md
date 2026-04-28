# aphid_spatial

Modélisation de la corrélation spatiale du puceron de la laitue
(*Nasonovia ribisnigri*) — composante géospatiale de la thèse d'Emma Dubrûle
(Université de Sherbrooke, Département de géomatique appliquée).

À partir des observations de ~20 capteurs binaires fixes répartis sur un champ
de laitue, on estime la probabilité de présence du puceron sur l'ensemble du
champ (des milliers de plants) en exploitant la corrélation spatiale.

Faute de données terrain, le module commence par une **simulation contrôlée** :
on génère un champ synthétique avec une vérité terrain connue, on y place des
capteurs, puis on applique différentes méthodes pour reconstruire la carte
probabiliste à partir des observations partielles.

Voir `CLAUDE.md` pour la spécification détaillée.

## État actuel (round 1)

Phases couvertes :

- **Phase 1** : simulation du champ (effet de bordure + GRF Matérn + foyers).
- **Phase 2** : 5 schémas de placement des capteurs.
- **Évaluation** : métriques de base (AUC ROC/PR, Brier, log-loss, MAE/RMSE,
  calibration).
- **Méthode baseline** : krigeage ordinaire indicateur.
- **Visualisation** : cartes 2D (vérité, prédiction, incertitude, erreur).
- **Notebooks** : `01_simulation`, `02_exploration`, `03_geostatistics`.

Méthodes prévues pour les rounds suivants : autocorrélation (Moran/LISA),
processus ponctuels (Ripley K), GP, Random Forest, MRF/Ising, GLMM bayésien,
SADIE.

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
from aphid_spatial.methods.geostatistics import OrdinaryKrigingIndicator
from aphid_spatial.evaluation.metrics import evaluate_all

field = simulate_field(FieldConfig(seed=42))
# Chaque capteur effectue 50 mesures temporelles ; obs ∈ [0, 1] est k/50.
readings = place_sensors(
    field, SensorConfig(n_sensors=20, placement="uniform", n_observations=50)
)

method = OrdinaryKrigingIndicator()
method.fit(readings, field)
p_pred = method.predict_proba(field.coords)

print(evaluate_all(field.presence, p_pred, field.prob))
```

### Modèle d'observation

Chaque capteur retourne une **probabilité de présence** dans `[0, 1]`, pas
une détection binaire. Concrètement, `n_observations` mesures Bernoulli
indépendantes de la probabilité locale `prob_local` (moyenne de `prob` sur
le voisinage `(2r+1)²`) sont effectuées, et l'observation est la fraction
de positifs `k / n_observations`. Avec `n_observations=None`, le capteur
retourne `prob_local` exact (cas idéal, sans bruit d'estimation).

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
