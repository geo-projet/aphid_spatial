"""SADIE (Spatial Analysis by Distance IndicEs) — version simplifiée.

D'après Perry (1995). On calcule deux quantités principales :

* **Indice d'agrégation global** ``I_a`` : comparaison d'une mesure de
  concentration spatiale des valeurs (distance pondérée des points au
  barycentre des valeurs) à sa distribution sous permutations aléatoires
  des valeurs sur les positions fixes.

  - ``I_a < 1`` → valeurs plus concentrées qu'au hasard (agrégation),
  - ``I_a ≈ 1`` → comparable à du random,
  - ``I_a > 1`` → valeurs plus dispersées qu'au hasard (régularité).

* **Indices locaux** ``v_i`` : z-score local de chaque capteur ;
  ``v_i > 0`` indique un capteur dans un *cluster* (valeur au-dessus de
  la moyenne), ``v_i < 0`` dans un *gap*.

CLAUDE.md §7.9 indique que SADIE est plus exploratoire que prédictif. La
classe :class:`SADIE` expose néanmoins l'interface ``SpatialMethod`` :
``predict_proba`` retourne une interpolation par distance inverse pondérée
des observations des capteurs (méthode rapide et raisonnable comme
référence).

Cette implémentation est simplifiée par rapport à la formulation originale
de Perry (qui utilise une distance de rearrangement pour atteindre la
régularité ; en Python pur, on évite la résolution de transport optimal
complexe et on retient la mesure de concentration au barycentre, suffisante
pour le diagnostic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)


def _concentration_metric(
    coords: NDArray[np.floating], values: NDArray[np.floating]
) -> float:
    """Distance moyenne pondérée des points au barycentre des valeurs.

    Plus c'est faible, plus les valeurs sont spatialement concentrées.
    """
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    cx = float((coords[:, 0] * values).sum() / total)
    cy = float((coords[:, 1] * values).sum() / total)
    d = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return float((d * values).sum() / total)


def aggregation_index(
    coords: NDArray[np.floating],
    values: NDArray[np.floating],
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Calcule un indice d'agrégation SADIE simplifié + p-value bilatérale.

    Returns
    -------
    dict avec :
        - ``I_a`` : ratio ``mean(perm_metric) / obs_metric``
          (plus grand = plus agrégé qu'au hasard)
        - ``p_value`` : proportion de permutations donnant un métrique
          ≤ ``obs_metric`` (test unilatéral d'agrégation)
        - ``obs_metric``, ``perm_mean``, ``perm_std``
    """
    obs_metric = _concentration_metric(coords, values)
    rng = np.random.default_rng(seed)
    perm_metrics = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(values)
        perm_metrics[i] = _concentration_metric(coords, perm)

    perm_mean = float(perm_metrics.mean())
    perm_std = float(perm_metrics.std())
    eps = 1e-12
    i_a = perm_mean / max(obs_metric, eps)
    # Test unilatéral d'agrégation (obs_metric < perm_metrics signifie
    # plus agrégé que random)
    p_value = float((perm_metrics <= obs_metric).mean())
    return {
        "I_a": i_a,
        "p_value": p_value,
        "obs_metric": obs_metric,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
    }


def local_indices(
    coords: NDArray[np.floating], values: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Z-score local de chaque capteur.

    ``v_i = (values_i - mean) / std`` ; positif = cluster local,
    négatif = gap. Si ``std == 0``, retourne un vecteur de zéros.
    """
    del coords  # gardé pour signature future ; non utilisé en simple z-score
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std())
    if sd <= 0.0:
        return np.zeros_like(arr)
    return ((arr - arr.mean()) / sd).astype(np.float64)


@dataclass
class SADIE:
    """SADIE simplifié + interpolation IDW pour l'interface ``SpatialMethod``.

    Parameters
    ----------
    n_permutations : int
        Nombre de permutations pour estimer la distribution sous hypothèse
        nulle (random labelling).
    idw_power : float
        Exposant de la pondération inverse-distance pour ``predict_proba``.
    idw_eps : float
        Distance plancher pour éviter les divisions par zéro.
    seed : int
        Graine pour les permutations.
    """

    n_permutations: int = 200
    idw_power: float = 2.0
    idw_eps: float = 1e-3
    seed: int = 0

    name: str = field(default="sadie_simplified")
    _readings: SensorReadings | None = field(default=None, init=False, repr=False)
    _stats: dict[str, float] | None = field(default=None, init=False, repr=False)
    _v_local: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        del field_meta
        self._readings = readings
        self._stats = aggregation_index(
            readings.coords, readings.obs, self.n_permutations, self.seed
        )
        self._v_local = local_indices(readings.coords, readings.obs)
        logger.info(
            "SADIE fit : I_a=%.3f, p=%.3f, n_clusters=%d, n_gaps=%d",
            self._stats["I_a"],
            self._stats["p_value"],
            int((self._v_local > 0).sum()),
            int((self._v_local < 0).sum()),
        )

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        """Interpolation IDW (inverse-distance-weighted) des observations."""
        assert self._readings is not None, "fit() avant predict_proba()"
        readings = self._readings
        if readings.obs.size == 0:
            return np.zeros(query_coords.shape[0], dtype=np.float64)
        if readings.obs.size == 1:
            return np.full(
                query_coords.shape[0], float(readings.obs[0]), dtype=np.float64
            )

        # Calcul direct (n_query × n_capteurs) ; OK tant que n_capteurs ~ 20
        diff = query_coords[:, None, :] - readings.coords[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        # Si une query coïncide exactement avec un capteur, retourner sa valeur
        weights = 1.0 / np.maximum(dist, self.idw_eps) ** self.idw_power
        norm = weights.sum(axis=1, keepdims=True)
        out = (weights * readings.obs[None, :]).sum(axis=1) / norm.ravel()
        return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        """IDW ne fournit pas d'incertitude propre. On approxime par la
        distance au capteur le plus proche normalisée."""
        assert self._readings is not None
        if self._readings.obs.size == 0:
            return None
        tree = cKDTree(self._readings.coords)
        dist, _ = tree.query(query_coords, k=1)
        d = np.asarray(dist, dtype=np.float64)
        # Normalisation simple : facteur ∝ distance, plafonné à 1
        return np.minimum(d / max(d.max(), 1e-9), 1.0)

    @property
    def stats(self) -> dict[str, float]:
        assert self._stats is not None, "fit() avant lecture des stats"
        return dict(self._stats)

    @property
    def v_local(self) -> NDArray[np.float64]:
        assert self._v_local is not None, "fit() avant lecture des v_local"
        return self._v_local


__all__ = ["SADIE", "aggregation_index", "local_indices"]
