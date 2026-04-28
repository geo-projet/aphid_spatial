"""Exploration et statistiques descriptives.

Ce module fournit :

* :class:`BaselineConstant` — méthode triviale qui prédit la prévalence/observation
  empirique partout, à utiliser comme borne inférieure de comparaison.
* Helpers descriptifs : distance au bord, distance au capteur le plus proche,
  histogramme des distances inter-capteurs.

Tous les helpers sont autonomes (n'imposent pas l'API ``SpatialMethod``) et
retournent des arrays NumPy bruts pour faciliter l'usage en notebook.
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


# ---------------------------------------------------------------------------
# Méthode de référence : prédiction constante
# ---------------------------------------------------------------------------


@dataclass
class BaselineConstant:
    """Méthode triviale : prédit ``mean(obs)`` partout.

    Sert de référence absolue (« si on n'utilise aucune information spatiale »)
    pour quantifier le gain des autres méthodes.
    """

    name: str = field(default="baseline_constant")
    _value: float = field(default=0.0, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        del field_meta
        if readings.obs.size == 0:
            self._value = 0.0
        else:
            self._value = float(readings.obs.mean())
        logger.info("BaselineConstant fit : value=%.4f", self._value)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        return np.full(query_coords.shape[0], self._value, dtype=np.float64)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        del query_coords
        return None


# ---------------------------------------------------------------------------
# Statistiques descriptives
# ---------------------------------------------------------------------------


def edge_distance(field_meta: Field) -> NDArray[np.float64]:
    """Distance (m) au bord le plus proche pour chaque plant, en row-major.

    Utile comme covariable pour les méthodes ``geo-aware``.
    """
    cfg = field_meta.config
    spacing = cfg.spacing_m
    rows = np.arange(cfg.n_rows)
    cols = np.arange(cfg.n_cols)
    dy = np.minimum(rows, cfg.n_rows - 1 - rows) * spacing
    dx = np.minimum(cols, cfg.n_cols - 1 - cols) * spacing
    return np.minimum(dy[:, None], dx[None, :]).ravel().astype(np.float64)


def nearest_sensor_distance(
    readings: SensorReadings, query_coords: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Distance euclidienne au capteur le plus proche pour chaque ``query_coords``."""
    if readings.coords.shape[0] == 0:
        return np.full(query_coords.shape[0], np.inf, dtype=np.float64)
    tree = cKDTree(readings.coords)
    dist, _ = tree.query(query_coords, k=1)
    return np.asarray(dist, dtype=np.float64)


def nearest_sensor_value(
    readings: SensorReadings, query_coords: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Valeur ``obs`` du capteur le plus proche pour chaque ``query_coords``."""
    if readings.coords.shape[0] == 0:
        return np.full(query_coords.shape[0], 0.0, dtype=np.float64)
    tree = cKDTree(readings.coords)
    _dist, idx = tree.query(query_coords, k=1)
    return readings.obs[idx].astype(np.float64)


def inter_sensor_distances(readings: SensorReadings) -> NDArray[np.float64]:
    """Toutes les distances inter-capteurs (1D, longueur ``n*(n-1)/2``)."""
    coords = readings.coords
    n = coords.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return d[iu].astype(np.float64)


def descriptive_summary(
    readings: SensorReadings, field_meta: Field
) -> dict[str, float]:
    """Résumé numérique en une seule passe : prévalence, distances, etc."""
    inter = inter_sensor_distances(readings)
    nearest = nearest_sensor_distance(readings, field_meta.coords)
    return {
        "n_sensors": float(readings.obs.size),
        "obs_mean": float(readings.obs.mean()) if readings.obs.size else float("nan"),
        "obs_std": float(readings.obs.std()) if readings.obs.size else float("nan"),
        "obs_min": float(readings.obs.min()) if readings.obs.size else float("nan"),
        "obs_max": float(readings.obs.max()) if readings.obs.size else float("nan"),
        "inter_sensor_dist_min_m": float(inter.min()) if inter.size else float("nan"),
        "inter_sensor_dist_mean_m": float(inter.mean()) if inter.size else float("nan"),
        "inter_sensor_dist_max_m": float(inter.max()) if inter.size else float("nan"),
        "coverage_dist_mean_m": float(nearest.mean()),
        "coverage_dist_max_m": float(nearest.max()),
    }


__all__ = [
    "BaselineConstant",
    "descriptive_summary",
    "edge_distance",
    "inter_sensor_distances",
    "nearest_sensor_distance",
    "nearest_sensor_value",
]
