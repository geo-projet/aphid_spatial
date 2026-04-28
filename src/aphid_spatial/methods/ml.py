"""Méthodes d'apprentissage automatique « géo-aware ».

:class:`SpatialRandomForest` entraîne un :class:`RandomForestRegressor` sur les
observations probabilistes des capteurs avec des features dérivées :
``(x, y, distance au bord, distance au capteur le plus proche, valeur du
capteur le plus proche)``. C'est une baseline pragmatique qui n'utilise
aucun modèle de covariance explicite — elle apprend purement des données.

Pièges :

* Avec ~20 capteurs c'est très peu pour un RF ; on garde donc des
  hyperparamètres modestes (peu d'arbres, profondeur limitée) pour
  éviter le surapprentissage.
* Les features ``dist_capteur`` et ``valeur_capteur`` valent 0 pour les
  capteurs eux-mêmes (auto-lookup) — c'est volontaire, ça rend les
  capteurs reconnaissables et le RF apprend à les fitter parfaitement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from aphid_spatial.methods.exploration import edge_distance
from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)


def _build_features(
    coords: NDArray[np.floating],
    edge_dist_at_coords: NDArray[np.floating],
    sensor_coords: NDArray[np.floating],
    sensor_obs: NDArray[np.floating],
    self_lookup_idx: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Construit la matrice de features pour le RF.

    Pour le voisin le plus proche :

    * Si ``self_lookup_idx is None`` (cas grille de prédiction), on prend le
      capteur le plus proche normal ;
    * Sinon (cas entraînement aux capteurs eux-mêmes), on retire le
      « self-match » pour éviter une feature triviale.
    """
    tree = cKDTree(sensor_coords)
    if self_lookup_idx is None:
        dist, idx = tree.query(coords, k=1)
        sensor_value = sensor_obs[idx]
    else:
        # Pour chaque capteur, on demande les 2 plus proches et on retient le 2ᵉ
        dist_pair, idx_pair = tree.query(coords, k=min(2, sensor_coords.shape[0]))
        if idx_pair.ndim == 1:
            # Un seul capteur : pas de voisin, on met 0
            dist = np.zeros(coords.shape[0])
            sensor_value = np.zeros(coords.shape[0])
        else:
            dist = dist_pair[:, 1]
            sensor_value = sensor_obs[idx_pair[:, 1]]
    return np.column_stack(
        [
            coords[:, 0].astype(np.float64),
            coords[:, 1].astype(np.float64),
            edge_dist_at_coords.astype(np.float64),
            dist.astype(np.float64),
            sensor_value.astype(np.float64),
        ]
    )


@dataclass
class SpatialRandomForest:
    """RandomForestRegressor avec features ``(x, y, d_bord, d_capteur, val_capteur)``.

    Parameters
    ----------
    n_estimators : int
        Nombre d'arbres.
    max_depth : int | None
        Profondeur max des arbres. ``None`` = arbres complets (risque de
        surapprentissage avec peu de données).
    min_samples_leaf : int
        Taille min d'une feuille.
    """

    n_estimators: int = 200
    max_depth: int | None = 8
    min_samples_leaf: int = 1
    random_state: int = 0

    name: str = field(default="spatial_random_forest")
    _model: RandomForestRegressor | None = field(default=None, init=False, repr=False)
    _readings: SensorReadings | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)
    _edge_dist_grid: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _fallback_value: float | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        self._readings = readings
        self._field = field_meta
        self._edge_dist_grid = edge_distance(field_meta)

        x_sensors = readings.coords.astype(np.float64)
        y_sensors = readings.obs.astype(np.float64)

        if x_sensors.shape[0] < 2 or float(y_sensors.var()) == 0.0:
            self._fallback_value = float(y_sensors.mean()) if y_sensors.size else 0.0
            self._model = None
            logger.warning(
                "SpatialRandomForest fallback (n=%d) -> constante=%.3f",
                x_sensors.shape[0],
                self._fallback_value,
            )
            return

        # Distance au bord aux positions des capteurs (lookup via index)
        edge_at_sensors = self._edge_dist_grid[readings.sensor_idx]
        feats = _build_features(
            x_sensors,
            edge_at_sensors,
            x_sensors,
            y_sensors,
            self_lookup_idx=readings.sensor_idx,
        )
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(feats, y_sensors)
        self._model = rf
        self._fallback_value = None
        logger.info(
            "SpatialRandomForest fit OK : n_estim=%d, oob_score=%.3f",
            self.n_estimators,
            float(rf.score(feats, y_sensors)),
        )

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._model is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)
        assert self._readings is not None and self._field is not None
        assert self._edge_dist_grid is not None

        # Pour les query_coords arbitraires, recalcul de la distance au bord
        cfg = self._field.config
        x_max = (cfg.n_cols - 1) * cfg.spacing_m
        y_max = (cfg.n_rows - 1) * cfg.spacing_m
        dx = np.minimum(query_coords[:, 0], x_max - query_coords[:, 0])
        dy = np.minimum(query_coords[:, 1], y_max - query_coords[:, 1])
        edge_at_q = np.minimum(dx, dy)

        feats = _build_features(
            query_coords.astype(np.float64),
            edge_at_q.astype(np.float64),
            self._readings.coords.astype(np.float64),
            self._readings.obs.astype(np.float64),
            self_lookup_idx=None,
        )
        pred = self._model.predict(feats)
        return np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        """Écart-type empirique des prédictions des arbres individuels."""
        if self._model is None:
            return None
        assert self._readings is not None and self._field is not None

        cfg = self._field.config
        x_max = (cfg.n_cols - 1) * cfg.spacing_m
        y_max = (cfg.n_rows - 1) * cfg.spacing_m
        dx = np.minimum(query_coords[:, 0], x_max - query_coords[:, 0])
        dy = np.minimum(query_coords[:, 1], y_max - query_coords[:, 1])
        edge_at_q = np.minimum(dx, dy)
        feats = _build_features(
            query_coords.astype(np.float64),
            edge_at_q.astype(np.float64),
            self._readings.coords.astype(np.float64),
            self._readings.obs.astype(np.float64),
            self_lookup_idx=None,
        )
        # Stack des prédictions de chaque arbre : (n_estimators, n_query)
        per_tree = np.stack([est.predict(feats) for est in self._model.estimators_])
        return np.asarray(per_tree.std(axis=0), dtype=np.float64)


__all__ = ["SpatialRandomForest"]
