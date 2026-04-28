"""Méthodes géostatistiques : variogrammes et krigeage.

Trois variantes sont fournies, toutes implémentant l'interface
:class:`~aphid_spatial.methods.SpatialMethod` :

* :class:`OrdinaryKrigingIndicator` — krigeage ordinaire sur les
  observations probabilistes (sans dérive). Référence simple et rapide.
* :class:`UniversalKrigingEdge` — krigeage universel avec dérive linéaire
  basée sur la distance au bord du champ, pour intégrer explicitement
  l'effet de bordure documenté en littérature.
* :class:`IndicatorKrigingThreshold` — krigeage indicateur en binarisant
  les observations à un seuil ``threshold`` (par défaut 0.5). Conserve
  l'esprit du « krigeage sur binaire » historique.

Pièges connus (CLAUDE.md §14) :

* Le krigeage peut produire des valeurs hors ``[0, 1]`` ; on les clippe et
  on logge le pourcentage clippé.
* Avec ~20 capteurs, le variogramme empirique est instable. On limite donc
  ``n_lags`` à une valeur raisonnable et on retombe sur une baseline
  constante si le variogramme dégénère.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)

VariogramModel = Literal[
    "linear", "power", "gaussian", "spherical", "exponential", "hole-effect"
]


@dataclass
class OrdinaryKrigingIndicator:
    """Krigeage ordinaire sur la variable indicateur ``obs ∈ {0, 1}``.

    Parameters
    ----------
    variogram_model : str
        Modèle de variogramme passé à ``pykrige.OrdinaryKriging``
        (par défaut ``"exponential"``, robuste sur peu de points).
    n_lags : int
        Nombre de classes de distance pour l'estimation du variogramme.
        Limité par défaut à 8 car ~20 capteurs ne supportent pas plus.
    nugget : float | None
        Effet pépite forcé. ``None`` laisse pykrige l'estimer.
    weight : bool
        Si ``True``, pondère l'ajustement du variogramme par le nombre de
        paires dans chaque lag (recommandé sur peu de points).
    fallback_warn_threshold : float
        Si plus que cette fraction de prédictions doit être clippée, un
        warning est émis (signal d'inadéquation du modèle).
    """

    variogram_model: VariogramModel = "exponential"
    n_lags: int = 8
    nugget: float | None = None
    weight: bool = True
    fallback_warn_threshold: float = 0.10

    name: str = field(default="ordinary_kriging_indicator")

    # État après fit
    _kriger: OrdinaryKriging | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)
    _readings: SensorReadings | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        """Ajuste le variogramme et prépare le système de krigeage.

        Si l'ajustement n'est pas possible (≤ 2 capteurs ou variance des
        observations nulle), la méthode bascule sur une baseline constante
        égale à la prévalence empirique.
        """
        del field_meta  # non utilisé pour l'OK simple ; gardé pour l'interface
        self._readings = readings
        x = readings.coords[:, 0].astype(np.float64)
        y = readings.coords[:, 1].astype(np.float64)
        z = readings.obs.astype(np.float64)

        n = z.size
        if n < 3 or float(z.var()) == 0.0:
            self._fallback_value = float(z.mean()) if n > 0 else 0.0
            self._kriger = None
            logger.warning(
                "OK fallback (n=%d, var=%g) -> probabilité constante=%.3f",
                n,
                float(z.var()) if n > 0 else 0.0,
                self._fallback_value,
            )
            return

        n_lags = max(2, min(self.n_lags, n - 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kriger = OrdinaryKriging(
                    x,
                    y,
                    z,
                    variogram_model=self.variogram_model,
                    nlags=n_lags,
                    weight=self.weight,
                    enable_plotting=False,
                    verbose=False,
                    exact_values=True,
                )
            self._fallback_value = None
            logger.info(
                "OK fit OK : n=%d, model=%s, nlags=%d, params=%s",
                n,
                self.variogram_model,
                n_lags,
                getattr(self._kriger, "variogram_model_parameters", None),
            )
        except Exception as exc:  # pragma: no cover - dépend des données
            self._fallback_value = float(z.mean())
            self._kriger = None
            logger.warning("OK fit a échoué (%s) -> baseline constante", exc)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        """Probabilité prédite en chaque point de ``query_coords`` (n_query, 2)."""
        if self._kriger is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)

        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, _ss = self._kriger.execute("points", xq, yq, backend="vectorized")
        z_arr = np.asarray(z, dtype=np.float64).ravel()
        clipped = (z_arr < 0.0) | (z_arr > 1.0)
        frac = float(clipped.mean())
        if frac > self.fallback_warn_threshold:
            logger.warning(
                "OK predict : %.1f%% des prédictions clippées dans [0,1]",
                100.0 * frac,
            )
        return np.clip(z_arr, 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        """Écart-type de krigeage en chaque point. ``None`` si fallback constant."""
        if self._kriger is None:
            return None
        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _z, ss = self._kriger.execute("points", xq, yq, backend="vectorized")
        ss_arr = np.asarray(ss, dtype=np.float64).ravel()
        # ss est la variance ; sécurité contre valeurs négatives numériques
        return np.sqrt(np.maximum(ss_arr, 0.0))


# ---------------------------------------------------------------------------
# Krigeage universel avec dérive distance-au-bord
# ---------------------------------------------------------------------------


def _edge_distance_at_coords(
    coords: NDArray[np.floating], field_meta: Field
) -> NDArray[np.float64]:
    """Distance au bord (m) pour des coordonnées arbitraires."""
    cfg = field_meta.config
    x_max = (cfg.n_cols - 1) * cfg.spacing_m
    y_max = (cfg.n_rows - 1) * cfg.spacing_m
    dx = np.minimum(coords[:, 0], x_max - coords[:, 0])
    dy = np.minimum(coords[:, 1], y_max - coords[:, 1])
    return np.minimum(dx, dy).astype(np.float64)


@dataclass
class UniversalKrigingEdge:
    """Krigeage universel avec dérive externe = distance au bord.

    L'effet de bordure documenté en littérature (Mackenzie & Vernon 1988,
    Severtson 2015) est intégré explicitement comme covariable. Le résidu
    après dérive est ensuite krigé ordinairement.

    Parameters
    ----------
    variogram_model : str
        Modèle de variogramme passé à ``pykrige.UniversalKriging``.
    n_lags : int
        Nombre de classes de distance pour l'estimation du variogramme.
    fallback_warn_threshold : float
        Si plus que cette fraction de prédictions est clippée, warning.
    """

    variogram_model: VariogramModel = "exponential"
    n_lags: int = 8
    fallback_warn_threshold: float = 0.10

    name: str = field(default="universal_kriging_edge")

    _kriger: UniversalKriging | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        self._field = field_meta
        x = readings.coords[:, 0].astype(np.float64)
        y = readings.coords[:, 1].astype(np.float64)
        z = readings.obs.astype(np.float64)

        if z.size < 4 or float(z.var()) == 0.0:
            self._fallback_value = float(z.mean()) if z.size else 0.0
            self._kriger = None
            logger.warning(
                "UK fallback (n=%d, var=%g) -> constante=%.3f",
                z.size,
                float(z.var()) if z.size else 0.0,
                self._fallback_value,
            )
            return

        # pykrige.UniversalKriging avec ``external_Z`` attend la dérive sur une
        # grille régulière 2D, qu'il interpole aux positions des capteurs.
        cfg = field_meta.config
        x_axis = np.arange(cfg.n_cols, dtype=np.float64) * cfg.spacing_m
        y_axis = np.arange(cfg.n_rows, dtype=np.float64) * cfg.spacing_m
        x_max = x_axis[-1]
        y_max = y_axis[-1]
        dx_grid = np.minimum(x_axis, x_max - x_axis)
        dy_grid = np.minimum(y_axis, y_max - y_axis)
        # Forme attendue par pykrige : (ny, nx)
        edge_grid = np.minimum(dy_grid[:, None], dx_grid[None, :]).astype(np.float64)

        n_lags = max(2, min(self.n_lags, z.size - 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kriger = UniversalKriging(
                    x,
                    y,
                    z,
                    variogram_model=self.variogram_model,
                    nlags=n_lags,
                    drift_terms=["external_Z"],
                    external_drift=edge_grid,
                    external_drift_x=x_axis,
                    external_drift_y=y_axis,
                    enable_plotting=False,
                    verbose=False,
                    exact_values=True,
                )
            # On garde la moyenne empirique comme valeur de repli en cas de
            # système singulier au moment du predict (placement "grid" surtout).
            self._fallback_value = float(z.mean())
            logger.info(
                "UK fit OK : n=%d, model=%s, params=%s",
                z.size,
                self.variogram_model,
                getattr(self._kriger, "variogram_model_parameters", None),
            )
        except Exception as exc:  # pragma: no cover - dépend des données
            self._fallback_value = float(z.mean())
            self._kriger = None
            logger.warning(
                "UK fit a échoué (%s) -> baseline constante", exc
            )

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._kriger is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)
        assert self._field is not None
        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        edge_q = _edge_distance_at_coords(query_coords, self._field)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z, _ss = self._kriger.execute(
                    "points",
                    xq,
                    yq,
                    backend="vectorized",
                    specified_drift_arrays=[edge_q],
                )
        except np.linalg.LinAlgError as exc:
            # Système de krigeage singulier (capteurs colinéaires sur la dérive,
            # cas typique du placement "grid"). On bascule sur la baseline.
            logger.warning("UK predict singular (%s) -> baseline constante", exc)
            return np.full(
                query_coords.shape[0],
                self._fallback_value if self._fallback_value is not None else 0.0,
                dtype=np.float64,
            )
        z_arr = np.asarray(z, dtype=np.float64).ravel()
        clipped = (z_arr < 0.0) | (z_arr > 1.0)
        frac = float(clipped.mean())
        if frac > self.fallback_warn_threshold:
            logger.warning(
                "UK predict : %.1f%% des prédictions clippées dans [0,1]",
                100.0 * frac,
            )
        return np.clip(z_arr, 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._kriger is None:
            return None
        assert self._field is not None
        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        edge_q = _edge_distance_at_coords(query_coords, self._field)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _z, ss = self._kriger.execute(
                    "points",
                    xq,
                    yq,
                    backend="vectorized",
                    specified_drift_arrays=[edge_q],
                )
        except np.linalg.LinAlgError:
            return None
        ss_arr = np.asarray(ss, dtype=np.float64).ravel()
        return np.sqrt(np.maximum(ss_arr, 0.0))


# ---------------------------------------------------------------------------
# Krigeage indicateur sur seuil
# ---------------------------------------------------------------------------


@dataclass
class IndicatorKrigingThreshold:
    """Krigeage ordinaire après binarisation de ``obs`` au seuil ``threshold``.

    Conserve l'esprit du « krigeage indicateur » historique pour comparaison
    avec :class:`OrdinaryKrigingIndicator`. La sortie est interprétée comme
    une probabilité ``P(obs > threshold)`` et clippée à ``[0, 1]``.
    """

    threshold: float = 0.5
    variogram_model: VariogramModel = "exponential"
    n_lags: int = 8

    name: str = field(default="indicator_kriging_threshold")

    _kriger: OrdinaryKriging | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        del field_meta
        x = readings.coords[:, 0].astype(np.float64)
        y = readings.coords[:, 1].astype(np.float64)
        z = (readings.obs > self.threshold).astype(np.float64)

        if z.size < 3 or float(z.var()) == 0.0:
            self._fallback_value = float(z.mean()) if z.size else 0.0
            self._kriger = None
            logger.warning(
                "IK fallback (n=%d, var=%g) -> constante=%.3f",
                z.size,
                float(z.var()) if z.size else 0.0,
                self._fallback_value,
            )
            return

        n_lags = max(2, min(self.n_lags, z.size - 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kriger = OrdinaryKriging(
                    x,
                    y,
                    z,
                    variogram_model=self.variogram_model,
                    nlags=n_lags,
                    enable_plotting=False,
                    verbose=False,
                    exact_values=True,
                )
            self._fallback_value = None
            logger.info(
                "IK fit OK (seuil=%.2f) : n=%d, model=%s",
                self.threshold,
                z.size,
                self.variogram_model,
            )
        except Exception as exc:  # pragma: no cover
            self._fallback_value = float(z.mean())
            self._kriger = None
            logger.warning("IK fit a échoué (%s)", exc)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._kriger is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)
        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, _ss = self._kriger.execute("points", xq, yq, backend="vectorized")
        return np.clip(np.asarray(z, dtype=np.float64).ravel(), 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._kriger is None:
            return None
        xq = query_coords[:, 0].astype(np.float64)
        yq = query_coords[:, 1].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _z, ss = self._kriger.execute("points", xq, yq, backend="vectorized")
        ss_arr = np.asarray(ss, dtype=np.float64).ravel()
        return np.sqrt(np.maximum(ss_arr, 0.0))


__all__ = [
    "IndicatorKrigingThreshold",
    "OrdinaryKrigingIndicator",
    "UniversalKrigingEdge",
    "VariogramModel",
]
