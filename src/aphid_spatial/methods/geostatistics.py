"""Méthodes géostatistiques : variogrammes et krigeage.

Ce module fournit pour l'instant la classe :class:`OrdinaryKrigingIndicator`,
qui ajuste un variogramme empirique sur les observations binaires des
capteurs puis effectue un krigeage ordinaire pour prédire la probabilité
de présence en tout point du champ.

Pièges connus (CLAUDE.md §14) :

* Le krigeage sur indicateur peut produire des valeurs hors ``[0, 1]`` ;
  on les clippe et on logge le pourcentage clippé.
* Avec ~20 capteurs, le variogramme empirique est instable. On limite
  donc ``n_lags`` à une valeur raisonnable et on retombe sur une baseline
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


__all__ = ["OrdinaryKrigingIndicator", "VariogramModel"]
