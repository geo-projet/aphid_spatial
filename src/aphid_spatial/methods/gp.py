"""Processus gaussien (GP) sur les observations probabilistes.

Deux variantes :

* :class:`MaternGPRegressor` — régression GP sur ``obs ∈ [0, 1]`` avec
  noyau Matérn, prédiction continue + écart-type postérieur. Variante
  la plus naturelle vu le modèle d'observation probabiliste.
* :class:`MaternGPClassifier` — classification GP sur ``obs > threshold``
  binarisé, conservée comme comparaison avec l'ancien pipeline binaire.

Les deux classes implémentent l'interface ``SpatialMethod``.

Pièges :

* La grille fait 100k cellules. ``GaussianProcessRegressor.predict`` sur 100k
  points avec 20 capteurs reste rapide (matrices 20×100k). Pas de problème.
* Si la variance des observations est trop faible (≤ 1e-9), sklearn lève une
  ConvergenceWarning. On ajoute un nugget plancher pour stabiliser.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)


def _make_matern_kernel(
    length_scale: float, nu: float, with_white: bool, white_level: float
) -> ConstantKernel:
    """Construit un noyau ``ConstantKernel * Matern + WhiteKernel`` standard."""
    base = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-3, 1e2)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=(0.5, 1e3),
        nu=nu,
    )
    if with_white:
        return base + WhiteKernel(
            noise_level=white_level, noise_level_bounds=(1e-6, 1.0)
        )
    return base


@dataclass
class MaternGPRegressor:
    """GP régression sur les observations probabilistes du capteur.

    Parameters
    ----------
    length_scale : float
        Initialisation de la portée du noyau Matérn (m).
    nu : float
        Paramètre de régularité Matérn (typique : 1.5 ou 2.5).
    alpha : float
        Bruit ajouté à la diagonale du Gram (régularisation numérique).
    learn_white_noise : bool
        Si ``True``, ajoute un ``WhiteKernel`` apprenable pour absorber
        le bruit binomial des capteurs ; sinon utilise ``alpha``.
    n_restarts : int
        Nombre de redémarrages de l'optimisation des hyperparamètres.
    """

    length_scale: float = 10.0
    nu: float = 1.5
    alpha: float = 1e-6
    learn_white_noise: bool = True
    n_restarts: int = 3

    name: str = field(default="gp_matern_regressor")
    _gp: GaussianProcessRegressor | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        del field_meta
        x = readings.coords.astype(np.float64)
        y = readings.obs.astype(np.float64)

        if x.shape[0] < 3 or float(y.var()) == 0.0:
            self._fallback_value = float(y.mean()) if y.size else 0.0
            self._gp = None
            logger.warning(
                "GPRegressor fallback (n=%d, var=%.3g) -> constante=%.3f",
                x.shape[0],
                float(y.var()) if y.size else 0.0,
                self._fallback_value,
            )
            return

        kernel = _make_matern_kernel(
            self.length_scale, self.nu, self.learn_white_noise, white_level=0.05
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x, y)
        self._gp = gp
        self._fallback_value = None
        logger.info("GPRegressor fit OK : kernel=%s", gp.kernel_)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._gp is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = self._gp.predict(query_coords.astype(np.float64), return_std=False)
        return np.clip(np.asarray(mean, dtype=np.float64), 0.0, 1.0)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._gp is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mean, std = self._gp.predict(
                query_coords.astype(np.float64), return_std=True
            )
        return np.asarray(std, dtype=np.float64)


@dataclass
class MaternGPClassifier:
    """GP classification binaire après seuillage de ``obs``.

    Parameters
    ----------
    threshold : float
        Seuil pour binariser ``obs`` (par défaut 0.5).
    length_scale : float
        Portée initiale du noyau Matérn.
    nu : float
        Paramètre Matérn.
    n_restarts : int
        Nombre de redémarrages de l'optimisation.
    """

    threshold: float = 0.5
    length_scale: float = 10.0
    nu: float = 1.5
    n_restarts: int = 1

    name: str = field(default="gp_matern_classifier")
    _gp: GaussianProcessClassifier | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        del field_meta
        x = readings.coords.astype(np.float64)
        y = (readings.obs > self.threshold).astype(np.int8)

        if len(np.unique(y)) < 2 or x.shape[0] < 3:
            self._fallback_value = float(y.mean()) if y.size else 0.0
            self._gp = None
            logger.warning(
                "GPClassifier fallback (n=%d, classes=%d) -> constante=%.3f",
                x.shape[0],
                len(np.unique(y)),
                self._fallback_value,
            )
            return

        kernel = ConstantKernel(constant_value=1.0) * Matern(
            length_scale=self.length_scale,
            length_scale_bounds=(0.5, 1e3),
            nu=self.nu,
        )
        gp = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x, y)
        self._gp = gp
        self._fallback_value = None
        logger.info("GPClassifier fit OK : kernel=%s", gp.kernel_)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._gp is None:
            assert self._fallback_value is not None
            return np.full(query_coords.shape[0], self._fallback_value, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self._gp.predict_proba(query_coords.astype(np.float64))
        # Colonne 1 = probabilité de la classe positive
        return np.asarray(proba[:, 1], dtype=np.float64)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        # Variance binomiale p(1-p) pour la classification
        p = self.predict_proba(query_coords)
        return np.sqrt(p * (1.0 - p))


__all__ = ["MaternGPClassifier", "MaternGPRegressor"]
