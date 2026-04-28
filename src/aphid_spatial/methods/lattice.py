"""Modèles de treillis : Markov Random Field (Ising V1 NumPy).

Cette première version implémente le modèle d'Ising binaire mentionné dans la
thèse comme « approche contextuelle markovienne ». Elle est entièrement
vectorisée NumPy (sans Numba) ; pour 100 k cellules et ~700 itérations elle
tourne en quelques secondes.

Conventions :

* Le champ latent ``y_i ∈ {0, 1}`` représente la présence/absence du puceron
  par plant. L'énergie est ``E(y) = -α Σ y_i - β Σ_{i~j} y_i y_j`` ; la loi
  conditionnelle d'un site est
  ``P(y_i = 1 | y_{-i}) = σ(α + β Σ_{j∈N(i)} y_j)``.
* Voisinage *rook* (4 voisins) ou *queen* (8 voisins).
* L'observation des capteurs est continue dans ``[0, 1]``. Pour la V1, on la
  binarise au seuil ``threshold`` (par défaut 0.5) et on impose ces valeurs
  à chaque itération du Gibbs.
* Les paramètres ``α, β`` sont :
  - imposés par l'utilisateur si fournis ;
  - sinon estimés par **pseudo-vraisemblance** (Besag) sur un champ initial
    bruité conditionné aux capteurs.
* L'échantillonneur Gibbs procède par mises à jour en damier (cellules
  « noires » puis « blanches »), ce qui garantit l'indépendance conditionnelle
  des sites mis à jour ensemble.

Une V2 optimisée Numba est prévue dans un round ultérieur si la performance
NumPy s'avère insuffisante sur le scénario complet.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)

Neighborhood = Literal["rook", "queen"]


# ---------------------------------------------------------------------------
# Outils bas-niveau : voisinage damier + somme des voisins vectorisée
# ---------------------------------------------------------------------------


def _neighbor_sum(grid: NDArray[np.integer | np.floating], queen: bool) -> NDArray[np.float64]:
    """Pour chaque cellule, somme de ses voisins (4 ou 8) avec bord à zéro.

    Implémenté par décalages NumPy vectorisés.
    """
    g = grid.astype(np.float64)
    s = np.zeros_like(g)
    # Rook (N, S, W, E) — bord = 0 (Dirichlet)
    s[1:, :] += g[:-1, :]
    s[:-1, :] += g[1:, :]
    s[:, 1:] += g[:, :-1]
    s[:, :-1] += g[:, 1:]
    if queen:
        s[1:, 1:] += g[:-1, :-1]
        s[1:, :-1] += g[:-1, 1:]
        s[:-1, 1:] += g[1:, :-1]
        s[:-1, :-1] += g[1:, 1:]
    return s


def _checkerboard_mask(n_rows: int, n_cols: int) -> NDArray[np.bool_]:
    """Masque booléen des cellules « noires » : ``(r + c) % 2 == 0``."""
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return ((rr + cc) % 2 == 0).astype(bool)


# ---------------------------------------------------------------------------
# Pseudo-vraisemblance pour estimer (α, β)
# ---------------------------------------------------------------------------


def _pseudo_log_likelihood(
    params: NDArray[np.floating],
    y: NDArray[np.integer],
    neigh_sum: NDArray[np.floating],
) -> float:
    """Négatif de la log-pseudo-vraisemblance (à minimiser).

    PL = Σ_i  y_i * η_i - log(1 + exp(η_i))   avec η = α + β * sum_neigh
    """
    alpha, beta = float(params[0]), float(params[1])
    eta = alpha + beta * neigh_sum
    # log(1 + exp(eta)) = softplus(eta), numériquement stable :
    softplus = np.where(eta > 0, eta + np.log1p(np.exp(-eta)), np.log1p(np.exp(eta)))
    ll = (y * eta - softplus).sum()
    return float(-ll)


def estimate_params_pseudo_likelihood(
    y_grid: NDArray[np.integer],
    queen: bool,
    *,
    alpha_init: float = 0.0,
    beta_init: float = 0.4,
) -> tuple[float, float]:
    """Estime (α, β) par pseudo-vraisemblance sur un champ binaire complet."""
    neigh_sum = _neighbor_sum(y_grid, queen).ravel()
    y_flat = y_grid.ravel().astype(np.int64)
    res = minimize(
        _pseudo_log_likelihood,
        x0=np.array([alpha_init, beta_init], dtype=np.float64),
        args=(y_flat, neigh_sum),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200},
    )
    alpha_est = float(res.x[0])
    beta_est = float(res.x[1])
    # Régularisation pratique : limiter β pour éviter les régimes super-critiques
    beta_est = float(np.clip(beta_est, -2.0, 2.0))
    return alpha_est, beta_est


# ---------------------------------------------------------------------------
# Échantillonneur Gibbs en damier conditionné
# ---------------------------------------------------------------------------


def _gibbs_one_color(
    y_grid: NDArray[np.integer],
    color_mask: NDArray[np.bool_],
    sensor_mask: NDArray[np.bool_],
    alpha: float,
    beta: float,
    queen: bool,
    rng: np.random.Generator,
) -> None:
    """Met à jour en place les cellules de la couleur ``color_mask``.

    Les cellules marquées dans ``sensor_mask`` ne sont pas modifiées
    (conditionnement strict aux capteurs).
    """
    sn = _neighbor_sum(y_grid, queen)
    eta = alpha + beta * sn
    p = expit(eta)
    update = color_mask & (~sensor_mask)
    if update.any():
        u = rng.random(y_grid.shape)
        new = (u < p).astype(np.int8)
        # Update only the cells of this color that are not sensors
        y_grid[update] = new[update]


# ---------------------------------------------------------------------------
# Classe SpatialMethod
# ---------------------------------------------------------------------------


@dataclass
class IsingMRF:
    """Modèle d'Ising binaire avec inférence par Gibbs conditionné.

    Parameters
    ----------
    neighborhood : {"rook", "queen"}
        Voisinage : 4 ou 8 voisins sur la grille.
    threshold : float
        Seuil de binarisation des observations continues : sensor_y_i = 1 si
        ``obs_i > threshold``, sinon 0.
    alpha, beta : float | None
        Paramètres du modèle. ``None`` ⇒ estimation par pseudo-vraisemblance
        sur le champ initial. Si fournis, ces valeurs sont utilisées telles
        quelles (utile pour reproductibilité ou diagnostic).
    n_burn : int
        Itérations de burn-in (non comptabilisées dans la moyenne).
    n_samples : int
        Itérations échantillonnées après burn-in.
    thin : int
        N'utilise qu'un échantillon sur ``thin`` (réduit la corrélation).
    seed : int
        Graine reproductibilité.
    """

    neighborhood: Neighborhood = "rook"
    threshold: float = 0.5
    alpha: float | None = None
    beta: float | None = None
    n_burn: int = 200
    n_samples: int = 500
    thin: int = 1
    seed: int = 0

    name: str = field(default="ising_mrf_v1")
    _p_grid: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _var_grid: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)
    _alpha_est: float | None = field(default=None, init=False, repr=False)
    _beta_est: float | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)
    _n_iter_kept: int = field(default=0, init=False, repr=False)
    _fit_seconds: float = field(default=0.0, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        """Estime les paramètres si nécessaire et lance l'échantillonneur."""
        self._field = field_meta
        cfg = field_meta.config
        n_rows, n_cols = cfg.n_rows, cfg.n_cols
        queen = self.neighborhood == "queen"
        rng = np.random.default_rng(self.seed)

        if readings.obs.size < 3:
            self._fallback_value = (
                float(readings.obs.mean()) if readings.obs.size else 0.0
            )
            self._p_grid = None
            logger.warning(
                "Ising fallback (n=%d) -> constante=%.3f",
                readings.obs.size,
                self._fallback_value,
            )
            return

        # 1) Champ initial : prévalence empirique partout, valeurs des
        #    capteurs imposées (binarisées au seuil).
        emp_prev = float(readings.obs.mean())
        y_grid = (rng.random((n_rows, n_cols)) < emp_prev).astype(np.int8)
        sensor_y = (readings.obs > self.threshold).astype(np.int8)
        sensor_rows, sensor_cols = np.divmod(readings.sensor_idx, n_cols)
        y_grid[sensor_rows, sensor_cols] = sensor_y
        sensor_mask = np.zeros_like(y_grid, dtype=bool)
        sensor_mask[sensor_rows, sensor_cols] = True

        # 2) Estimation des paramètres par pseudo-vraisemblance (sur le champ
        #    initial conditionné). Si l'utilisateur a fourni alpha/beta, on
        #    skip.
        if self.alpha is None or self.beta is None:
            alpha_init = float(np.log(max(emp_prev, 1e-3) / max(1 - emp_prev, 1e-3)))
            self._alpha_est, self._beta_est = estimate_params_pseudo_likelihood(
                y_grid, queen, alpha_init=alpha_init, beta_init=0.4
            )
        else:
            self._alpha_est = float(self.alpha)
            self._beta_est = float(self.beta)

        # 3) Échantillonneur Gibbs en damier conditionné.
        black = _checkerboard_mask(n_rows, n_cols)
        white = ~black
        sum_p = np.zeros_like(y_grid, dtype=np.float64)
        sum_p2 = np.zeros_like(y_grid, dtype=np.float64)
        n_kept = 0

        t0 = time.perf_counter()
        total_iter = self.n_burn + self.n_samples
        for it in range(total_iter):
            _gibbs_one_color(
                y_grid, black, sensor_mask,
                self._alpha_est, self._beta_est, queen, rng,
            )
            _gibbs_one_color(
                y_grid, white, sensor_mask,
                self._alpha_est, self._beta_est, queen, rng,
            )
            if it >= self.n_burn and ((it - self.n_burn) % max(self.thin, 1) == 0):
                sum_p += y_grid
                sum_p2 += y_grid  # y est binaire ⇒ y² = y
                n_kept += 1

        self._fit_seconds = time.perf_counter() - t0
        self._n_iter_kept = n_kept
        if n_kept > 0:
            mean = sum_p / n_kept
            # Var d'un Bernoulli observé : E[y²] - E[y]² = mean - mean²
            var = mean * (1.0 - mean)
            self._p_grid = mean.astype(np.float64)
            self._var_grid = var.astype(np.float64)
        else:  # pragma: no cover - défense
            self._p_grid = np.full((n_rows, n_cols), emp_prev, dtype=np.float64)
            self._var_grid = np.full(
                (n_rows, n_cols), emp_prev * (1 - emp_prev), dtype=np.float64
            )

        logger.info(
            "Ising fit OK : alpha=%.3f, beta=%.3f, n_kept=%d, t=%.1fs",
            self._alpha_est,
            self._beta_est,
            n_kept,
            self._fit_seconds,
        )

    def _grid_lookup(
        self, query_coords: NDArray[np.floating], grid: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Récupère la valeur de la grille au plant le plus proche."""
        assert self._field is not None
        cfg = self._field.config
        spacing = cfg.spacing_m
        # Indices entiers les plus proches sur la grille du champ
        cols = np.clip(
            np.round(query_coords[:, 0] / spacing).astype(np.int64),
            0,
            cfg.n_cols - 1,
        )
        rows = np.clip(
            np.round(query_coords[:, 1] / spacing).astype(np.int64),
            0,
            cfg.n_rows - 1,
        )
        return grid[rows, cols].astype(np.float64)

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._p_grid is None:
            assert self._fallback_value is not None
            return np.full(
                query_coords.shape[0], self._fallback_value, dtype=np.float64
            )
        return self._grid_lookup(query_coords, self._p_grid)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._var_grid is None:
            return None
        var = self._grid_lookup(query_coords, self._var_grid)
        return np.sqrt(np.maximum(var, 0.0))

    @property
    def params(self) -> dict[str, float]:
        """Paramètres estimés / utilisés."""
        return {
            "alpha": float(self._alpha_est) if self._alpha_est is not None else float("nan"),
            "beta": float(self._beta_est) if self._beta_est is not None else float("nan"),
            "fit_seconds": float(self._fit_seconds),
            "n_iter_kept": int(self._n_iter_kept),
        }


__all__ = [
    "IsingMRF",
    "Neighborhood",
    "checkerboard_mask",
    "estimate_params_pseudo_likelihood",
    "gibbs_one_color",
    "neighbor_sum",
]


# Alias publics pour exploration et tests (les helpers ne sont pas internes
# au point de mériter une visibilité strictement privée)
checkerboard_mask = _checkerboard_mask
gibbs_one_color = _gibbs_one_color
neighbor_sum = _neighbor_sum

# Ré-exporter cKDTree juste pour silence des linters si non utilisé ici
_ = cKDTree
