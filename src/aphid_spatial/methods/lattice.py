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
import warnings
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


# ---------------------------------------------------------------------------
# CAR / BYM (PyMC) et SAR (spreg) sur les capteurs
# ---------------------------------------------------------------------------
#
# Ces modèles sont déclarés sur les capteurs uniquement (~20 nœuds), avec
# une matrice d'adjacence KNN. Pour la prédiction sur la grille complète,
# on utilise une **interpolation IDW** des effets latents postérieurs des
# capteurs (approximation pragmatique : CAR/BYM ne sont pas définis hors
# du graphe ; étendre à 100 k cellules nécessiterait une formulation
# géostatistique séparée, ce qui sort du cadre V1).


def _knn_adjacency(coords: NDArray[np.floating], k: int = 4) -> NDArray[np.float64]:
    """Construit une matrice d'adjacence binaire symétrique KNN."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    nearest = np.argsort(d, axis=1)[:, :k]
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in nearest[i]:
            W[i, int(j)] = 1.0
    # Symétriser : si i -> j alors j -> i
    return np.asarray(np.maximum(W, W.T), dtype=np.float64)


def _idw_interp(
    query_coords: NDArray[np.floating],
    sensor_coords: NDArray[np.floating],
    sensor_values: NDArray[np.floating],
    *,
    power: float = 2.0,
    eps: float = 1e-3,
) -> NDArray[np.float64]:
    """Interpolation inverse-distance (IDW) sur ``query_coords``."""
    diff = query_coords[:, None, :] - sensor_coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    weights = 1.0 / np.maximum(d, eps) ** power
    norm = weights.sum(axis=1, keepdims=True)
    return ((weights * sensor_values[None, :]).sum(axis=1) / norm.ravel()).astype(
        np.float64
    )


def _bayesian_lattice_fit_predict(
    readings: SensorReadings,
    field_meta: Field,
    model_kind: Literal["car", "bym"],
    *,
    n_neighbors: int,
    n_draws: int,
    n_tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    n_predict_samples: int,
) -> tuple[NDArray[np.float64] | None, dict[str, NDArray[np.floating]] | None]:
    """Ajuste un modèle CAR ou BYM via PyMC, renvoie les échantillons utiles.

    Retourne ``(None, None)`` en cas de fallback (peu de capteurs).
    """
    import warnings as _warnings

    import pymc as pm  # import local pour éviter import inutile si non utilisé

    n = readings.obs.size
    if n < 4:
        return None, None

    cfg = readings.config
    k_obs = cfg.n_observations if cfg.n_observations is not None else 100
    counts = np.clip(np.round(readings.obs * k_obs), 0, k_obs).astype(np.int64)

    coords = readings.coords.astype(np.float64)
    W_adj = _knn_adjacency(coords, k=min(n_neighbors, n - 1))

    # Distance au bord normalisée
    cfg_field = field_meta.config
    spacing = cfg_field.spacing_m
    rows = np.arange(cfg_field.n_rows)
    cols = np.arange(cfg_field.n_cols)
    dy = np.minimum(rows, cfg_field.n_rows - 1 - rows) * spacing
    dx = np.minimum(cols, cfg_field.n_cols - 1 - cols) * spacing
    d_full = np.minimum(dy[:, None], dx[None, :]).ravel()
    d_sensors = d_full[readings.sensor_idx].astype(np.float64)
    d_mean = float(d_sensors.mean())
    d_std = float(d_sensors.std()) or 1.0
    d_z = (d_sensors - d_mean) / d_std

    prev_emp = float(readings.obs.mean())
    beta0_mu = float(np.log(np.clip(prev_emp, 1e-3, 1 - 1e-3) /
                            (1 - np.clip(prev_emp, 1e-3, 1 - 1e-3))))

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with pm.Model() as model:
            del model  # silence linter
            tau_struct = pm.HalfNormal("tau_struct", sigma=2.0)
            alpha_car = pm.Beta("alpha_car", 2.0, 2.0)
            beta0 = pm.Normal("beta0", mu=beta0_mu, sigma=2.0)
            beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)

            W_struct = pm.CAR(
                "W_struct",
                mu=np.zeros(n),
                W=W_adj,
                alpha=alpha_car,
                tau=tau_struct,
                shape=n,
            )

            if model_kind == "bym":
                sigma_iid = pm.HalfNormal("sigma_iid", sigma=1.0)
                W_iid = pm.Normal("W_iid", mu=0.0, sigma=sigma_iid, shape=n)
                w_total = W_struct + W_iid
            else:
                w_total = W_struct

            logit_p = beta0 + beta1 * d_z + w_total
            pm.Binomial(
                "obs_count",
                n=k_obs,
                p=pm.math.sigmoid(logit_p),
                observed=counts,
            )

            trace = pm.sample(
                draws=n_draws,
                tune=n_tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                progressbar=False,
                return_inferencedata=True,
            )

    post = trace.posterior
    beta0_s = post["beta0"].values.flatten()
    beta1_s = post["beta1"].values.flatten()
    W_struct_s = post["W_struct"].values.reshape(-1, n)
    if model_kind == "bym":
        W_iid_s = post["W_iid"].values.reshape(-1, n)
        W_tot_s = W_struct_s + W_iid_s
    else:
        W_tot_s = W_struct_s

    rng = np.random.default_rng(seed)
    n_total = beta0_s.size
    idx = rng.choice(n_total, size=min(n_predict_samples, n_total), replace=False)
    samples = {
        "beta0": beta0_s[idx],
        "beta1": beta1_s[idx],
        "W_total": W_tot_s[idx],
        "d_mean": np.array([d_mean]),
        "d_std": np.array([d_std]),
    }
    return None, samples  # premier élément réservé pour usages futurs


@dataclass
class CARModel:
    """Modèle CAR (Conditional Autoregressive, Besag) via PyMC.

    Le champ latent ``W`` est défini sur les capteurs avec un voisinage KNN
    et une distribution ``pm.CAR`` (précision creuse). Pour la prédiction
    sur la grille, on interpole les ``W_sensors`` postérieurs par IDW.

    Modèle :

    .. code-block:: text

        logit(p_i) = β₀ + β₁ · d_bord_i + W_i
        W ~ CAR(mu=0, W=KNN(k), α, τ)
        k_i ~ Binomial(K, p_i)

    Parameters
    ----------
    n_neighbors : int
        Nombre de voisins par capteur dans la matrice d'adjacence.
    n_draws, n_tune, chains, target_accept, seed, n_predict_samples
        Paramètres NUTS standards.
    idw_power : float
        Exposant IDW pour l'interpolation des effets latents sur la grille.
    """

    n_neighbors: int = 4
    n_draws: int = 500
    n_tune: int = 500
    chains: int = 2
    target_accept: float = 0.9
    seed: int = 0
    n_predict_samples: int = 100
    idw_power: float = 2.0

    name: str = field(default="car_pymc")

    _readings: SensorReadings | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)
    _samples: dict[str, NDArray[np.floating]] | None = field(
        default=None, init=False, repr=False
    )
    _fallback_value: float | None = field(default=None, init=False, repr=False)
    _model_kind: str = field(default="car", init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        self._readings = readings
        self._field = field_meta
        if readings.obs.size < 4:
            self._fallback_value = (
                float(readings.obs.mean()) if readings.obs.size else 0.0
            )
            self._samples = None
            return
        _, samples = _bayesian_lattice_fit_predict(
            readings, field_meta,
            model_kind=self._model_kind,  # type: ignore[arg-type]
            n_neighbors=self.n_neighbors,
            n_draws=self.n_draws,
            n_tune=self.n_tune,
            chains=self.chains,
            target_accept=self.target_accept,
            seed=self.seed,
            n_predict_samples=self.n_predict_samples,
        )
        self._samples = samples
        self._fallback_value = None
        if samples is not None:
            logger.info(
                "%s fit OK : kept=%d samples, mean(beta1)=%+.3f",
                self._model_kind.upper(),
                samples["beta0"].size,
                float(samples["beta1"].mean()),
            )

    def _predict_per_sample(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        assert self._samples is not None and self._readings is not None
        assert self._field is not None
        s = self._samples
        # d_z aux query
        cfg = self._field.config
        x_max = (cfg.n_cols - 1) * cfg.spacing_m
        y_max = (cfg.n_rows - 1) * cfg.spacing_m
        dx = np.minimum(query_coords[:, 0], x_max - query_coords[:, 0])
        dy = np.minimum(query_coords[:, 1], y_max - query_coords[:, 1])
        d_q = np.minimum(dx, dy)
        d_z_q = (d_q - float(s["d_mean"][0])) / float(s["d_std"][0])

        m = s["beta0"].size
        out = np.empty((m, query_coords.shape[0]), dtype=np.float64)
        for i in range(m):
            w_grid = _idw_interp(
                query_coords.astype(np.float64),
                self._readings.coords.astype(np.float64),
                s["W_total"][i],
                power=self.idw_power,
            )
            from scipy.special import expit as _expit
            out[i] = _expit(s["beta0"][i] + s["beta1"][i] * d_z_q + w_grid)
        return out

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._samples is None:
            assert self._fallback_value is not None
            return np.full(
                query_coords.shape[0], self._fallback_value, dtype=np.float64
            )
        per = self._predict_per_sample(query_coords)
        return np.clip(per.mean(axis=0), 0.0, 1.0).astype(np.float64)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._samples is None:
            return None
        per = self._predict_per_sample(query_coords)
        return per.std(axis=0).astype(np.float64)


@dataclass
class BYMModel(CARModel):
    """Besag-York-Mollié : CAR structuré + composante iid Normal(0, σ_iid)."""

    name: str = field(default="bym_pymc")
    _model_kind: str = field(default="bym", init=False, repr=False)


# ---------------------------------------------------------------------------
# SAR fréquentiste via spreg
# ---------------------------------------------------------------------------


@dataclass
class SARLagModel:
    """SAR Lag model (spreg.GM_Lag) sur les observations probabilistes.

    Modèle :

    .. code-block:: text

        y = ρ · W · y + X β + ε

    Inférence fréquentiste par moments généralisés (``GM_Lag``) sur les
    capteurs. Pour la prédiction sur la grille, on utilise la décomposition
    spatiale ``y_grid ≈ X_grid β + (matrice de propagation IDW depuis les
    résidus capteurs)`` qui est une approximation pragmatique vu que SAR
    n'est pas formulé pour des positions quelconques.

    Parameters
    ----------
    n_neighbors : int
        Voisinage KNN pour construire la matrice de poids.
    idw_power : float
        Exposant IDW pour la propagation aux query coords.
    """

    n_neighbors: int = 4
    idw_power: float = 2.0

    name: str = field(default="sar_lag_spreg")
    _readings: SensorReadings | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)
    _betas: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    _rho: float = field(default=0.0, init=False, repr=False)
    _residuals: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _d_mean: float = field(default=0.0, init=False, repr=False)
    _d_std: float = field(default=1.0, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        from libpysal.weights import KNN as _KNN
        from spreg import GM_Lag

        self._readings = readings
        self._field = field_meta
        n = readings.obs.size
        if n < 5 or float(readings.obs.var()) == 0.0:
            self._fallback_value = float(readings.obs.mean()) if n else 0.0
            self._betas = None
            return

        # Distance au bord aux capteurs (+ z-score)
        cfg = field_meta.config
        spacing = cfg.spacing_m
        rows = np.arange(cfg.n_rows)
        cols = np.arange(cfg.n_cols)
        dy = np.minimum(rows, cfg.n_rows - 1 - rows) * spacing
        dx = np.minimum(cols, cfg.n_cols - 1 - cols) * spacing
        d_full = np.minimum(dy[:, None], dx[None, :]).ravel()
        d_sensors = d_full[readings.sensor_idx].astype(np.float64)
        self._d_mean = float(d_sensors.mean())
        self._d_std = float(d_sensors.std()) or 1.0
        d_z = (d_sensors - self._d_mean) / self._d_std

        y = readings.obs.astype(np.float64).reshape(-1, 1)
        X = d_z.reshape(-1, 1)
        w = _KNN.from_array(readings.coords, k=min(self.n_neighbors, n - 1))
        w.transform = "r"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GM_Lag(y, X, w=w)
        except Exception as exc:  # pragma: no cover
            self._fallback_value = float(y.mean())
            self._betas = None
            logger.warning("SAR fit a échoué (%s) -> baseline", exc)
            return

        # GM_Lag.betas inclut [intercept, X β, ρ]
        self._betas = np.asarray(model.betas, dtype=np.float64).ravel()
        self._rho = float(self._betas[-1])
        self._residuals = np.asarray(model.u, dtype=np.float64).ravel()
        self._fallback_value = None
        logger.info(
            "SAR Lag fit OK : intercept=%.3f, beta1=%.3f, rho=%.3f",
            float(self._betas[0]),
            float(self._betas[1]),
            self._rho,
        )

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._betas is None:
            assert self._fallback_value is not None
            return np.full(
                query_coords.shape[0], self._fallback_value, dtype=np.float64
            )
        assert self._readings is not None and self._field is not None
        # Composante linéaire : β₀ + β₁ * d_z(query)
        cfg = self._field.config
        x_max = (cfg.n_cols - 1) * cfg.spacing_m
        y_max = (cfg.n_rows - 1) * cfg.spacing_m
        dx = np.minimum(query_coords[:, 0], x_max - query_coords[:, 0])
        dy = np.minimum(query_coords[:, 1], y_max - query_coords[:, 1])
        d_q = np.minimum(dx, dy)
        d_z = (d_q - self._d_mean) / self._d_std
        linear = self._betas[0] + self._betas[1] * d_z

        # Propagation spatiale : ρ · IDW(y_sensors)
        y_sens = self._readings.obs.astype(np.float64)
        spatial = self._rho * _idw_interp(
            query_coords.astype(np.float64),
            self._readings.coords.astype(np.float64),
            y_sens,
            power=self.idw_power,
        )
        out = linear + spatial
        return np.clip(out, 0.0, 1.0).astype(np.float64)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        # Pas d'incertitude analytique simple ; on ne renvoie rien.
        del query_coords
        return None

    @property
    def params(self) -> dict[str, float]:
        if self._betas is None:
            return {"rho": float("nan"), "beta0": float("nan"), "beta1": float("nan")}
        return {
            "rho": float(self._rho),
            "beta0": float(self._betas[0]),
            "beta1": float(self._betas[1]),
        }


__all__ = [
    "BYMModel",
    "CARModel",
    "IsingMRF",
    "Neighborhood",
    "SARLagModel",
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
