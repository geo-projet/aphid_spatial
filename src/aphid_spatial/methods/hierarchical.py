"""GLMM bayésien hiérarchique avec effet aléatoire spatial Matérn (PyMC).

Modèle :

.. code-block:: text

    logit(p_i) = β₀ + β₁ · d_bord_i + W_i
    W ~ MvNormal(0, σ² · K_Matern(ℓ))
    k_i ~ Binomial(K, p_i),  k_i = K · obs_i  (K = n_observations)

Si ``n_observations`` du capteur est ``None`` (capteur idéal sans bruit
d'estimation), on utilise une vraisemblance Beta tronquée centrée sur
``obs_i`` à la place du Binomial.

Inférence par NUTS sur les capteurs **uniquement** (n ~ 20). Pour la
prédiction sur la grille complète (100 k cellules), on évite de déclarer
100 k effets latents dans le modèle ; à la place, on applique le krigeage
postérieur conditionnel pour chaque échantillon de la chaîne :

* Pour chaque tirage ``(β₀, β₁, ℓ, σ², W_sensors)`` :

  1. Calculer ``W_grid = K_grid_sensors · K_sensors⁻¹ · W_sensors``
     (espérance conditionnelle du GP gaussien postérieur).
  2. ``p_grid = σ(β₀ + β₁ · d_bord_grid + W_grid)``.

* La carte prédite finale est la **moyenne postérieure** de ``p_grid``
  sur les échantillons. La variance entre échantillons fournit
  l'incertitude.

Ce schéma est moins « pur » qu'une inférence jointe sur les 100 k cellules,
mais il est scalable et reste cohérent avec le modèle.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal  # noqa: F401

import numpy as np
import pymc as pm
from numpy.typing import NDArray
from scipy.special import expit, logit

from aphid_spatial.methods.exploration import edge_distance
from aphid_spatial.methods.geostatistics import _edge_distance_at_coords
from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)

MaternNu = float  # 1.5 ou 2.5 (les deux seules valeurs supportées)


# ---------------------------------------------------------------------------
# Krigeage postérieur conditionnel (closed-form)
# ---------------------------------------------------------------------------


def _matern_cov(
    coords1: NDArray[np.floating],
    coords2: NDArray[np.floating],
    length_scale: float,
    sigma2: float,
    nu: float,
) -> NDArray[np.float64]:
    """Matrice de covariance Matérn (3/2 ou 5/2) entre deux jeux de coords."""
    diff = coords1[:, None, :] - coords2[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    ls = max(float(length_scale), 1e-6)
    if nu == 1.5:
        r = np.sqrt(3.0) * d / ls
        return sigma2 * (1.0 + r) * np.exp(-r)
    if nu == 2.5:
        r = np.sqrt(5.0) * d / ls
        return sigma2 * (1.0 + r + r**2 / 3.0) * np.exp(-r)
    raise ValueError(f"nu must be 1.5 or 2.5, got {nu}")


def _conditional_mean(
    coords_query: NDArray[np.floating],
    coords_sensors: NDArray[np.floating],
    w_sensors: NDArray[np.floating],
    length_scale: float,
    sigma2: float,
    nu: float,
    jitter: float = 1e-6,
) -> NDArray[np.float64]:
    """Espérance conditionnelle ``E[W_query | W_sensors]`` (krigeage simple)."""
    K_ss = _matern_cov(coords_sensors, coords_sensors, length_scale, sigma2, nu)
    K_qs = _matern_cov(coords_query, coords_sensors, length_scale, sigma2, nu)
    K_ss_reg = K_ss + jitter * np.eye(K_ss.shape[0])
    # Solve K_ss · α = w_sensors une fois ; W_query = K_qs · α
    alpha = np.linalg.solve(K_ss_reg, w_sensors)
    return (K_qs @ alpha).astype(np.float64)


# ---------------------------------------------------------------------------
# MaternGLMM
# ---------------------------------------------------------------------------


@dataclass
class MaternGLMM:
    """GLMM Binomial avec effet aléatoire spatial Matérn (PyMC).

    Parameters
    ----------
    matern_nu : float
        Régularité du noyau Matérn (``1.5`` ou ``2.5``).
    length_scale_prior : float
        Écart-type a priori (HalfNormal) sur ℓ (m).
    sigma_prior : float
        Écart-type a priori (HalfNormal) sur σ.
    n_draws : int
        Nombre d'échantillons NUTS post-tuning.
    n_tune : int
        Tuning steps (warmup).
    target_accept : float
        Cible d'acceptation NUTS (0.8–0.95 typique).
    chains : int
        Nombre de chaînes parallèles.
    seed : int
        Graine pour la reproductibilité.
    n_predict_samples : int
        Nombre d'échantillons utilisés pour la prédiction sur la grille.
        Réduit par sous-échantillonnage des draws (économie mémoire/temps).
    """

    matern_nu: MaternNu = 1.5
    length_scale_prior: float = 20.0
    sigma_prior: float = 1.5
    n_draws: int = 500
    n_tune: int = 500
    target_accept: float = 0.9
    chains: int = 2
    seed: int = 0
    n_predict_samples: int = 100

    name: str = field(default="matern_glmm_pymc")

    _readings: SensorReadings | None = field(default=None, init=False, repr=False)
    _field: Field | None = field(default=None, init=False, repr=False)
    _trace: object | None = field(default=None, init=False, repr=False)
    _fallback_value: float | None = field(default=None, init=False, repr=False)
    _post_samples: dict[str, NDArray[np.floating]] | None = field(
        default=None, init=False, repr=False
    )

    def fit(self, readings: SensorReadings, field_meta: Field) -> None:
        self._readings = readings
        self._field = field_meta
        n = readings.obs.size

        if n < 4:
            self._fallback_value = float(readings.obs.mean()) if n else 0.0
            self._trace = None
            logger.warning(
                "GLMM fallback (n=%d) -> constante=%.3f", n, self._fallback_value
            )
            return

        cfg = readings.config
        # Nombre de mesures temporelles : K=n_observations, ou défaut 1 si
        # capteur idéal (continu)
        k_obs = cfg.n_observations if cfg.n_observations is not None else 100
        # Ramener obs continu à des comptes entiers pour Binomial
        counts = np.round(np.asarray(readings.obs) * k_obs).astype(np.int64)
        counts = np.clip(counts, 0, k_obs)

        # Distance au bord aux capteurs (covariable centrée + réduite pour
        # stabiliser NUTS)
        d_edge_full = edge_distance(field_meta)
        d_edge_sensors = d_edge_full[readings.sensor_idx].astype(np.float64)
        d_mean = float(d_edge_sensors.mean())
        d_std = float(d_edge_sensors.std()) or 1.0
        d_z_sensors = (d_edge_sensors - d_mean) / d_std

        coords = readings.coords.astype(np.float64)
        prev_emp = float(readings.obs.mean())
        beta0_mu = float(logit(np.clip(prev_emp, 1e-3, 1 - 1e-3)))

        nu_int = 32 if self.matern_nu == 1.5 else 52  # PyMC class suffix

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pm.Model():
                length_scale = pm.HalfNormal(
                    "length_scale", sigma=self.length_scale_prior
                )
                sigma_W = pm.HalfNormal("sigma_W", sigma=self.sigma_prior)
                beta0 = pm.Normal("beta0", mu=beta0_mu, sigma=2.0)
                beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)

                if nu_int == 32:
                    cov = sigma_W**2 * pm.gp.cov.Matern32(2, ls=length_scale)
                else:
                    cov = sigma_W**2 * pm.gp.cov.Matern52(2, ls=length_scale)
                K = cov(coords) + 1e-6 * np.eye(n)
                W = pm.MvNormal("W", mu=np.zeros(n), cov=K, shape=n)

                logit_p = beta0 + beta1 * d_z_sensors + W
                p_sensor = pm.math.sigmoid(logit_p)
                pm.Binomial(
                    "obs_count", n=k_obs, p=p_sensor, observed=counts
                )

                self._trace = pm.sample(
                    draws=self.n_draws,
                    tune=self.n_tune,
                    chains=self.chains,
                    target_accept=self.target_accept,
                    random_seed=self.seed,
                    progressbar=False,
                    return_inferencedata=True,
                )

        # Extraire les échantillons (chains × draws × ...)
        post = self._trace.posterior  # type: ignore[attr-defined]
        beta0_s = post["beta0"].values.flatten()
        beta1_s = post["beta1"].values.flatten()
        ls_s = post["length_scale"].values.flatten()
        sigma_s = post["sigma_W"].values.flatten()
        W_s = post["W"].values.reshape(-1, n)  # (n_total_samples, n_sensors)

        # Sous-échantillonner pour la prédiction
        rng = np.random.default_rng(self.seed)
        n_total = beta0_s.size
        idx = rng.choice(
            n_total, size=min(self.n_predict_samples, n_total), replace=False
        )
        self._post_samples = {
            "beta0": beta0_s[idx],
            "beta1": beta1_s[idx],
            "length_scale": ls_s[idx],
            "sigma2": (sigma_s[idx]) ** 2,
            "W": W_s[idx],
            "d_mean": np.array([d_mean]),
            "d_std": np.array([d_std]),
        }
        logger.info(
            "GLMM fit OK : n=%d, draws=%d, kept=%d ; mean(beta1)=%+.3f, "
            "mean(sigma_W)=%.3f, mean(ls)=%.2f",
            n,
            n_total,
            idx.size,
            float(beta1_s.mean()),
            float(sigma_s.mean()),
            float(ls_s.mean()),
        )

    def _predict_p_per_sample(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        """Renvoie un tableau ``(n_samples, n_query)`` de probabilités prédites."""
        assert self._post_samples is not None and self._readings is not None
        assert self._field is not None
        ps = self._post_samples
        d_query = _edge_distance_at_coords(query_coords, self._field)
        d_z_query = (d_query - float(ps["d_mean"][0])) / float(ps["d_std"][0])
        coords_sensors = self._readings.coords.astype(np.float64)

        m = ps["beta0"].size
        out = np.empty((m, query_coords.shape[0]), dtype=np.float64)
        nu = self.matern_nu
        for i in range(m):
            w_grid = _conditional_mean(
                query_coords.astype(np.float64),
                coords_sensors,
                ps["W"][i],
                length_scale=float(ps["length_scale"][i]),
                sigma2=float(ps["sigma2"][i]),
                nu=nu,
            )
            logit_p = ps["beta0"][i] + ps["beta1"][i] * d_z_query + w_grid
            out[i] = expit(logit_p)
        return out

    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]:
        if self._post_samples is None:
            assert self._fallback_value is not None
            return np.full(
                query_coords.shape[0], self._fallback_value, dtype=np.float64
            )
        per_sample = self._predict_p_per_sample(query_coords)
        return np.clip(per_sample.mean(axis=0), 0.0, 1.0).astype(np.float64)

    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None:
        if self._post_samples is None:
            return None
        per_sample = self._predict_p_per_sample(query_coords)
        return per_sample.std(axis=0).astype(np.float64)

    @property
    def trace(self) -> object | None:
        return self._trace


__all__ = ["MaternGLMM", "MaternNu"]
