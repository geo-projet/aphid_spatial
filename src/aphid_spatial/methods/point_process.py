"""Statistiques de processus ponctuels (Ripley K/L, KDE d'intensité, CSR).

Utile si on traite les capteurs « positifs » (ou pondérés par ``obs``) comme un
semis ponctuel sous la référence de tous les capteurs. La complète aléatoire
spatiale (CSR) sert d'hypothèse nulle de référence.

Fonctions exposées :

* :func:`ripley_k`           — K(r) avec correction de bord (Ripley).
* :func:`ripley_l`           — L(r) = √(K/π) - r (linéaire sous CSR).
* :func:`pair_correlation`   — g(r) approximée par différences finies sur K.
* :func:`csr_envelope`       — enveloppe Monte Carlo de la statistique
  choisie sous CSR (poisson homogène sur le rectangle support).
* :func:`kde_intensity`      — λ(s) via Gaussian KDE 2D pondérée.
* :func:`weighted_ripley_k`  — variante pondérée de K(r) (utile vu que
  ``obs ∈ [0, 1]`` est continu).

Toutes les fonctions acceptent un ``support`` rectangulaire (xmin, xmax,
ymin, ymax) ou le déduisent automatiquement de la bounding box des points.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pointpats import distance_statistics as ds
from scipy.stats import gaussian_kde

from aphid_spatial.simulation.field import Field

logger = logging.getLogger(__name__)

Statistic = Literal["K", "L", "g"]


def _bbox_support(coords: NDArray[np.floating]) -> tuple[float, float, float, float]:
    """Bounding box des points : (xmin, xmax, ymin, ymax)."""
    return (
        float(coords[:, 0].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].min()),
        float(coords[:, 1].max()),
    )


def _support_array(
    radii: NDArray[np.floating] | None,
    coords: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Construit une grille de rayons par défaut si non fournie."""
    if radii is not None:
        return np.asarray(radii, dtype=np.float64)
    xmin, xmax, ymin, ymax = _bbox_support(coords)
    max_r = 0.5 * min(xmax - xmin, ymax - ymin)
    return np.linspace(0.0, max(max_r, 1.0), 30, dtype=np.float64)


def ripley_k(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Fonction K de Ripley.

    Returns
    -------
    dict
        ``radii`` : tableau des rayons utilisés.
        ``K`` : K(r) à chaque rayon.
    """
    support = _support_array(radii, coords)
    r_arr, k_arr = ds.k(coords, support=support)
    return {
        "radii": np.asarray(r_arr, dtype=np.float64),
        "K": np.asarray(k_arr, dtype=np.float64),
    }


def ripley_l(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Fonction L de Besag : L(r) = √(K(r)/π) - r ; ≈ 0 sous CSR."""
    out = ripley_k(coords, radii)
    r = out["radii"]
    k = out["K"]
    l_vals = np.sqrt(np.maximum(k, 0.0) / np.pi) - r
    return {"radii": r, "L": l_vals.astype(np.float64)}


def pair_correlation(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Fonction g de pair-corrélation, approchée par g(r) = K'(r) / (2πr).

    L'approximation par différences finies est moins précise qu'un estimateur
    à noyau, mais suffisante pour un diagnostic visuel.
    """
    out = ripley_k(coords, radii)
    r = out["radii"]
    k = out["K"]
    dk_dr = np.gradient(k, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_vals = np.where(r > 0, dk_dr / (2.0 * np.pi * r), np.nan)
    return {"radii": r, "g": g_vals.astype(np.float64)}


def _sample_csr(
    n: int,
    support: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Tirage uniforme de ``n`` points sur le rectangle ``support``."""
    xmin, xmax, ymin, ymax = support
    x = rng.uniform(xmin, xmax, size=n)
    y = rng.uniform(ymin, ymax, size=n)
    return np.column_stack([x, y]).astype(np.float64)


def csr_envelope(
    coords: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
    *,
    statistic: Statistic = "L",
    n_sim: int = 99,
    seed: int = 0,
    support: tuple[float, float, float, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Enveloppe Monte Carlo de la statistique sous CSR.

    Returns
    -------
    dict avec ``radii``, ``observed``, ``low``, ``high``, ``mean``.
    Si ``observed`` sort de l'enveloppe ``[low, high]``, on rejette CSR
    au seuil approximatif ``2 / (n_sim + 1)``.
    """
    radii_arr = _support_array(radii, coords)
    bbox = support if support is not None else _bbox_support(coords)
    n = coords.shape[0]
    rng = np.random.default_rng(seed)

    if statistic == "K":
        observed = ripley_k(coords, radii_arr)["K"]

        def compute(c: NDArray[np.floating]) -> NDArray[np.float64]:
            return ripley_k(c, radii_arr)["K"]
    elif statistic == "L":
        observed = ripley_l(coords, radii_arr)["L"]

        def compute(c: NDArray[np.floating]) -> NDArray[np.float64]:
            return ripley_l(c, radii_arr)["L"]
    elif statistic == "g":
        observed = pair_correlation(coords, radii_arr)["g"]

        def compute(c: NDArray[np.floating]) -> NDArray[np.float64]:
            return pair_correlation(c, radii_arr)["g"]
    else:  # pragma: no cover
        raise ValueError(f"unknown statistic {statistic!r}")

    sims = np.empty((n_sim, radii_arr.size), dtype=np.float64)
    for i in range(n_sim):
        sims[i] = compute(_sample_csr(n, bbox, rng))

    # Enveloppe pointwise : 2.5e/97.5e percentiles
    low = np.nanpercentile(sims, 2.5, axis=0)
    high = np.nanpercentile(sims, 97.5, axis=0)
    mean = np.nanmean(sims, axis=0)
    return {
        "radii": radii_arr,
        "observed": np.asarray(observed, dtype=np.float64),
        "low": low.astype(np.float64),
        "high": high.astype(np.float64),
        "mean": mean.astype(np.float64),
    }


def kde_intensity(
    coords: NDArray[np.floating],
    query_coords: NDArray[np.floating],
    *,
    bandwidth: float | None = None,
    weights: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """λ(s) — intensité par lissage gaussien 2D.

    Parameters
    ----------
    coords : NDArray
        Positions des points (n, 2).
    query_coords : NDArray
        Positions où évaluer l'intensité (m, 2).
    bandwidth : float
        Largeur de bande (m). ``None`` = règle de Scott (par défaut scipy).
    weights : NDArray, optional
        Pondération de chaque point (par ex. ``obs`` du capteur).
    """
    if coords.shape[0] < 2:
        return np.zeros(query_coords.shape[0], dtype=np.float64)
    kde = gaussian_kde(
        coords.T,
        bw_method=(bandwidth / coords.std() if bandwidth is not None else "scott"),
        weights=weights,
    )
    # gaussian_kde retourne une densité (intégrale 1). Pour avoir une
    # « intensité » au sens point-process (nombre attendu / unité²), on
    # multiplie par n (ou somme des poids si pondéré).
    scale = float(coords.shape[0]) if weights is None else float(np.sum(weights))
    density = kde(query_coords.T)
    return (np.asarray(density, dtype=np.float64) * scale).astype(np.float64)


def weighted_ripley_k(
    coords: NDArray[np.floating],
    weights: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
    *,
    support: tuple[float, float, float, float] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """K(r) pondéré : chaque paire (i, j) contribue ``w_i * w_j``.

    Sous CSR avec poids constants, K(r) ≈ π r². On normalise par
    ``λ̂² = (Σ w)² / area`` pour rester comparable au K classique.
    """
    radii_arr = _support_array(radii, coords)
    bbox = support if support is not None else _bbox_support(coords)
    xmin, xmax, ymin, ymax = bbox
    area = max((xmax - xmin) * (ymax - ymin), 1e-9)
    n = coords.shape[0]

    # Distances pairwise
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)  # exclure auto-paires

    w_outer = weights[:, None] * weights[None, :]
    sum_w = float(weights.sum())
    if sum_w <= 0.0:
        return {"radii": radii_arr, "K": np.zeros_like(radii_arr)}

    # K(r) = (area / sum_w²) * Σ_{i≠j} w_i w_j 𝟙(d_ij ≤ r)
    k_vals = np.empty_like(radii_arr)
    for i, r in enumerate(radii_arr):
        mask = d <= r
        k_vals[i] = (area / max(sum_w**2, 1e-9)) * float(w_outer[mask].sum())
    logger.debug("weighted_ripley_k : n=%d, sum_w=%.2f, area=%.2f", n, sum_w, area)
    return {"radii": radii_arr, "K": k_vals.astype(np.float64)}


def support_from_field(field: Field) -> tuple[float, float, float, float]:
    """Bounding box du champ en mètres."""
    cfg = field.config
    return (
        0.0,
        (cfg.n_cols - 1) * cfg.spacing_m,
        0.0,
        (cfg.n_rows - 1) * cfg.spacing_m,
    )


__all__ = [
    "Statistic",
    "csr_envelope",
    "kde_intensity",
    "pair_correlation",
    "ripley_k",
    "ripley_l",
    "support_from_field",
    "weighted_ripley_k",
]
