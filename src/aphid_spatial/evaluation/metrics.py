"""Métriques d'évaluation des cartes probabilistes prédites.

Toutes les fonctions s'attendent à des tableaux 1D de même longueur,
``y_true ∈ {0, 1}`` et ``p_pred ∈ [0, 1]``.  Le clipping des probabilités
est appliqué uniquement là où c'est numériquement nécessaire (log-loss).

Note sur le CRPS : pour des prédictions probabilistes binaires, le CRPS
se réduit au Brier score. La fonction est mentionnée dans le projet pour
les méthodes futures qui produiront des distributions prédictives
complètes ; elle n'est pas implémentée dans ce round.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as skm


class MetricResults(TypedDict, total=False):
    auc_roc: float
    auc_pr: float
    brier: float
    log_loss: float
    mae_prob: float
    rmse_prob: float
    prevalence_true: float
    prevalence_pred: float


def _as1d(arr: NDArray[np.floating | np.integer]) -> NDArray[np.float64]:
    a = np.asarray(arr).ravel().astype(np.float64)
    return a


def auc_roc(y_true: NDArray[np.integer], p_pred: NDArray[np.floating]) -> float:
    """Aire sous la courbe ROC. Indéfini si une seule classe : retourne ``nan``."""
    y = _as1d(y_true)
    p = _as1d(p_pred)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(skm.roc_auc_score(y, p))


def auc_pr(y_true: NDArray[np.integer], p_pred: NDArray[np.floating]) -> float:
    """Aire sous la courbe précision-rappel (average precision)."""
    y = _as1d(y_true)
    p = _as1d(p_pred)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(skm.average_precision_score(y, p))


def brier(y_true: NDArray[np.integer], p_pred: NDArray[np.floating]) -> float:
    """Brier score : ``mean((p̂ - y)²)``."""
    y = _as1d(y_true)
    p = _as1d(p_pred)
    return float(np.mean((p - y) ** 2))


def log_loss_clipped(
    y_true: NDArray[np.integer],
    p_pred: NDArray[np.floating],
    eps: float = 1e-7,
) -> float:
    """Log-loss binaire avec clipping de ``p̂`` dans ``[eps, 1-eps]``."""
    y = _as1d(y_true)
    p = np.clip(_as1d(p_pred), eps, 1.0 - eps)
    # Quand y_true ne contient qu'une seule classe sklearn raise, on calcule à la main
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def mae_prob(p_true: NDArray[np.floating], p_pred: NDArray[np.floating]) -> float:
    """MAE entre la probabilité vraie ``p`` et la prédiction ``p̂``."""
    return float(np.mean(np.abs(_as1d(p_true) - _as1d(p_pred))))


def rmse_prob(p_true: NDArray[np.floating], p_pred: NDArray[np.floating]) -> float:
    """RMSE entre la probabilité vraie ``p`` et la prédiction ``p̂``."""
    diff = _as1d(p_true) - _as1d(p_pred)
    return float(np.sqrt(np.mean(diff**2)))


def calibration_curve_data(
    y_true: NDArray[np.integer],
    p_pred: NDArray[np.floating],
    n_bins: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Données pour une courbe de fiabilité.

    Returns
    -------
    dict
        ``bin_edges`` (n_bins+1), ``mean_pred`` (n_bins), ``frac_pos``
        (n_bins, fréquence empirique), ``count`` (n_bins).
        Les bins vides sont remplis de NaN sauf ``count`` (=0).
    """
    y = _as1d(y_true)
    p = _as1d(p_pred)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, p, side="right") - 1, 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        count[b] = c
        if c > 0:
            mean_pred[b] = float(p[mask].mean())
            frac_pos[b] = float(y[mask].mean())
    return {
        "bin_edges": bin_edges,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "count": count.astype(np.float64),
    }


def evaluate_all(
    y_true: NDArray[np.integer],
    p_pred: NDArray[np.floating],
    p_true: NDArray[np.floating] | None = None,
) -> MetricResults:
    """Calcule toutes les métriques principales d'un coup.

    Parameters
    ----------
    y_true : NDArray
        Présence binaire vraie sur la grille (0/1).
    p_pred : NDArray
        Probabilité prédite par la méthode.
    p_true : NDArray, optional
        Probabilité vraie (sortie de ``simulate_field``). Permet de calculer
        MAE/RMSE sur la probabilité, en plus des métriques sur la classe.
    """
    out: MetricResults = {
        "auc_roc": auc_roc(y_true, p_pred),
        "auc_pr": auc_pr(y_true, p_pred),
        "brier": brier(y_true, p_pred),
        "log_loss": log_loss_clipped(y_true, p_pred),
        "prevalence_true": float(_as1d(y_true).mean()),
        "prevalence_pred": float(_as1d(p_pred).mean()),
    }
    if p_true is not None:
        out["mae_prob"] = mae_prob(p_true, p_pred)
        out["rmse_prob"] = rmse_prob(p_true, p_pred)
    return out


__all__ = [
    "MetricResults",
    "auc_pr",
    "auc_roc",
    "brier",
    "calibration_curve_data",
    "evaluate_all",
    "log_loss_clipped",
    "mae_prob",
    "rmse_prob",
]
