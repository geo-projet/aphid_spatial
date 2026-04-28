"""Cartes 2D pour la visualisation du champ et des prédictions.

Toutes les fonctions acceptent un argument optionnel ``ax`` ; si ``None``,
elles créent leur propre figure et retournent ``(fig, ax)``. Sinon
elles peignent sur l'axe fourni et retournent ``(ax.figure, ax)``.

Conventions de couleurs (CLAUDE.md §9.1) :

* Probabilités → palette séquentielle ``viridis`` sur ``[0, 1]``.
* Erreur ``p̂ - p`` → palette divergente ``RdBu_r`` centrée sur 0.
* Incertitude → ``magma``.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings


def _ensure_ax(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    parent = ax.figure
    # ax.figure peut être Figure ou SubFigure ; on remonte au Figure racine
    fig = parent.figure if not isinstance(parent, Figure) else parent
    return fig, ax


def _extent(field: Field) -> tuple[float, float, float, float]:
    cfg = field.config
    x_max = (cfg.n_cols - 1) * cfg.spacing_m
    y_max = (cfg.n_rows - 1) * cfg.spacing_m
    return (0.0, x_max, 0.0, y_max)


def plot_field(
    field: Field,
    what: Literal["prob", "presence"] = "prob",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2.4),
    cmap: str = "viridis",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap de la vérité terrain (probabilité ou présence)."""
    fig, ax = _ensure_ax(ax, figsize)
    arr = field.to_grid(what)
    vmin, vmax = (0.0, 1.0) if what == "prob" else (0, 1)
    im = ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        extent=_extent(field),
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"Champ — {what}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig, ax


def plot_sensors(
    readings: SensorReadings,
    field: Field,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2.4),
    show_field: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Affiche les capteurs (couleur selon ``obs ∈ [0, 1]``) ; optionnellement
    superposés à la probabilité vraie en arrière-plan.

    La taille du cercle croît linéairement avec ``obs`` pour rendre visibles les
    capteurs à forte présence même sur des fonds sombres.
    """
    fig, ax = _ensure_ax(ax, figsize)
    if show_field:
        plot_field(field, "prob", ax=ax, title="")
    sizes = 30.0 + 90.0 * np.clip(readings.obs, 0.0, 1.0)
    sc = ax.scatter(
        readings.coords[:, 0],
        readings.coords[:, 1],
        c=readings.obs,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=sizes,
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    n_pos = int((readings.obs > 0.5).sum())
    n_zero = int((readings.obs == 0.0).sum())
    ax.set_title(
        title
        or f"Capteurs ({readings.config.placement}) — obs>0.5 : {n_pos}/{readings.obs.size}, obs=0 : {n_zero}"
    )
    if not show_field:
        fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label="obs (fraction)")
    return fig, ax


def plot_prediction(
    p_pred: NDArray[np.floating],
    field: Field,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2.4),
    cmap: str = "viridis",
    title: str = "Probabilité prédite",
    readings: SensorReadings | None = None,
) -> tuple[Figure, Axes]:
    """Carte de la probabilité prédite (mêmes coords que le champ)."""
    fig, ax = _ensure_ax(ax, figsize)
    grid = np.asarray(p_pred).reshape(field.n_rows, field.n_cols)
    im = ax.imshow(
        grid,
        cmap=cmap,
        origin="lower",
        extent=_extent(field),
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if readings is not None:
        ax.scatter(
            readings.coords[:, 0],
            readings.coords[:, 1],
            c=readings.obs,
            cmap="coolwarm",
            edgecolors="black",
            s=30,
            zorder=3,
        )
    return fig, ax


def plot_uncertainty(
    sigma: NDArray[np.floating],
    field: Field,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2.4),
    cmap: str = "magma",
    title: str = "Incertitude (écart-type)",
) -> tuple[Figure, Axes]:
    """Carte de l'écart-type de la prédiction."""
    fig, ax = _ensure_ax(ax, figsize)
    grid = np.asarray(sigma).reshape(field.n_rows, field.n_cols)
    im = ax.imshow(
        grid,
        cmap=cmap,
        origin="lower",
        extent=_extent(field),
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig, ax


def plot_error(
    p_pred: NDArray[np.floating],
    p_true: NDArray[np.floating],
    field: Field,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2.4),
    cmap: str = "RdBu_r",
    vlim: float | None = None,
    title: str = "Erreur p̂ - p",
) -> tuple[Figure, Axes]:
    """Carte d'erreur (palette divergente centrée sur 0)."""
    fig, ax = _ensure_ax(ax, figsize)
    err = np.asarray(p_pred).reshape(field.n_rows, field.n_cols) - np.asarray(p_true).reshape(
        field.n_rows, field.n_cols
    )
    if vlim is None:
        vlim = float(np.nanmax(np.abs(err))) or 1e-9
    im = ax.imshow(
        err,
        cmap=cmap,
        origin="lower",
        extent=_extent(field),
        vmin=-vlim,
        vmax=vlim,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig, ax


def plot_summary(
    field: Field,
    readings: SensorReadings,
    p_pred: NDArray[np.floating],
    sigma: NDArray[np.floating] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Figure récapitulative 2×2 : vérité, prédiction, erreur, incertitude."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        constrained_layout=True,
    )
    plot_field(field, "prob", ax=axes[0, 0], title="Probabilité vraie")
    plot_sensors(readings, field, ax=axes[0, 1], show_field=True, title="Capteurs")
    plot_prediction(p_pred, field, ax=axes[1, 0], readings=readings, title="Probabilité prédite")
    if sigma is not None:
        plot_uncertainty(sigma, field, ax=axes[1, 1], title="Incertitude (σ)")
    else:
        plot_error(p_pred, field.prob, field, ax=axes[1, 1], title="Erreur p̂ - p")
    return fig


__all__ = [
    "plot_error",
    "plot_field",
    "plot_prediction",
    "plot_sensors",
    "plot_summary",
    "plot_uncertainty",
]
