"""Schémas de placement des capteurs et modèle d'observation.

Cinq schémas sont supportés :

* ``uniform`` — tirage uniforme sans remise
* ``grid`` — sous-grille régulière la plus uniforme possible
* ``stratified`` — découpage en blocs ; 1 capteur/bloc tiré uniformément
* ``edge_biased`` — pondération par ``exp(-d_bord / λ)``
* ``poisson_disk`` — *dart throwing* avec contrainte de distance minimale

**Modèle d'observation** : chaque capteur observe la probabilité de présence
sur son voisinage (moyenne de ``prob`` sur la fenêtre ``(2r+1)²``). On suppose
que le capteur effectue ``n_observations`` mesures temporelles indépendantes
chacune Bernoulli de ce ``prob_local`` ; il retourne la fraction de positifs
``k / n_observations``. Avec ``n_observations=None``, le capteur retourne la
probabilité exacte (limite ``K→∞``, utile comme baseline théorique).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from aphid_spatial.simulation.field import Field

logger = logging.getLogger(__name__)

Placement = Literal["uniform", "grid", "stratified", "edge_biased", "poisson_disk"]


@dataclass(frozen=True)
class SensorConfig:
    """Configuration des capteurs.

    Attributes
    ----------
    n_sensors : int
        Nombre de capteurs à placer.
    placement : str
        Schéma de placement (voir module).
    sensor_radius : int
        Rayon du voisinage observé (en cellules de la grille). 0 = un seul plant ;
        > 0 = moyenne de ``prob`` sur la fenêtre ``(2r+1)²``.
    n_observations : int | None
        Nombre de mesures temporelles indépendantes par capteur. ``None`` =
        capteur idéal (lecture exacte de ``prob``). Sinon l'observation est
        ``Binomial(n_observations, prob_local) / n_observations`` et appartient
        à ``{0, 1/K, 2/K, ..., 1}``.
    seed : int
        Graine pour la reproductibilité (placement et bruit).
    edge_lambda_m : float
        Longueur caractéristique pour le schéma ``edge_biased``.
    """

    n_sensors: int = 20
    placement: Placement = "uniform"
    sensor_radius: int = 0
    n_observations: int | None = 50
    seed: int = 0
    edge_lambda_m: float = 20.0


@dataclass
class SensorReadings:
    """Observations des capteurs.

    Attributes
    ----------
    sensor_idx : NDArray[np.int64]
        Indices linéaires (row-major) des capteurs dans la grille du champ.
    coords : NDArray[np.float64]
        Coordonnées ``(n_sensors, 2)`` en mètres.
    obs : NDArray[np.float64]
        Probabilité de présence estimée par chaque capteur dans ``[0, 1]``.
        Vaut exactement ``prob_local`` si ``config.n_observations is None``,
        sinon ``Binomial(K, prob_local) / K``.
    prob_local : NDArray[np.float64]
        Probabilité réelle au voisinage du capteur (moyenne de ``prob`` sur
        la fenêtre). Sert de référence pour les analyses ; les méthodes
        d'estimation ne doivent pas l'utiliser directement.
    config : SensorConfig
        Configuration ayant produit ces observations.
    """

    sensor_idx: NDArray[np.int64]
    coords: NDArray[np.float64]
    obs: NDArray[np.float64]
    prob_local: NDArray[np.float64]
    config: SensorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factor_aspect(n: int, h: int, w: int) -> tuple[int, int]:
    """Trouve ``(nr, nc)`` tel que ``nr * nc >= n`` et le ratio approche ``h/w``.

    Utilisé pour construire une sous-grille ou un découpage en blocs.
    """
    aspect = max(h, 1) / max(w, 1)
    nr = max(1, int(round(np.sqrt(n * aspect))))
    nc = int(np.ceil(n / nr))
    # Au moins 1 dans chaque dimension
    nr = max(1, min(nr, h))
    nc = max(1, min(nc, w))
    # Si nr * nc < n on rééquilibre
    while nr * nc < n and (nr < h or nc < w):
        if nr < h and (nc >= w or nr * (nc + 1) <= nr * nc + 1):
            nr += 1
        elif nc < w:
            nc += 1
        else:
            break
    return nr, nc


def _edge_distance_grid(field: Field) -> NDArray[np.float64]:
    """Distance au bord (en mètres) pour chaque cellule, en row-major (N,)."""
    cfg = field.config
    spacing = cfg.spacing_m
    rows = np.arange(cfg.n_rows)
    cols = np.arange(cfg.n_cols)
    dy = np.minimum(rows, cfg.n_rows - 1 - rows) * spacing
    dx = np.minimum(cols, cfg.n_cols - 1 - cols) * spacing
    grid = np.minimum(dy[:, None], dx[None, :])
    return grid.ravel()


# ---------------------------------------------------------------------------
# Schémas de placement
# ---------------------------------------------------------------------------


def _place_uniform(
    field: Field, n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    n_total = field.config.n_rows * field.config.n_cols
    return rng.choice(n_total, size=n, replace=False).astype(np.int64)


def _place_grid(field: Field, n: int) -> NDArray[np.int64]:
    cfg = field.config
    nr, nc = _factor_aspect(n, cfg.n_rows, cfg.n_cols)
    row_idx = np.linspace(0, cfg.n_rows - 1, nr).round().astype(np.int64)
    col_idx = np.linspace(0, cfg.n_cols - 1, nc).round().astype(np.int64)
    rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")
    flat = rr.ravel() * cfg.n_cols + cc.ravel()
    flat = np.unique(flat)
    if flat.size > n:
        # Échantillonnage uniforme régulier des candidats
        sel = np.linspace(0, flat.size - 1, n).round().astype(np.int64)
        flat = flat[sel]
    elif flat.size < n:
        # Rare : on complète par tirage uniforme parmi les non-sélectionnés
        all_idx = np.arange(cfg.n_rows * cfg.n_cols, dtype=np.int64)
        remaining = np.setdiff1d(all_idx, flat, assume_unique=True)
        rng_pad = np.random.default_rng(0)
        extra = rng_pad.choice(remaining, size=n - flat.size, replace=False)
        flat = np.concatenate([flat, extra])
    return flat.astype(np.int64)


def _place_stratified(
    field: Field, n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    cfg = field.config
    nr, nc = _factor_aspect(n, cfg.n_rows, cfg.n_cols)
    # Bornes des blocs (inclusives - exclusives)
    row_edges = np.linspace(0, cfg.n_rows, nr + 1).astype(np.int64)
    col_edges = np.linspace(0, cfg.n_cols, nc + 1).astype(np.int64)

    indices: list[int] = []
    for i in range(nr):
        r0, r1 = row_edges[i], row_edges[i + 1]
        if r1 <= r0:
            continue
        for j in range(nc):
            c0, c1 = col_edges[j], col_edges[j + 1]
            if c1 <= c0:
                continue
            r = int(rng.integers(r0, r1))
            c = int(rng.integers(c0, c1))
            indices.append(r * cfg.n_cols + c)

    arr = np.array(indices, dtype=np.int64)
    # Recouper / compléter pour atteindre exactement n
    if arr.size > n:
        sel = rng.choice(arr.size, size=n, replace=False)
        arr = arr[sel]
    elif arr.size < n:
        all_idx = np.arange(cfg.n_rows * cfg.n_cols, dtype=np.int64)
        remaining = np.setdiff1d(all_idx, arr, assume_unique=False)
        extra = rng.choice(remaining, size=n - arr.size, replace=False)
        arr = np.concatenate([arr, extra])
    return arr


def _place_edge_biased(
    field: Field,
    n: int,
    rng: np.random.Generator,
    lambda_m: float,
) -> NDArray[np.int64]:
    edge_dist = _edge_distance_grid(field)
    weights = np.exp(-edge_dist / max(lambda_m, 1e-9))
    weights /= weights.sum()
    return rng.choice(weights.size, size=n, replace=False, p=weights).astype(np.int64)


def _place_poisson_disk(
    field: Field,
    n: int,
    rng: np.random.Generator,
    max_attempts_per_point: int = 200,
) -> NDArray[np.int64]:
    """Échantillonnage type *dart throwing* avec distance minimale adaptative.

    Plus simple que Bridson et suffisant pour ~20 points sur 100k cellules.
    """
    cfg = field.config
    spacing = cfg.spacing_m
    width_m = (cfg.n_cols - 1) * spacing
    height_m = (cfg.n_rows - 1) * spacing
    area = max(width_m * height_m, 1e-9)

    # Distance min initiale : grossièrement sqrt(area / (π * n))
    base_d = float(np.sqrt(area / (np.pi * max(n, 1))))
    chosen_xy: list[tuple[float, float]] = []
    chosen_idx: list[int] = []

    d_min = base_d
    for _ in range(10):  # boucle de secours, on relâche d_min si nécessaire
        chosen_xy.clear()
        chosen_idx.clear()
        attempts = 0
        max_attempts = max_attempts_per_point * n
        while len(chosen_xy) < n and attempts < max_attempts:
            r = int(rng.integers(0, cfg.n_rows))
            c = int(rng.integers(0, cfg.n_cols))
            x = c * spacing
            y = r * spacing
            ok = True
            for cx, cy in chosen_xy:
                if (x - cx) ** 2 + (y - cy) ** 2 < d_min * d_min:
                    ok = False
                    break
            if ok:
                idx = r * cfg.n_cols + c
                if idx not in chosen_idx:
                    chosen_xy.append((x, y))
                    chosen_idx.append(idx)
            attempts += 1
        if len(chosen_xy) >= n:
            break
        d_min *= 0.8
        logger.debug("poisson_disk: relâchement d_min -> %.3f", d_min)

    if len(chosen_xy) < n:
        # Fallback : compléter uniformément
        logger.warning(
            "poisson_disk: contrainte non satisfiable, complément uniforme (%d/%d)",
            len(chosen_xy),
            n,
        )
        all_idx = np.arange(cfg.n_rows * cfg.n_cols, dtype=np.int64)
        already = np.array(chosen_idx, dtype=np.int64)
        remaining = np.setdiff1d(all_idx, already, assume_unique=False)
        extra = rng.choice(remaining, size=n - len(chosen_idx), replace=False)
        chosen_idx.extend(int(x) for x in extra)
    return np.array(chosen_idx[:n], dtype=np.int64)


# ---------------------------------------------------------------------------
# Modèle d'observation
# ---------------------------------------------------------------------------


def _observe(
    field: Field,
    sensor_idx: NDArray[np.int64],
    config: SensorConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calcule l'observation probabiliste et la probabilité réelle locale.

    Returns
    -------
    obs : NDArray[np.float64]
        Observation du capteur dans ``[0, 1]``. Lecture exacte si
        ``config.n_observations is None``, sinon échantillon binomial.
    prob_local : NDArray[np.float64]
        Vraie probabilité moyennée sur le voisinage (sert de référence).
    """
    prob_grid = field.to_grid("prob")
    n_rows, n_cols = field.config.n_rows, field.config.n_cols
    rows, cols = np.divmod(sensor_idx, n_cols)

    if config.sensor_radius <= 0:
        prob_local = prob_grid[rows, cols].astype(np.float64)
    else:
        r = config.sensor_radius
        prob_local = np.empty(sensor_idx.shape, dtype=np.float64)
        for k, (rr, cc) in enumerate(zip(rows, cols, strict=True)):
            r0 = max(0, int(rr) - r)
            r1 = min(n_rows, int(rr) + r + 1)
            c0 = max(0, int(cc) - r)
            c1 = min(n_cols, int(cc) + r + 1)
            prob_local[k] = float(prob_grid[r0:r1, c0:c1].mean())

    if config.n_observations is None:
        obs = prob_local.copy()
    else:
        K = int(config.n_observations)
        if K <= 0:
            raise ValueError(
                f"n_observations must be a positive int or None, got {K}"
            )
        # Binomial(K, prob_local) / K, vectorisé
        counts = rng.binomial(K, np.clip(prob_local, 0.0, 1.0))
        obs = counts.astype(np.float64) / float(K)
    return obs, prob_local


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def place_sensors(field: Field, config: SensorConfig | None = None) -> SensorReadings:
    """Place les capteurs selon le schéma demandé et calcule leurs observations.

    Parameters
    ----------
    field : Field
        Vérité terrain produite par :func:`~aphid_spatial.simulation.field.simulate_field`.
    config : SensorConfig, optional
        Configuration des capteurs ; valeurs par défaut si ``None``.
    """
    if config is None:
        config = SensorConfig()

    n_total = field.config.n_rows * field.config.n_cols
    if config.n_sensors <= 0 or config.n_sensors > n_total:
        raise ValueError(
            f"n_sensors must be in (0, {n_total}], got {config.n_sensors}"
        )

    rng = np.random.default_rng(config.seed)

    if config.placement == "uniform":
        idx = _place_uniform(field, config.n_sensors, rng)
    elif config.placement == "grid":
        idx = _place_grid(field, config.n_sensors)
    elif config.placement == "stratified":
        idx = _place_stratified(field, config.n_sensors, rng)
    elif config.placement == "edge_biased":
        idx = _place_edge_biased(field, config.n_sensors, rng, config.edge_lambda_m)
    elif config.placement == "poisson_disk":
        idx = _place_poisson_disk(field, config.n_sensors, rng)
    else:  # pragma: no cover
        raise ValueError(f"Unknown placement scheme {config.placement!r}")

    if np.unique(idx).size != idx.size:
        # Sécurité supplémentaire (ne devrait pas arriver après les schémas)
        raise RuntimeError("Sensor placement produced duplicate indices")

    coords = field.coords[idx]
    obs, prob_local = _observe(field, idx, config, rng)

    logger.info(
        "place_sensors: %s, n=%d, obs_mean=%.3f, prob_local_mean=%.3f",
        config.placement,
        config.n_sensors,
        float(obs.mean()),
        float(prob_local.mean()),
    )

    return SensorReadings(
        sensor_idx=idx,
        coords=coords,
        obs=obs,
        prob_local=prob_local,
        config=config,
    )


__all__ = ["Placement", "SensorConfig", "SensorReadings", "place_sensors"]
