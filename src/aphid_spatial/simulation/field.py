"""Génération de la vérité terrain pour un champ de laitue.

Le modèle générateur combine quatre composantes pour ``logit(p)`` :

1. Niveau de base ``μ = logit(base_prevalence)``
2. Effet de bordure ``β_edge * exp(-d_bord / λ_edge)``
3. Champ gaussien stationnaire (covariance Matérn) via ``gstools``
4. Foyers d'infestation gaussiens 2D, préférentiellement près des bords

La probabilité ``p`` est obtenue par sigmoïde, et ``y ~ Bernoulli(p)``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import gstools as gs
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldConfig:
    """Paramètres du générateur du champ.

    Attributes
    ----------
    n_rows, n_cols : int
        Dimensions de la grille (par défaut 100 × 1000 plants).
    spacing_m : float
        Espacement régulier entre plants, en mètres.
    base_prevalence : float
        Prévalence moyenne attendue (avant effets edge / GRF / hotspots).
    edge_strength : float
        Amplitude maximale de l'effet de bordure sur ``logit(p)``.
    edge_lambda_m : float
        Longueur caractéristique de décroissance de l'effet de bordure (m).
    matern_range_m : float
        Portée (``len_scale``) du noyau Matérn (m).
    matern_nu : float
        Paramètre de régularité du noyau Matérn.
    matern_var : float
        Variance du champ gaussien sur l'échelle ``logit``.
    n_hotspots : tuple[int, int]
        Bornes [min, max] (incluses) du nombre de foyers tirés.
    hotspot_strength : float
        Amplitude maximale d'un foyer sur ``logit(p)``.
    hotspot_radius_m : float
        Écart-type spatial d'un foyer (m).
    seed : int
        Graine pour la reproductibilité.
    """

    n_rows: int = 100
    n_cols: int = 1000
    spacing_m: float = 0.30
    base_prevalence: float = 0.05
    edge_strength: float = 1.5
    edge_lambda_m: float = 20.0
    matern_range_m: float = 10.0
    matern_nu: float = 1.5
    matern_var: float = 1.5
    n_hotspots: tuple[int, int] = (3, 8)
    hotspot_strength: float = 2.0
    hotspot_radius_m: float = 8.0
    seed: int = 42


@dataclass
class Field:
    """Vérité terrain simulée sur une grille régulière.

    Attributes
    ----------
    coords : NDArray[np.float64]
        Coordonnées ``(N, 2)`` des plants en mètres ; colonnes ``[x, y]``.
    prob : NDArray[np.float64]
        Probabilité vraie de présence pour chaque plant, ``(N,)``.
    presence : NDArray[np.int8]
        Tirage Bernoulli associé, ``(N,)`` à valeurs dans {0, 1}.
    config : FieldConfig
        Paramètres ayant servi à la simulation.
    """

    coords: NDArray[np.float64]
    prob: NDArray[np.float64]
    presence: NDArray[np.int8]
    config: FieldConfig

    @property
    def n_rows(self) -> int:
        return self.config.n_rows

    @property
    def n_cols(self) -> int:
        return self.config.n_cols

    @property
    def spacing_m(self) -> float:
        return self.config.spacing_m

    def to_grid(self, what: Literal["prob", "presence"]) -> NDArray[Any]:
        """Reformer un tableau 2D ``(n_rows, n_cols)`` à partir de l'attribut demandé.

        Le stockage interne est en *row-major* : l'index ``k = r * n_cols + c``.
        """
        arr: NDArray[Any]
        if what == "prob":
            arr = self.prob
        elif what == "presence":
            arr = self.presence
        else:  # pragma: no cover - défensif
            raise ValueError(f"what must be 'prob' or 'presence', got {what!r}")
        return arr.reshape(self.n_rows, self.n_cols)

    def save(self, path: str | Path) -> None:
        """Sauvegarde le champ et sa configuration en NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coords=self.coords,
            prob=self.prob,
            presence=self.presence,
            config=np.array([asdict(self.config)], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> Field:
        """Charge un champ depuis un fichier NPZ produit par :meth:`save`."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            cfg_dict = dict(npz["config"][0])
            # n_hotspots peut avoir été désérialisé en list ; on le rend en tuple
            if isinstance(cfg_dict.get("n_hotspots"), list):
                cfg_dict["n_hotspots"] = tuple(cfg_dict["n_hotspots"])
            config = FieldConfig(**cfg_dict)
            return cls(
                coords=npz["coords"],
                prob=npz["prob"],
                presence=npz["presence"].astype(np.int8),
                config=config,
            )


def _grid_coords(config: FieldConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Retourne les axes 1D ``x`` (colonnes) et ``y`` (lignes) en mètres."""
    x_axis = np.arange(config.n_cols, dtype=np.float64) * config.spacing_m
    y_axis = np.arange(config.n_rows, dtype=np.float64) * config.spacing_m
    return x_axis, y_axis


def _edge_distance(x_axis: NDArray[np.float64], y_axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance au bord le plus proche pour chaque cellule de la grille (n_rows, n_cols)."""
    x_max = x_axis[-1]
    y_max = y_axis[-1]
    dx = np.minimum(x_axis, x_max - x_axis)
    dy = np.minimum(y_axis, y_max - y_axis)
    # broadcasting : dy[:, None] (n_rows, 1) et dx[None, :] (1, n_cols)
    return np.minimum(dy[:, None], dx[None, :])


def _generate_grf(
    x_axis: NDArray[np.float64],
    y_axis: NDArray[np.float64],
    config: FieldConfig,
) -> NDArray[np.float64]:
    """Champ gaussien Matérn centré, variance ``matern_var``, portée ``matern_range_m``.

    Utilise la simulation structurée FFT-based de ``gstools`` pour gérer 100k cellules.
    """
    model = gs.Matern(
        dim=2,
        var=config.matern_var,
        len_scale=config.matern_range_m,
        nu=config.matern_nu,
    )
    srf = gs.SRF(model, mean=0.0, seed=config.seed)
    # gstools structured prend les axes ; la sortie est de forme (len(x), len(y))
    grf_xy = srf.structured((x_axis, y_axis))
    # On veut (n_rows, n_cols) = (len(y), len(x)) → transpose
    return np.asarray(grf_xy.T, dtype=np.float64)


def _sample_hotspot_centers(
    rng: np.random.Generator,
    x_axis: NDArray[np.float64],
    y_axis: NDArray[np.float64],
    config: FieldConfig,
) -> NDArray[np.float64]:
    """Tire ``k ∈ [n_hotspots]`` centres de foyers, biaisés vers les bords.

    La densité de tirage est proportionnelle à ``exp(-d_bord / (2 * edge_lambda_m))``,
    de manière à concentrer les foyers près des bords sans les y confiner exclusivement.
    """
    k_min, k_max = config.n_hotspots
    k = int(rng.integers(k_min, k_max + 1))
    if k == 0:
        return np.empty((0, 2), dtype=np.float64)

    edge_grid = _edge_distance(x_axis, y_axis)  # (n_rows, n_cols)
    weights = np.exp(-edge_grid / max(config.edge_lambda_m * 2.0, 1e-9))
    weights = weights.ravel()
    weights /= weights.sum()

    flat_idx = rng.choice(weights.size, size=k, replace=False, p=weights)
    rows, cols = np.unravel_index(flat_idx, (config.n_rows, config.n_cols))
    centers = np.column_stack([x_axis[cols], y_axis[rows]]).astype(np.float64)
    return centers


def _hotspot_field(
    centers: NDArray[np.float64],
    x_axis: NDArray[np.float64],
    y_axis: NDArray[np.float64],
    config: FieldConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Somme de bosses gaussiennes 2D ``hotspot_strength * exp(-r²/(2σ²))``.

    Chaque foyer reçoit une amplitude tirée uniformément dans ``[0.5, 1] * hotspot_strength``
    pour briser la symétrie.
    """
    out = np.zeros((config.n_rows, config.n_cols), dtype=np.float64)
    if centers.size == 0:
        return out
    sigma = max(config.hotspot_radius_m, 1e-9)
    XX, YY = np.meshgrid(x_axis, y_axis)  # (n_rows, n_cols)
    for cx, cy in centers:
        amp = config.hotspot_strength * float(rng.uniform(0.5, 1.0))
        r2 = (XX - cx) ** 2 + (YY - cy) ** 2
        out += amp * np.exp(-r2 / (2.0 * sigma**2))
    return out


def simulate_field(config: FieldConfig | None = None) -> Field:
    """Génère un champ synthétique selon le modèle décrit dans le module.

    Parameters
    ----------
    config : FieldConfig, optional
        Paramètres de simulation. Si ``None``, utilise les valeurs par défaut.

    Returns
    -------
    Field
        Champ simulé (coordonnées, probabilité vraie, présence binaire).
    """
    if config is None:
        config = FieldConfig()

    rng = np.random.default_rng(config.seed)
    x_axis, y_axis = _grid_coords(config)

    # 1) base
    mu = float(logit(config.base_prevalence))

    # 2) effet de bordure
    edge_grid = _edge_distance(x_axis, y_axis)
    f_edge = config.edge_strength * np.exp(-edge_grid / max(config.edge_lambda_m, 1e-9))

    # 3) champ gaussien Matérn
    f_grf = _generate_grf(x_axis, y_axis, config)

    # 4) foyers
    centers = _sample_hotspot_centers(rng, x_axis, y_axis, config)
    f_hot = _hotspot_field(centers, x_axis, y_axis, config, rng)

    logit_p = mu + f_edge + f_grf + f_hot
    prob_grid = expit(logit_p)

    # Tirage Bernoulli
    presence_grid = (rng.random(prob_grid.shape) < prob_grid).astype(np.int8)

    # Coordonnées (row-major : k = r * n_cols + c)
    XX, YY = np.meshgrid(x_axis, y_axis)
    coords = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float64)

    logger.info(
        "simulate_field: grid %dx%d, prevalence empirique=%.4f, n_hotspots=%d",
        config.n_rows,
        config.n_cols,
        float(presence_grid.mean()),
        centers.shape[0],
    )

    return Field(
        coords=coords,
        prob=prob_grid.ravel(),
        presence=presence_grid.ravel(),
        config=config,
    )


# Re-export utility for callers who want to derive a tweaked config
def field_meta_from(config: FieldConfig, **overrides: Any) -> FieldConfig:
    """Petit helper pour cloner une config avec quelques champs modifiés."""
    return replace(config, **overrides)


__all__ = ["Field", "FieldConfig", "field_meta_from", "simulate_field"]
