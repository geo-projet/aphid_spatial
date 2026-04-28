"""Statistiques d'autocorrélation spatiale.

Avec ~20 capteurs c'est limite mais utile pour caractériser. Les fonctions
exposées ici renvoient des dictionnaires de scalaires (pour les statistiques
globales) ou de tableaux (pour les statistiques locales). Elles ne sont pas
appliquées aux 100k cellules de la grille — uniquement aux observations
ponctuelles des capteurs.

Trois schémas de matrice de poids spatiaux sont supportés :

* ``knn``      : K plus proches voisins (par défaut K=5).
* ``distance`` : voisinage à seuil de distance (m).
* ``gaussian`` : noyau gaussien décroissant + normalisation par ligne.

Les statistiques implémentées :

* **Moran I** global et local (LISA, avec quadrants HH/LL/HL/LH).
* **Geary c** global.
* **Getis-Ord G** global et local **G\*ᵢ** (hot/cold spots).

Garde-fou : si ``var(obs) == 0`` ou ``n_sensors < 4``, les fonctions retournent
des ``NaN`` et émettent un warning — ces statistiques ne sont pas définies
dans ce régime dégénéré.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
from esda.geary import Geary
from esda.getisord import G, G_Local
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, DistanceBand, W
from numpy.typing import NDArray

from aphid_spatial.simulation.sensors import SensorReadings

logger = logging.getLogger(__name__)

WeightsScheme = Literal["knn", "distance", "gaussian"]


def _gaussian_weights(coords: NDArray[np.floating], bandwidth: float) -> W:
    """Construit ``W`` avec poids gaussiens normalisés par ligne."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff**2).sum(axis=-1)
    raw = np.exp(-d2 / (2.0 * bandwidth**2))
    np.fill_diagonal(raw, 0.0)
    row_sum = raw.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    norm = raw / row_sum
    neighbors = {i: list(np.flatnonzero(norm[i] > 0)) for i in range(n)}
    weights = {i: list(norm[i, neighbors[i]]) for i in range(n)}
    return W(neighbors, weights, silence_warnings=True)


def compute_weights(
    coords: NDArray[np.floating],
    scheme: WeightsScheme = "knn",
    *,
    k: int = 5,
    threshold: float | None = None,
    bandwidth: float | None = None,
) -> W:
    """Construit la matrice de poids spatiaux pour les capteurs.

    Parameters
    ----------
    coords : NDArray
        Positions des capteurs (n, 2).
    scheme : {"knn", "distance", "gaussian"}
        Type de matrice. Pour ``distance``, ``threshold`` est requis (m).
        Pour ``gaussian``, ``bandwidth`` est requis (m).
    k : int
        Nombre de voisins pour ``knn`` (défaut 5).
    threshold : float
        Seuil de distance pour ``distance`` (m).
    bandwidth : float
        Largeur de bande pour ``gaussian`` (m).
    """
    if scheme == "knn":
        k_eff = min(k, coords.shape[0] - 1)
        w = KNN.from_array(coords, k=k_eff)
    elif scheme == "distance":
        if threshold is None:
            raise ValueError("scheme='distance' requires threshold (m)")
        w = DistanceBand.from_array(coords, threshold=threshold, silence_warnings=True)
    elif scheme == "gaussian":
        if bandwidth is None:
            raise ValueError("scheme='gaussian' requires bandwidth (m)")
        w = _gaussian_weights(coords, bandwidth=bandwidth)
    else:  # pragma: no cover
        raise ValueError(f"unknown scheme {scheme!r}")
    w.transform = "r"  # row-standardisation (somme des poids par ligne = 1)
    return w


def _is_degenerate(values: NDArray[np.floating]) -> bool:
    return values.size < 4 or float(np.var(values)) == 0.0


def _nan_dict(*keys: str) -> dict[str, float]:
    return dict.fromkeys(keys, float("nan"))


class _SeedScope:
    """Context manager to set numpy global RNG state for reproducibility.

    ``esda`` versions 2.5–2.7 do not all accept a ``seed`` kwarg ; les
    permutations utilisent l'état global de ``numpy.random``. On sauve
    l'état avant, on fixe le seed, puis on restaure à la sortie.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._state: object | None = None

    def __enter__(self) -> _SeedScope:
        self._state = np.random.get_state()
        np.random.seed(self._seed)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._state is not None:
            np.random.set_state(self._state)  # type: ignore[arg-type]


def moran_global(
    readings: SensorReadings, w: W, *, n_perm: int = 999, seed: int = 0
) -> dict[str, float]:
    """Moran I global avec permutations Monte Carlo."""
    y = np.asarray(readings.obs, dtype=np.float64)
    if _is_degenerate(y):
        logger.warning("Moran global : régime dégénéré, retourne NaN")
        return _nan_dict("I", "p_value", "z_score")
    with warnings.catch_warnings(), _SeedScope(seed):
        warnings.simplefilter("ignore")
        m = Moran(y, w, permutations=n_perm)
    return {"I": float(m.I), "p_value": float(m.p_sim), "z_score": float(m.z_sim)}


def geary_global(
    readings: SensorReadings, w: W, *, n_perm: int = 999, seed: int = 0
) -> dict[str, float]:
    """Geary c global. ``c < 1`` ⇒ autocorrélation positive."""
    y = np.asarray(readings.obs, dtype=np.float64)
    if _is_degenerate(y):
        logger.warning("Geary global : régime dégénéré, retourne NaN")
        return _nan_dict("c", "p_value", "z_score")
    with warnings.catch_warnings(), _SeedScope(seed):
        warnings.simplefilter("ignore")
        g = Geary(y, w, permutations=n_perm)
    return {"c": float(g.C), "p_value": float(g.p_sim), "z_score": float(g.z_sim)}


def getis_ord_global(
    readings: SensorReadings, w: W, *, n_perm: int = 999, seed: int = 0
) -> dict[str, float]:
    """Getis-Ord G global. ``G > E[G]`` ⇒ clustering de valeurs hautes."""
    y = np.asarray(readings.obs, dtype=np.float64)
    if _is_degenerate(y):
        logger.warning("Getis-Ord global : régime dégénéré, retourne NaN")
        return _nan_dict("G", "p_value", "z_score")
    with warnings.catch_warnings(), _SeedScope(seed):
        warnings.simplefilter("ignore")
        g = G(y, w, permutations=n_perm)
    return {"G": float(g.G), "p_value": float(g.p_sim), "z_score": float(g.z_sim)}


def moran_local(
    readings: SensorReadings, w: W, *, n_perm: int = 999, seed: int = 0
) -> dict[str, NDArray[np.floating] | NDArray[np.integer]]:
    """LISA (Moran local).

    Returns
    -------
    dict
        ``Is`` : statistique locale (n_sensors,)
        ``p_sim`` : p-values locales
        ``q`` : quadrant LISA, ``{1=HH, 2=LH, 3=LL, 4=HL}``
        ``hh`` / ``ll`` / ``hl`` / ``lh`` : booléens (cluster significatif p<0.05)
    """
    y = np.asarray(readings.obs, dtype=np.float64)
    if _is_degenerate(y):
        logger.warning("Moran local : régime dégénéré, retourne tableaux NaN")
        n = y.size
        return {
            "Is": np.full(n, np.nan),
            "p_sim": np.full(n, np.nan),
            "q": np.zeros(n, dtype=np.int64),
            "hh": np.zeros(n, dtype=bool),
            "ll": np.zeros(n, dtype=bool),
            "hl": np.zeros(n, dtype=bool),
            "lh": np.zeros(n, dtype=bool),
        }
    with warnings.catch_warnings(), _SeedScope(seed):
        warnings.simplefilter("ignore")
        m = Moran_Local(y, w, permutations=n_perm)
    sig = m.p_sim < 0.05
    return {
        "Is": np.asarray(m.Is, dtype=np.float64),
        "p_sim": np.asarray(m.p_sim, dtype=np.float64),
        "q": np.asarray(m.q, dtype=np.int64),
        "hh": (m.q == 1) & sig,
        "lh": (m.q == 2) & sig,
        "ll": (m.q == 3) & sig,
        "hl": (m.q == 4) & sig,
    }


def getis_ord_local(
    readings: SensorReadings, w: W, *, n_perm: int = 999, seed: int = 0
) -> dict[str, NDArray[np.floating]]:
    """Getis-Ord Gᵢ\\* local (hot / cold spots)."""
    y = np.asarray(readings.obs, dtype=np.float64)
    if _is_degenerate(y):
        logger.warning("Getis-Ord local : régime dégénéré, retourne NaN")
        n = y.size
        return {
            "Gs": np.full(n, np.nan),
            "p_sim": np.full(n, np.nan),
            "Zs": np.full(n, np.nan),
        }
    with warnings.catch_warnings(), _SeedScope(seed):
        warnings.simplefilter("ignore")
        g = G_Local(y, w, permutations=n_perm, star=True)
    return {
        "Gs": np.asarray(g.Gs, dtype=np.float64),
        "p_sim": np.asarray(g.p_sim, dtype=np.float64),
        "Zs": np.asarray(g.Zs, dtype=np.float64),
    }


def autocorrelation_summary(
    readings: SensorReadings,
    w: W,
    *,
    n_perm: int = 999,
    seed: int = 0,
) -> dict[str, float | NDArray[np.floating] | NDArray[np.integer]]:
    """Résumé global + local en une passe."""
    out: dict[str, float | NDArray[np.floating] | NDArray[np.integer]] = {}
    for name, val in moran_global(readings, w, n_perm=n_perm, seed=seed).items():
        out[f"moran_{name}"] = val
    for name, val in geary_global(readings, w, n_perm=n_perm, seed=seed).items():
        out[f"geary_{name}"] = val
    for name, val in getis_ord_global(readings, w, n_perm=n_perm, seed=seed).items():
        out[f"getis_{name}"] = val
    local_m = moran_local(readings, w, n_perm=n_perm, seed=seed)
    local_g = getis_ord_local(readings, w, n_perm=n_perm, seed=seed)
    out["lisa_q"] = local_m["q"]
    out["lisa_p_sim"] = local_m["p_sim"]
    out["gistar_Zs"] = local_g["Zs"]
    out["gistar_p_sim"] = local_g["p_sim"]
    return out


__all__ = [
    "WeightsScheme",
    "autocorrelation_summary",
    "compute_weights",
    "geary_global",
    "getis_ord_global",
    "getis_ord_local",
    "moran_global",
    "moran_local",
]
