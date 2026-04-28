"""Tests pour ``aphid_spatial.simulation.field``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aphid_spatial.simulation import Field, FieldConfig, simulate_field


@pytest.fixture(scope="module")
def small_field() -> Field:
    """Petit champ pour tester rapidement (1000 plants)."""
    return simulate_field(FieldConfig(n_rows=20, n_cols=50, seed=123))


@pytest.fixture(scope="module")
def medium_field() -> Field:
    """Champ moyen pour des tests plus stables (10k plants)."""
    return simulate_field(FieldConfig(n_rows=50, n_cols=200, seed=7))


def test_dimensions(small_field: Field) -> None:
    cfg = small_field.config
    n = cfg.n_rows * cfg.n_cols
    assert small_field.coords.shape == (n, 2)
    assert small_field.prob.shape == (n,)
    assert small_field.presence.shape == (n,)
    assert small_field.coords.dtype == np.float64
    assert small_field.presence.dtype == np.int8


def test_prob_range(small_field: Field) -> None:
    assert (small_field.prob >= 0.0).all()
    assert (small_field.prob <= 1.0).all()


def test_presence_binary(small_field: Field) -> None:
    uniq = np.unique(small_field.presence)
    assert set(uniq.tolist()).issubset({0, 1})


def test_to_grid_shape(small_field: Field) -> None:
    cfg = small_field.config
    assert small_field.to_grid("prob").shape == (cfg.n_rows, cfg.n_cols)
    assert small_field.to_grid("presence").shape == (cfg.n_rows, cfg.n_cols)


def test_reproducibility() -> None:
    cfg = FieldConfig(n_rows=20, n_cols=40, seed=42)
    f1 = simulate_field(cfg)
    f2 = simulate_field(cfg)
    np.testing.assert_array_equal(f1.prob, f2.prob)
    np.testing.assert_array_equal(f1.presence, f2.presence)


def test_seed_changes_output() -> None:
    f1 = simulate_field(FieldConfig(n_rows=20, n_cols=40, seed=1))
    f2 = simulate_field(FieldConfig(n_rows=20, n_cols=40, seed=2))
    assert not np.allclose(f1.prob, f2.prob)


def test_prevalence_above_baseline(medium_field: Field) -> None:
    """Avec edge + GRF + hotspots, la prévalence empirique dépasse nettement
    la base (les paramètres par défaut amplifient ``logit(p)`` partout).
    On vérifie simplement qu'elle reste un Bernoulli sain dans (0, 1)."""
    base = medium_field.config.base_prevalence
    emp = float(medium_field.presence.mean())
    # Le tirage doit produire un mélange (au moins quelques 0 et quelques 1)
    assert 0.0 < emp < 1.0
    # La prévalence empirique doit être supérieure à la base (effet edge/hotspots)
    assert emp >= base


def test_edge_effect_visible() -> None:
    """La probabilité moyenne près du bord doit dépasser celle au centre.

    On utilise un champ assez grand pour que le centre soit hors de la
    zone d'influence ``edge_lambda_m``.
    """
    cfg = FieldConfig(n_rows=200, n_cols=200, edge_lambda_m=10.0, seed=13)
    f = simulate_field(cfg)
    spacing = cfg.spacing_m
    rows = np.arange(cfg.n_rows)
    cols = np.arange(cfg.n_cols)
    dy = np.minimum(rows, cfg.n_rows - 1 - rows) * spacing
    dx = np.minimum(cols, cfg.n_cols - 1 - cols) * spacing
    dist = np.minimum(dy[:, None], dx[None, :]).ravel()
    near_edge = dist < (cfg.edge_lambda_m * 0.3)
    far_from_edge = dist > (cfg.edge_lambda_m * 2.0)
    assert near_edge.any() and far_from_edge.any()
    p = f.prob
    assert p[near_edge].mean() > p[far_from_edge].mean(), (
        f"edge mean={p[near_edge].mean():.3f}, "
        f"center mean={p[far_from_edge].mean():.3f}"
    )


def test_save_load_roundtrip(tmp_path: Path, small_field: Field) -> None:
    path = tmp_path / "field.npz"
    small_field.save(path)
    loaded = Field.load(path)
    np.testing.assert_array_equal(small_field.prob, loaded.prob)
    np.testing.assert_array_equal(small_field.presence, loaded.presence)
    np.testing.assert_array_equal(small_field.coords, loaded.coords)
    assert loaded.config.seed == small_field.config.seed
    assert loaded.config.n_hotspots == small_field.config.n_hotspots


def test_variogram_recovers_range(medium_field: Field) -> None:
    """Le variogramme empirique du logit(p) doit retrouver une portée à
    l'ordre de grandeur de ``matern_range_m`` (test à seuil large)."""
    import gstools as gs
    from scipy.special import logit

    cfg = medium_field.config
    p = medium_field.prob
    p_clipped = np.clip(p, 1e-3, 1 - 1e-3)
    z = logit(p_clipped)
    coords = medium_field.coords

    # Sous-échantillonnage pour le variogramme (sinon trop lent)
    rng = np.random.default_rng(0)
    sub = rng.choice(coords.shape[0], size=min(2000, coords.shape[0]), replace=False)
    bin_centers, gamma = gs.vario_estimate(
        (coords[sub, 0], coords[sub, 1]),
        z[sub],
        bin_edges=np.linspace(0.5, 4.0 * cfg.matern_range_m, 16),
    )
    # Ajustement Matérn
    model = gs.Matern(dim=2, nu=cfg.matern_nu)
    model.fit_variogram(bin_centers, gamma, nugget=False)
    estimated_range = float(model.len_scale)
    target = cfg.matern_range_m
    # L'effet edge + hotspots gonfle l'échelle apparente -> seuil large
    assert 0.25 * target <= estimated_range <= 4.0 * target, (
        f"portée estimée {estimated_range:.2f} hors [{0.25 * target:.2f}, {4 * target:.2f}]"
    )
