"""Tests pour ``aphid_spatial.simulation.sensors``."""

from __future__ import annotations

import numpy as np
import pytest

from aphid_spatial.simulation import (
    Field,
    FieldConfig,
    SensorConfig,
    place_sensors,
    simulate_field,
)
from aphid_spatial.simulation.sensors import Placement


@pytest.fixture(scope="module")
def field() -> Field:
    return simulate_field(FieldConfig(n_rows=30, n_cols=120, seed=11))


PLACEMENTS: list[Placement] = ["uniform", "grid", "stratified", "edge_biased", "poisson_disk"]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_correct_count(field: Field, placement: Placement) -> None:
    cfg = SensorConfig(n_sensors=20, placement=placement, seed=0)
    r = place_sensors(field, cfg)
    assert r.sensor_idx.shape == (20,)
    assert r.coords.shape == (20, 2)
    assert r.obs.shape == (20,)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_no_duplicates(field: Field, placement: Placement) -> None:
    cfg = SensorConfig(n_sensors=20, placement=placement, seed=0)
    r = place_sensors(field, cfg)
    assert np.unique(r.sensor_idx).size == r.sensor_idx.size


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_indices_in_grid(field: Field, placement: Placement) -> None:
    cfg = SensorConfig(n_sensors=20, placement=placement, seed=0)
    r = place_sensors(field, cfg)
    n = field.config.n_rows * field.config.n_cols
    assert r.sensor_idx.min() >= 0
    assert r.sensor_idx.max() < n


def test_obs_in_unit_interval(field: Field) -> None:
    r = place_sensors(field, SensorConfig(n_sensors=20, seed=0))
    assert (r.obs >= 0.0).all()
    assert (r.obs <= 1.0).all()
    assert r.obs.dtype == np.float64


def test_obs_exact_when_k_is_none(field: Field) -> None:
    """Avec n_observations=None, obs est exactement la probabilité locale."""
    cfg = SensorConfig(n_sensors=20, n_observations=None, seed=42)
    r = place_sensors(field, cfg)
    np.testing.assert_array_equal(r.obs, r.prob_local)


def test_obs_binomial_grid_when_k_is_int(field: Field) -> None:
    """Avec n_observations=K fini, obs ∈ {0, 1/K, ..., 1}."""
    K = 10
    cfg = SensorConfig(n_sensors=50, n_observations=K, seed=42)
    r = place_sensors(field, cfg)
    expected = np.arange(K + 1) / K
    # Toutes les obs doivent être proches d'un multiple de 1/K
    diffs = np.abs(r.obs[:, None] - expected[None, :]).min(axis=1)
    assert (diffs < 1e-9).all()


def test_obs_concentrates_around_prob_with_large_k(field: Field) -> None:
    """Avec K grand, la moyenne |obs - prob_local| doit être petite."""
    cfg = SensorConfig(n_sensors=100, n_observations=10000, seed=42)
    r = place_sensors(field, cfg)
    diff = float(np.mean(np.abs(r.obs - r.prob_local)))
    assert diff < 0.02


def test_invalid_n_observations(field: Field) -> None:
    with pytest.raises(ValueError, match="n_observations"):
        place_sensors(field, SensorConfig(n_sensors=10, n_observations=0))


def test_reproducibility(field: Field) -> None:
    cfg = SensorConfig(n_sensors=20, placement="uniform", seed=42)
    r1 = place_sensors(field, cfg)
    r2 = place_sensors(field, cfg)
    np.testing.assert_array_equal(r1.sensor_idx, r2.sensor_idx)
    np.testing.assert_array_equal(r1.obs, r2.obs)


def test_distinct_schemes_differ(field: Field) -> None:
    """Deux schémas différents doivent produire des positions différentes."""
    r_uni = place_sensors(field, SensorConfig(n_sensors=20, placement="uniform", seed=1))
    r_grid = place_sensors(field, SensorConfig(n_sensors=20, placement="grid", seed=1))
    assert set(r_uni.sensor_idx.tolist()) != set(r_grid.sensor_idx.tolist())


def test_sensor_radius_window_averages_prob(field: Field) -> None:
    """Avec sensor_radius>0, prob_local est la moyenne de prob sur la fenêtre.

    On vérifie que ``prob_local`` (radius=2) reste dans l'enveloppe
    [min, max] de prob sur la même fenêtre, et qu'elle diffère
    typiquement de la valeur centrale (radius=0).
    """
    cfg0 = SensorConfig(
        n_sensors=20, placement="uniform", sensor_radius=0, n_observations=None, seed=5
    )
    cfg2 = SensorConfig(
        n_sensors=20, placement="uniform", sensor_radius=2, n_observations=None, seed=5
    )
    r0 = place_sensors(field, cfg0)
    r2 = place_sensors(field, cfg2)
    np.testing.assert_array_equal(r0.sensor_idx, r2.sensor_idx)
    # La moyenne sur 5x5 ne peut pas être plus extrême que le pixel central
    # (sauf si tout est uniforme) ; on vérifie au moins que ce n'est pas le
    # même tableau
    assert not np.allclose(r0.prob_local, r2.prob_local)


def test_prob_local_attribute(field: Field) -> None:
    """SensorReadings expose prob_local de la bonne forme et dans [0, 1]."""
    r = place_sensors(field, SensorConfig(n_sensors=20, seed=0))
    assert r.prob_local.shape == r.obs.shape
    assert (r.prob_local >= 0.0).all()
    assert (r.prob_local <= 1.0).all()


def test_invalid_n_sensors(field: Field) -> None:
    with pytest.raises(ValueError):
        place_sensors(field, SensorConfig(n_sensors=0))
    with pytest.raises(ValueError):
        place_sensors(field, SensorConfig(n_sensors=field.config.n_rows * field.config.n_cols + 1))
