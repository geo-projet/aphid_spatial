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


def test_obs_binary(field: Field) -> None:
    r = place_sensors(field, SensorConfig(n_sensors=20, seed=0))
    assert set(np.unique(r.obs).tolist()).issubset({0, 1})


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


def test_fnr_flips_positives(field: Field) -> None:
    """Avec fnr=1, tous les vrais positifs doivent devenir 0."""
    cfg = SensorConfig(n_sensors=50, placement="uniform", fnr=1.0, fpr=0.0, seed=3)
    r = place_sensors(field, cfg)
    assert int(r.obs.sum()) == 0


def test_fpr_flips_negatives(field: Field) -> None:
    """Avec fpr=1 et fnr=0, toutes les obs doivent être 1."""
    cfg = SensorConfig(n_sensors=50, placement="uniform", fnr=0.0, fpr=1.0, seed=3)
    r = place_sensors(field, cfg)
    assert int(r.obs.sum()) == r.obs.size


def test_sensor_radius_window(field: Field) -> None:
    """Avec sensor_radius>0, l'observation prend le max sur un voisinage."""
    cfg0 = SensorConfig(n_sensors=20, placement="uniform", sensor_radius=0, seed=5)
    cfg1 = SensorConfig(n_sensors=20, placement="uniform", sensor_radius=2, seed=5)
    r0 = place_sensors(field, cfg0)
    r1 = place_sensors(field, cfg1)
    # mêmes capteurs (même seed) ; r1 ≥ r0 partout (max sur fenêtre)
    np.testing.assert_array_equal(r0.sensor_idx, r1.sensor_idx)
    assert (r1.obs >= r0.obs).all()


def test_invalid_n_sensors(field: Field) -> None:
    with pytest.raises(ValueError):
        place_sensors(field, SensorConfig(n_sensors=0))
    with pytest.raises(ValueError):
        place_sensors(field, SensorConfig(n_sensors=field.config.n_rows * field.config.n_cols + 1))
