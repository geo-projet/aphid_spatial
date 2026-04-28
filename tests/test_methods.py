"""Tests pour ``aphid_spatial.methods``."""

from __future__ import annotations

import numpy as np
import pytest

from aphid_spatial.methods import SpatialMethod
from aphid_spatial.methods.exploration import (
    BaselineConstant,
    descriptive_summary,
    edge_distance,
    inter_sensor_distances,
    nearest_sensor_distance,
    nearest_sensor_value,
)
from aphid_spatial.methods.geostatistics import (
    IndicatorKrigingThreshold,
    OrdinaryKrigingIndicator,
    UniversalKrigingEdge,
)
from aphid_spatial.methods.gp import MaternGPClassifier, MaternGPRegressor
from aphid_spatial.methods.ml import SpatialRandomForest
from aphid_spatial.methods.sadie import SADIE, aggregation_index, local_indices
from aphid_spatial.simulation import (
    Field,
    FieldConfig,
    SensorConfig,
    SensorReadings,
    place_sensors,
    simulate_field,
)


@pytest.fixture(scope="module")
def mini_field() -> Field:
    """Mini-champ 10×100 pour smoke tests rapides."""
    # Prévalence boostée pour s'assurer d'avoir des positifs sur 5 capteurs
    return simulate_field(
        FieldConfig(
            n_rows=10,
            n_cols=100,
            base_prevalence=0.20,
            edge_strength=0.5,
            n_hotspots=(2, 4),
            seed=99,
        )
    )


def test_protocol_satisfied() -> None:
    method = OrdinaryKrigingIndicator()
    assert isinstance(method, SpatialMethod)
    assert method.name == "ordinary_kriging_indicator"


def test_kriging_fits_and_predicts(mini_field: Field) -> None:
    readings = place_sensors(
        mini_field, SensorConfig(n_sensors=8, placement="uniform", seed=7)
    )
    method = OrdinaryKrigingIndicator()
    method.fit(readings, mini_field)
    p_pred = method.predict_proba(mini_field.coords)
    assert p_pred.shape == (mini_field.coords.shape[0],)
    assert (p_pred >= 0.0).all()
    assert (p_pred <= 1.0).all()


def test_kriging_uncertainty_shape(mini_field: Field) -> None:
    readings = place_sensors(
        mini_field, SensorConfig(n_sensors=8, placement="uniform", seed=7)
    )
    method = OrdinaryKrigingIndicator()
    method.fit(readings, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)
    assert (sigma >= 0.0).all()


def test_fallback_constant_when_zero_variance(mini_field: Field) -> None:
    """Si toutes les observations sont identiques, on tombe sur la baseline."""
    readings = place_sensors(
        mini_field, SensorConfig(n_sensors=8, placement="uniform", seed=7)
    )
    # Force toutes les obs à 0
    readings.obs[:] = 0
    method = OrdinaryKrigingIndicator()
    method.fit(readings, mini_field)
    p_pred = method.predict_proba(mini_field.coords)
    assert np.allclose(p_pred, 0.0)
    assert method.predict_uncertainty(mini_field.coords) is None


def test_kriging_predicts_at_arbitrary_coords(mini_field: Field) -> None:
    readings = place_sensors(
        mini_field, SensorConfig(n_sensors=8, placement="uniform", seed=7)
    )
    method = OrdinaryKrigingIndicator()
    method.fit(readings, mini_field)
    # 5 points arbitraires dans le champ
    rng = np.random.default_rng(0)
    cfg = mini_field.config
    x_max = (cfg.n_cols - 1) * cfg.spacing_m
    y_max = (cfg.n_rows - 1) * cfg.spacing_m
    query = np.column_stack([rng.uniform(0, x_max, 5), rng.uniform(0, y_max, 5)])
    p = method.predict_proba(query)
    assert p.shape == (5,)
    assert (p >= 0.0).all() and (p <= 1.0).all()


# ---------------------------------------------------------------------------
# Méthodes ajoutées : exploration, geostatistics étendu, GP, ML, SADIE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_readings(mini_field: Field) -> SensorReadings:
    return place_sensors(
        mini_field, SensorConfig(n_sensors=8, placement="uniform", seed=7)
    )


def _check_method_basic(method: SpatialMethod, field: Field) -> None:
    """Smoke-check : interface, sortie ∈ [0, 1], dimensions correctes."""
    assert isinstance(method, SpatialMethod)
    assert isinstance(method.name, str) and method.name
    p = method.predict_proba(field.coords)
    assert p.shape == (field.coords.shape[0],)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_baseline_constant(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = BaselineConstant()
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    p = method.predict_proba(mini_field.coords)
    assert np.allclose(p, mini_readings.obs.mean())
    assert method.predict_uncertainty(mini_field.coords) is None


def test_edge_distance_shape(mini_field: Field) -> None:
    d = edge_distance(mini_field)
    assert d.shape == (mini_field.coords.shape[0],)
    assert (d >= 0.0).all()


def test_nearest_sensor_helpers(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    d = nearest_sensor_distance(mini_readings, mini_field.coords)
    v = nearest_sensor_value(mini_readings, mini_field.coords)
    assert d.shape == (mini_field.coords.shape[0],)
    assert v.shape == (mini_field.coords.shape[0],)
    # En un point capteur, la distance est exactement 0
    d_at_sensors = nearest_sensor_distance(mini_readings, mini_readings.coords)
    np.testing.assert_allclose(d_at_sensors, 0.0)


def test_inter_sensor_distances_count(mini_readings: SensorReadings) -> None:
    n = mini_readings.coords.shape[0]
    inter = inter_sensor_distances(mini_readings)
    assert inter.shape == (n * (n - 1) // 2,)
    assert (inter > 0.0).all()


def test_descriptive_summary_keys(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    d = descriptive_summary(mini_readings, mini_field)
    for k in (
        "n_sensors",
        "obs_mean",
        "obs_std",
        "inter_sensor_dist_mean_m",
        "coverage_dist_mean_m",
    ):
        assert k in d


def test_universal_kriging_edge(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    method = UniversalKrigingEdge()
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)


def test_indicator_kriging_threshold(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    method = IndicatorKrigingThreshold(threshold=0.4)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)


def test_gp_regressor(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = MaternGPRegressor(n_restarts=0)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)
    assert (sigma >= 0.0).all()


def test_gp_classifier(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = MaternGPClassifier(threshold=0.4, n_restarts=0)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)


def test_random_forest(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = SpatialRandomForest(n_estimators=20, max_depth=5)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)


def test_sadie_method(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = SADIE(n_permutations=50)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    stats = method.stats
    for k in ("I_a", "p_value", "obs_metric", "perm_mean", "perm_std"):
        assert k in stats
    v = method.v_local
    assert v.shape == mini_readings.obs.shape


def test_aggregation_index_random_close_to_one() -> None:
    """Sur des valeurs i.i.d., I_a doit être proche de 1 (pas d'agrégation)."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(50, 2))
    values = rng.uniform(0, 1, size=50)
    out = aggregation_index(coords, values, n_permutations=200, seed=0)
    assert 0.7 < out["I_a"] < 1.3


def test_local_indices_zero_when_constant() -> None:
    coords = np.zeros((10, 2))
    values = np.full(10, 0.5)
    v = local_indices(coords, values)
    assert np.allclose(v, 0.0)
