"""Tests pour ``aphid_spatial.methods``."""

from __future__ import annotations

import numpy as np
import pytest

from aphid_spatial.methods import SpatialMethod
from aphid_spatial.methods.geostatistics import OrdinaryKrigingIndicator
from aphid_spatial.simulation import (
    Field,
    FieldConfig,
    SensorConfig,
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
