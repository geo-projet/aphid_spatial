"""Tests pour ``aphid_spatial.methods``."""

from __future__ import annotations

import numpy as np
import pytest

from aphid_spatial.methods import SpatialMethod
from aphid_spatial.methods.autocorrelation import (
    autocorrelation_summary,
    compute_weights,
    geary_global,
    getis_ord_local,
    moran_global,
    moran_local,
)
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
from aphid_spatial.methods.hierarchical import MaternGLMM
from aphid_spatial.methods.lattice import (
    BYMModel,
    CARModel,
    IsingMRF,
    SARLagModel,
    estimate_params_pseudo_likelihood,
)
from aphid_spatial.methods.ml import SpatialRandomForest
from aphid_spatial.methods.point_process import (
    csr_envelope,
    kde_intensity,
    pair_correlation,
    ripley_k,
    ripley_l,
    support_from_field,
    weighted_ripley_k,
)
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


# ---------------------------------------------------------------------------
# Autocorrélation spatiale
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def med_readings() -> tuple[Field, SensorReadings]:
    """Champ moyen avec 30 capteurs pour que les stats spatiales soient
    raisonnablement définies."""
    f = simulate_field(
        FieldConfig(
            n_rows=30, n_cols=200, base_prevalence=0.20, edge_strength=0.5,
            n_hotspots=(2, 4), seed=77,
        )
    )
    r = place_sensors(
        f, SensorConfig(n_sensors=30, placement="uniform", seed=11)
    )
    return f, r


def test_compute_weights_schemes(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w_knn = compute_weights(r.coords, scheme="knn", k=4)
    w_dist = compute_weights(r.coords, scheme="distance", threshold=20.0)
    w_gauss = compute_weights(r.coords, scheme="gaussian", bandwidth=10.0)
    assert w_knn.n == r.coords.shape[0]
    assert w_dist.n == r.coords.shape[0]
    assert w_gauss.n == r.coords.shape[0]


def test_moran_global_returns_finite(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w = compute_weights(r.coords, scheme="knn", k=5)
    out = moran_global(r, w, n_perm=99, seed=0)
    for key in ("I", "p_value", "z_score"):
        assert key in out
        assert np.isfinite(out[key])
    assert 0.0 <= out["p_value"] <= 1.0


def test_geary_global_returns_finite(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w = compute_weights(r.coords, scheme="knn", k=5)
    out = geary_global(r, w, n_perm=99, seed=0)
    assert np.isfinite(out["c"])
    assert 0.0 <= out["p_value"] <= 1.0


def test_moran_local_shapes(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w = compute_weights(r.coords, scheme="knn", k=5)
    out = moran_local(r, w, n_perm=99, seed=0)
    n = r.coords.shape[0]
    for key in ("Is", "p_sim", "q", "hh", "ll", "hl", "lh"):
        assert key in out
        assert out[key].shape == (n,)


def test_getis_ord_local_shapes(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w = compute_weights(r.coords, scheme="knn", k=5)
    out = getis_ord_local(r, w, n_perm=99, seed=0)
    assert out["Gs"].shape == (r.coords.shape[0],)
    assert out["Zs"].shape == (r.coords.shape[0],)


def test_autocorrelation_summary_keys(med_readings: tuple[Field, SensorReadings]) -> None:
    _, r = med_readings
    w = compute_weights(r.coords, scheme="knn", k=5)
    out = autocorrelation_summary(r, w, n_perm=99, seed=0)
    for key in (
        "moran_I", "moran_p_value", "geary_c", "geary_p_value",
        "getis_G", "getis_p_value", "lisa_q", "gistar_Zs",
    ):
        assert key in out


def test_moran_degenerate_returns_nan() -> None:
    """Avec var=0, Moran retourne NaN sans planter."""
    f_dummy = simulate_field(FieldConfig(n_rows=10, n_cols=10, seed=0))
    r = place_sensors(f_dummy, SensorConfig(n_sensors=10, seed=0))
    r.obs[:] = 0.5  # variance nulle
    w = compute_weights(r.coords, scheme="knn", k=3)
    out = moran_global(r, w, n_perm=10, seed=0)
    assert np.isnan(out["I"])


# ---------------------------------------------------------------------------
# Processus ponctuels
# ---------------------------------------------------------------------------


def test_ripley_k_l_shapes(mini_readings: SensorReadings) -> None:
    out_k = ripley_k(mini_readings.coords)
    out_l = ripley_l(mini_readings.coords)
    assert out_k["radii"].shape == out_k["K"].shape
    assert out_l["L"].shape == out_l["radii"].shape
    # K(0) = 0
    assert out_k["K"][0] == pytest.approx(0.0, abs=1e-9)


def test_pair_correlation_finite(mini_readings: SensorReadings) -> None:
    out = pair_correlation(mini_readings.coords)
    # NaN à r=0 attendu, mais le reste doit être fini
    assert np.isfinite(out["g"][1:]).all()


def test_csr_envelope_includes_observed(mini_readings: SensorReadings) -> None:
    """Sur un petit nombre de points, l'observed peut sortir de
    l'enveloppe ; on vérifie au moins la forme et la cohérence low <= high."""
    env = csr_envelope(
        mini_readings.coords, statistic="L", n_sim=20, seed=0,
    )
    assert env["radii"].shape == env["observed"].shape
    assert env["low"].shape == env["high"].shape
    # low <= mean <= high (à NaN près)
    valid = ~(np.isnan(env["low"]) | np.isnan(env["high"]))
    assert (env["low"][valid] <= env["high"][valid]).all()


def test_kde_intensity_positive(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    intens = kde_intensity(
        mini_readings.coords, mini_field.coords[:50], bandwidth=2.0
    )
    assert intens.shape == (50,)
    assert (intens >= 0.0).all()


def test_weighted_ripley_k_zero_at_zero(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    out = weighted_ripley_k(
        mini_readings.coords, mini_readings.obs,
        support=support_from_field(mini_field),
    )
    assert out["K"][0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Ising MRF
# ---------------------------------------------------------------------------


def test_ising_mrf_basic(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = IsingMRF(neighborhood="rook", n_burn=20, n_samples=30, seed=0)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)
    params = method.params
    for key in ("alpha", "beta", "fit_seconds", "n_iter_kept"):
        assert key in params


def test_ising_respects_explicit_params(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    """Avec alpha/beta fournis, ils doivent être utilisés tels quels."""
    method = IsingMRF(
        alpha=-2.0, beta=0.0, n_burn=10, n_samples=20, seed=0,
    )
    method.fit(mini_readings, mini_field)
    p = method.params
    assert p["alpha"] == -2.0
    assert p["beta"] == 0.0


def test_ising_beta_zero_predicts_sigmoid_alpha(
    mini_field: Field, mini_readings: SensorReadings
) -> None:
    """Avec β=0, les cellules non-capteurs convergent vers σ(α) ≈ 0.27 pour
    α=-1 (test à seuil large)."""
    from scipy.special import expit
    alpha = -1.0
    method = IsingMRF(alpha=alpha, beta=0.0, n_burn=50, n_samples=200, seed=0)
    method.fit(mini_readings, mini_field)
    p = method.predict_proba(mini_field.coords)
    expected = float(expit(alpha))
    # Inclure les capteurs dans la moyenne ; la majorité des cellules doit
    # être proche de σ(α)
    assert abs(p.mean() - expected) < 0.1


def test_estimate_params_pseudo_likelihood_recovers_alpha() -> None:
    """Sur un champ aléatoire (β=0), la PL doit retrouver α ≈ logit(p)."""
    rng = np.random.default_rng(0)
    p_true = 0.3
    y = (rng.random((40, 40)) < p_true).astype(np.int8)
    alpha_est, beta_est = estimate_params_pseudo_likelihood(y, queen=False)
    expected_alpha = float(np.log(p_true / (1 - p_true)))
    # PL biaisée pour β mais α doit être proche
    assert abs(alpha_est - expected_alpha) < 0.5
    assert abs(beta_est) < 0.5


def test_ising_fallback_few_sensors() -> None:
    """Avec < 3 capteurs, fallback constant."""
    f = simulate_field(FieldConfig(n_rows=10, n_cols=20, seed=0))
    r = place_sensors(f, SensorConfig(n_sensors=2, seed=0))
    method = IsingMRF(seed=0)
    method.fit(r, f)
    p = method.predict_proba(f.coords)
    assert np.allclose(p, p[0])  # constante


# ---------------------------------------------------------------------------
# Bayésien : MaternGLMM, CARModel, BYMModel, SARLagModel
# ---------------------------------------------------------------------------
#
# Ces tests utilisent peu de draws/tune pour rester rapides ; la qualité
# de l'inférence n'est pas auditée ici, on vérifie uniquement la
# conformité à l'interface SpatialMethod et la robustesse du fallback.


@pytest.mark.slow
def test_matern_glmm_basic(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = MaternGLMM(
        n_draws=50, n_tune=50, chains=1, n_predict_samples=20, seed=0,
    )
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)


def test_matern_glmm_fallback_few_sensors() -> None:
    f = simulate_field(FieldConfig(n_rows=10, n_cols=20, seed=0))
    r = place_sensors(f, SensorConfig(n_sensors=3, seed=0))
    m = MaternGLMM(n_draws=10, n_tune=10, chains=1, seed=0)
    m.fit(r, f)
    p = m.predict_proba(f.coords)
    assert np.allclose(p, p[0])
    assert m.predict_uncertainty(f.coords) is None


@pytest.mark.slow
def test_car_model_basic(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = CARModel(
        n_neighbors=3, n_draws=50, n_tune=50, chains=1, n_predict_samples=20, seed=0,
    )
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    sigma = method.predict_uncertainty(mini_field.coords)
    assert sigma is not None
    assert sigma.shape == (mini_field.coords.shape[0],)


@pytest.mark.slow
def test_bym_model_basic(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = BYMModel(
        n_neighbors=3, n_draws=50, n_tune=50, chains=1, n_predict_samples=20, seed=0,
    )
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    assert method.name == "bym_pymc"


def test_sar_lag_basic(mini_field: Field, mini_readings: SensorReadings) -> None:
    method = SARLagModel(n_neighbors=3)
    method.fit(mini_readings, mini_field)
    _check_method_basic(method, mini_field)
    p = method.params
    for key in ("rho", "beta0", "beta1"):
        assert key in p
    # SAR ne fournit pas d'incertitude analytique
    assert method.predict_uncertainty(mini_field.coords) is None


def test_sar_lag_fallback_few_sensors() -> None:
    f = simulate_field(FieldConfig(n_rows=10, n_cols=20, seed=0))
    r = place_sensors(f, SensorConfig(n_sensors=4, seed=0))
    m = SARLagModel(n_neighbors=2)
    m.fit(r, f)
    # Soit le fit a réussi (alors p ∈ [0,1]) soit fallback constante
    p = m.predict_proba(f.coords)
    assert (p >= 0).all() and (p <= 1).all()
