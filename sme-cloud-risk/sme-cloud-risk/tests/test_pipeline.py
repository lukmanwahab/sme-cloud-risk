"""
Test suite for the cloudrisk pipeline.

Purpose : guard the properties that matter for a credible modelling project:
          reproducibility of the simulator, schema stability, leakage-safe
          preprocessing, sensible model performance and a control-gap table
          that recovers the strongest effects built into the simulator.
Run     : pytest -q
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from cloudrisk import config, explain, features, simulate, train
from cloudrisk.simulate import COLUMN_DESCRIPTIONS, CONTROL_EFFECTS


@pytest.fixture(scope="module")
def survey():
    return simulate.simulate_sme_survey(n_firms=1500, seed=7)


@pytest.fixture(scope="module")
def split(survey):
    X, y = features.split_xy(survey)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)


# ------------------------------------------------------------ simulator ----
def test_simulator_is_reproducible():
    a = simulate.simulate_sme_survey(n_firms=300, seed=1)
    b = simulate.simulate_sme_survey(n_firms=300, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_schema_matches_documentation(survey):
    assert list(survey.columns) == list(COLUMN_DESCRIPTIONS)


def test_incident_rate_is_realistic(survey):
    rate = survey[config.TARGET].mean()
    assert 0.15 < rate < 0.45, f"incident rate {rate:.3f} outside plausible band"


def test_missingness_only_in_controls(survey):
    assert survey[config.CONTROLS].isna().any().any()
    non_control = [c for c in survey.columns if c not in config.CONTROLS]
    assert not survey[non_control].isna().any().any()


# ------------------------------------------------------------- features ----
def test_derived_features_present(survey):
    X, _ = features.split_xy(survey)
    for col in features.DERIVED:
        assert col in X.columns
    assert "firm_id" not in X.columns and config.TARGET not in X.columns


def test_preprocessor_has_no_nans_after_fit(split):
    X_tr, X_te, _, _ = split
    prep = features.make_preprocessor().fit(X_tr)
    Xt = prep.transform(X_te)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(Xt).any()


# ---------------------------------------------------------------- models ----
def test_cv_benchmark_beats_chance(split):
    X_tr, _, y_tr, _ = split
    results, _ = train.benchmark(X_tr, y_tr, n_folds=3, seed=7, tune=False)
    assert set(results["model"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert (results["roc_auc_mean"] > 0.65).all()


def test_selected_model_generalises(split):
    X_tr, X_te, y_tr, y_te = split
    results, candidates = train.benchmark(X_tr, y_tr, n_folds=3, seed=7, tune=False)
    name, best = train.select_and_fit(results, candidates, X_tr, y_tr)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
    assert auc > 0.68, f"{name} hold-out AUC {auc:.3f} too low"


# ------------------------------------------------------------- explain ----
def test_control_gap_ranking_recovers_strong_effects(split):
    """The three largest protective effects in the simulator should all be
    protective (OR < 1) in the logistic baseline, and at least two of them
    should sit in the top five control gaps."""
    X_tr, X_te, y_tr, y_te = split
    _, candidates = train.benchmark(X_tr, y_tr, n_folds=3, seed=7, tune=False)
    fitted = train.fit_all(candidates, X_tr, y_tr)
    ors = explain.odds_ratios(fitted["logistic_regression"], X_tr, y_tr).set_index("feature")
    strongest = sorted(CONTROL_EFFECTS, key=CONTROL_EFFECTS.get)[:3]
    assert (ors.loc[strongest, "odds_ratio"] < 1).all()

    sv, _, names = explain.shap_values(fitted["logistic_regression"], X_te, seed=7)
    imp = explain.shap_importance(sv, names)
    gaps = explain.control_gap_table(imp, ors.reset_index(), pd.concat([X_tr, X_te]), top_n=5)
    assert len(set(strongest) & set(gaps["feature"])) >= 2


# ------------------------------------------------------------ new stages ----
def test_search_spaces_cover_every_candidate():
    assert set(train.search_spaces()) == set(train.build_candidates())


def test_bootstrap_ci_brackets_estimate(split):
    from cloudrisk import uncertainty
    X_tr, X_te, y_tr, y_te = split
    pipe = train.build_candidates(seed=7)["logistic_regression"].fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    boot = uncertainty.bootstrap_metrics(y_te, p, n_boot=200, seed=7)
    assert ((boot["ci_low"] <= boot["estimate"]) & (boot["estimate"] <= boot["ci_high"])).all()
    assert boot["n_boot"].min() > 180


def test_decision_curve_model_beats_treat_none_at_low_threshold(split):
    from cloudrisk import decision
    X_tr, X_te, y_tr, y_te = split
    pipe = train.build_candidates(seed=7)["logistic_regression"].fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    dc = decision.decision_curve(y_te, p)
    row = dc.loc[(dc["threshold"] - 0.3).abs().idxmin()]
    assert row["net_benefit_model"] > row["net_benefit_treat_none"]
    assert row["net_benefit_model"] >= row["net_benefit_treat_all"]
    ct = decision.cost_sensitive_threshold(y_te, p)
    assert ct["optimal"].sum() >= 1


def test_counterfactual_gains_are_mostly_protective(survey, split):
    from cloudrisk import intervene
    X_tr, X_te, y_tr, y_te = split
    pipe = train.build_candidates(seed=7)["logistic_regression"].fit(X_tr, y_tr)
    raw = survey.loc[X_te.index]
    gains = intervene.counterfactual_gains(pipe, raw)
    # Adopting a control should lower predicted risk for the large majority
    # of firm-control pairs, and never change firms that already have it.
    assert (gains["risk_reduction"] > 0).mean() > 0.8
    recs = intervene.firm_recommendations(gains)
    assert recs["firm_id"].is_unique
    impact = intervene.portfolio_impact(gains)
    assert impact.iloc[0]["expected_incidents_averted"] > 0


def test_validator_rejects_bad_schema(survey):
    from cloudrisk import validate
    validate.validate_survey(survey)                     # clean frame passes
    bad = survey.copy()
    bad.loc[0, "employees"] = 500
    with pytest.raises(validate.SchemaError, match="employees"):
        validate.validate_survey(bad)
    with pytest.raises(validate.SchemaError, match="missing columns"):
        validate.validate_survey(survey.drop(columns=["sector"]))
