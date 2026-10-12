"""Phase 3/4 machinery tests on SYNTHETIC data (never research results)."""
import json

import numpy as np
import pandas as pd
import pytest

from trialsight import analysis, modeling
from trialsight.features import LeakageError, assert_no_leakage, feature_set, add_derived_features
from trialsight.storage import ResearchPopulationError
from v3_synthetic import synthetic_cohort, write_synthetic_research_db

FAST = modeling.ModelingConfig(cv_folds=3, n_bootstrap=40, permutation_repeats=2, min_events_train=10,
                               min_events_test=10, fast=True)


# ------------------------------------------------------------------ leakage
@pytest.mark.parametrize("bad", ["overall_status", "target_terminated", "why_stopped", "enrollment_count",
                                 "completion_date", "has_results", "nct_id", "lead_sponsor_name", "not_a_column"])
def test_post_outcome_and_unknown_features_rejected(bad):
    with pytest.raises(LeakageError):
        assert_no_leakage(["phase_group", bad])


def test_time_varying_only_in_its_sensitivity_set():
    with pytest.raises(LeakageError):
        assert_no_leakage(["n_locations"])
    cat, num = feature_set("with_time_varying")
    assert "n_locations" in num
    pcat, pnum = feature_set("primary")
    assert not {"n_locations", "n_countries", "has_us_site", "start_year"} & set(pcat + pnum)


def test_every_feature_set_is_leakage_free():
    for name in ("primary", "with_time_varying", "with_start_year"):
        cat, num = feature_set(name)
        assert "target_terminated" not in cat + num


def test_derived_flags():
    d = add_derived_features(pd.DataFrame({"intervention_types": ["DRUG|OTHER", None], "std_ages": ["ADULT", None]}))
    assert d.loc[0, ["itype_DRUG", "itype_OTHER", "itype_RADIATION", "age_ADULT"]].tolist() == [1, 1, 0, 1]
    assert d.loc[1, ["itype_DRUG", "age_ADULT"]].tolist() == [0, 0]


# ------------------------------------------------------------------ guard
@pytest.mark.parametrize("state,status", [("rejected", "research_verified"), ("unstaged", "research_verified"),
                                          ("development", "development_not_for_research")])
def test_downstream_commands_refuse_invalid_populations(tmp_path, state, status):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=200, output_state=state, status=status)
    with pytest.raises(ResearchPopulationError):
        analysis.run_eda(db, tmp_path / "eda", n_boot=10)
    with pytest.raises(ResearchPopulationError):
        modeling.run_modeling(db, tmp_path / "m", cfg=FAST)


# ------------------------------------------------------------------ EDA
def test_eda_statistics(tmp_path):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=400)
    meta = analysis.run_eda(db, tmp_path / "eda", n_boot=50)
    assoc = pd.read_csv(tmp_path / "eda" / "eda_associations.csv")
    tested = assoc[assoc["p_value"].notna()]
    assert len(tested) == meta["n_tests"] > 10
    assert (tested["q_value"] >= tested["p_value"] - 1e-12).all()          # BH q >= p
    assert set(assoc["availability"]) <= {"baseline_candidate", "time_varying_suspect"}
    levels = pd.read_csv(tmp_path / "eda" / "eda_levels.csv")
    ok = levels.dropna(subset=["termination_rate"])
    assert ((ok["ci_low"] <= ok["termination_rate"] + 1e-9) & (ok["termination_rate"] <= ok["ci_high"] + 1e-9)).all()
    for f in ("eda_summary.md", "eda_figures.html", "eda.duckdb", "parquet/eda_associations.parquet"):
        assert (tmp_path / "eda" / f).exists()


def test_cramers_v_bounds_and_rank_biserial_sign():
    tab = np.array([[50, 0], [0, 50]])
    from scipy.stats import chi2_contingency
    v = analysis.cramers_v_corrected(tab, chi2_contingency(tab, correction=False)[0])
    assert 0.95 < v <= 1.0
    df = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12], "target_terminated": [0, 0, 0, 1, 1, 1]})
    r = analysis.numeric_associations(df, ["x"], n_boot=20, seed=1).iloc[0]
    assert r["effect_size"] == 1.0 and r["median_diff"] > 0


# ------------------------------------------------------------------ modelling
def test_temporal_split_is_disjoint_and_ordered():
    df = modeling.prepare_frame(synthetic_cohort(300))
    tr, te, cutoff = modeling.temporal_split(df, 0.25)
    assert not set(tr["nct_id"]) & set(te["nct_id"])
    assert tr["start_date"].max() < te["start_date"].min()
    assert len(tr) + len(te) == len(df)


def test_insufficient_events_raise():
    df = modeling.prepare_frame(synthetic_cohort(60))
    tr, te, _ = modeling.temporal_split(df, 0.25)
    with pytest.raises(modeling.InsufficientDataError):
        modeling.check_split(tr, te, modeling.ModelingConfig())


def test_point_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    m = modeling.point_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]), threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 0, 1, 2)
    assert m["roc_auc"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    assert abs(m["brier"] - np.mean((np.array([0.1, 0.4, 0.35, 0.8]) - y) ** 2)) < 1e-12


def test_full_modeling_run_and_reproduction(tmp_path):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=600)
    exp = modeling.run_modeling(db, tmp_path / "m", cfg=FAST)
    metrics = pd.read_csv(tmp_path / "m" / "metrics_test.csv").set_index("model")
    assert list(metrics.index) == modeling.MODEL_ORDER
    assert metrics.loc["majority", "recall"] == 0.0                     # majority predicts all-completed
    assert abs(metrics.loc["prevalence", "roc_auc"] - 0.5) < 1e-9       # constant score
    assert exp["n_train"] + exp["n_test"] == 600
    for f in ("bootstrap.csv", "subgroups.csv", "calibration_bins.csv", "curves.csv", "logistic_coefficients.csv",
              "permutation_importance.csv", "sensitivity.csv", "predictions_test.csv", "evaluation_summary.md"):
        assert (tmp_path / "m" / f).exists(), f
    coefs = pd.read_csv(tmp_path / "m" / "logistic_coefficients.csv")
    assert set(coefs["availability"]) == {"baseline_candidate"}
    assert "SMOKE RUN" in (tmp_path / "m" / "evaluation_summary.md").read_text(encoding="utf-8")
    rep = modeling.reproduce(db, tmp_path / "m")
    assert rep["match"].all() and len(rep) == 6 * len(modeling.MODEL_ORDER)


def test_training_is_deterministic(tmp_path):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=400)
    cfg = modeling.ModelingConfig(cv_folds=3, n_bootstrap=20, permutation_repeats=2, min_events_train=10,
                                  min_events_test=10, fast=True)
    modeling.run_modeling(db, tmp_path / "a", cfg=cfg)
    modeling.run_modeling(db, tmp_path / "b", cfg=cfg)
    a = pd.read_csv(tmp_path / "a" / "metrics_test.csv")
    b = pd.read_csv(tmp_path / "b" / "metrics_test.csv")
    pd.testing.assert_frame_equal(a, b)


def test_reproduce_detects_changed_cohort(tmp_path):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=400)
    modeling.run_modeling(db, tmp_path / "m", cfg=modeling.ModelingConfig(
        cv_folds=3, n_bootstrap=10, permutation_repeats=1, min_events_train=10, min_events_test=10, fast=True))
    write_synthetic_research_db(db, n=400, seed=99)
    with pytest.raises(ValueError, match="fingerprint"):
        modeling.reproduce(db, tmp_path / "m")


def test_missing_and_unseen_feature_values_handled(tmp_path):
    db = tmp_path / "t.duckdb"
    write_synthetic_research_db(db, n=400)
    modeling.run_modeling(db, tmp_path / "m", cfg=modeling.ModelingConfig(
        cv_folds=3, n_bootstrap=10, permutation_repeats=1, min_events_train=10, min_events_test=10, fast=True))
    import joblib
    model = joblib.load(tmp_path / "m" / "models" / "logistic.joblib")
    exp = json.loads((tmp_path / "m" / "experiment.json").read_text())
    cat, num = exp["features_categorical"], exp["features_numeric"]
    row = {c: "NEVER_SEEN_LEVEL" for c in cat} | {c: np.nan for c in num}
    p = model.predict_proba(pd.DataFrame([row]))[:, 1]
    assert 0 < p[0] < 1
