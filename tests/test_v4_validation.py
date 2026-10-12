"""v0.4: display hygiene, subgroup stability, ablation, consistency checks, publication hygiene. SYNTHETIC data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialsight import analysis, modeling
from trialsight.cohort import wilson_interval
from trialsight.consistency import check_consistency
from trialsight.docs_publish import publish
from trialsight.report import fmt_cell, flag_stability, md_table
from v3_synthetic import write_synthetic_research_db

FAST = modeling.ModelingConfig(cv_folds=3, n_bootstrap=30, permutation_repeats=1, min_events_train=10,
                               min_events_test=10, fast=True)


def test_fmt_cell_and_md_table_never_show_nan():
    assert [fmt_cell(x) for x in (float("nan"), None, -0.0, 0.25, 3e-16)] == ["n/a", "n/a", "0", "0.25", "3.00e-16"]
    out = md_table(pd.DataFrame({"a": [np.nan, -0.0], "b": [None, "x"]}))
    assert "nan" not in out.lower() and "-0" not in out and out.count("n/a") == 2


def test_wilson_bounds_clipped():
    lo, hi = wilson_interval(pd.Series([0, 5]), pd.Series([5, 5]))
    assert lo.iloc[0] == 0 and str(lo.iloc[0]) != "-0.0" and hi.iloc[1] == 1


def test_flag_stability():
    d = flag_stability(pd.DataFrame({"n": [2, 45, 400], "n_positive": [0, 16, 100]}))
    assert d["stability"].str.startswith("UNSTABLE").tolist() == [True, True, False]
    assert d["n_negative"].tolist() == [2, 29, 300]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v4")
    db = tmp / "processed" / "t.duckdb"
    db.parent.mkdir(parents=True)
    write_synthetic_research_db(db, n=500)
    analysis.run_eda(db, tmp / "artifacts" / "eda", n_boot=20)
    modeling.run_modeling(db, tmp / "artifacts" / "models", cfg=FAST)
    modeling.run_ablation(db, tmp / "artifacts", ["registration_lag_days"])
    exp = json.loads((tmp / "artifacts" / "models" / "experiment.json").read_text())
    (tmp / "processed" / "build_info.json").write_text(json.dumps({
        "population_status": "research_verified", "cohort_validated": True, "output_state": "accepted",
        "raw_run_id": exp["source"]["raw_run_id"], "built_at_utc": exp["source"]["built_at_utc"]}))
    return tmp, db


def test_ablation_uses_saved_protocol(env):
    tmp, _ = env
    out = tmp / "artifacts" / "ablation" / "registration_lag_days"
    meta = json.loads((out / "ablation.json").read_text())
    exp = json.loads((tmp / "artifacts" / "models" / "experiment.json").read_text())
    assert meta["test_ids_sha256"] == exp["test_ids_sha256"] and meta["config"] == exp["config"]
    comp = pd.read_csv(out / "comparison.csv")
    saved = pd.read_csv(tmp / "artifacts" / "models" / "metrics_test.csv").set_index("model")
    full = comp[comp["variant"] == "full (saved primary)"].set_index("model")
    for name in full.index:                      # "full" rows reproduce the saved primary metrics exactly
        for k in ("roc_auc", "pr_auc", "brier"):
            assert abs(full.loc[name, k] - saved.loc[name, k]) < 1e-12
    deltas = pd.read_csv(out / "paired_deltas.csv")
    assert set(deltas["metric"]) == {"roc_auc", "pr_auc", "brier", "f1"} and "majority" not in set(deltas["model"])
    timing = pd.read_csv(out / "registration_timing.csv")
    assert {"registered after primary completion date", "median registration lag (days)"} <= set(timing["check"])


def test_ablation_refuses_bad_requests(env, tmp_path):
    tmp, db = env
    with pytest.raises(ValueError, match="Not primary features"):
        modeling.run_ablation(db, tmp / "artifacts", ["n_locations"])
    other = tmp_path / "other.duckdb"
    write_synthetic_research_db(other, n=500, seed=99)
    with pytest.raises(ValueError, match="fingerprint"):
        modeling.run_ablation(other, tmp / "artifacts", ["registration_lag_days"])


def _publish_unfast(tmp, docs):
    p = tmp / "artifacts" / "models" / "experiment.json"
    original = p.read_text()
    e = json.loads(original)
    e["config"]["fast"] = False
    p.write_text(json.dumps(e))
    try:
        return publish(tmp / "artifacts", docs, tmp / "processed")
    finally:
        p.write_text(original)


def test_published_docs_are_clean_and_explain_limits(env, tmp_path):
    tmp, _ = env
    _publish_unfast(tmp, tmp_path)
    ev = (tmp_path / "EVALUATION_REPORT.md").read_text()
    card = (tmp_path / "MODEL_CARD.md").read_text()
    for text in (ev, card, (tmp_path / "RESEARCH_FINDINGS.md").read_text()):
        assert not __import__("re").search(r"(?<![A-Za-z_])nan(?![A-Za-z_])", text, flags=2)
    assert "Discrimination is modest" in ev and "False positives are high" in ev and "## Feature ablation" in ev
    assert "registration_lag_days" in ev and "not evidence of prospective predictive validity" in ev
    assert "Selection bias" in card and "UNKNOWN-status analysis unavailable" in card   # no cohort CSVs supplied


def test_consistency_pass_and_failures(env, tmp_path):
    tmp, _ = env
    _publish_unfast(tmp, tmp_path)
    exp_p = tmp / "artifacts" / "models" / "experiment.json"
    original = exp_p.read_text()
    e = json.loads(original)
    e["config"]["fast"] = False
    exp_p.write_text(json.dumps(e))
    try:
        rep = check_consistency(tmp / "processed", tmp / "artifacts", tmp_path, db_path=tmp / "processed" / "t.duckdb",
                                publish_files=list(tmp_path.glob("*.md")))
        assert (rep["status"] != "fail").all(), rep[rep["status"] == "fail"]
        assert rep.set_index("check_id").loc["C10", "status"] == "pass"
        leak = tmp_path / "LEAK.md"
        leak.write_text("output in " + "C:" + "\\".join(["", "Users", "someone", "Downloads", "x"]))  # built at runtime
        (tmp_path / "RESEARCH_FINDINGS.md").write_text((tmp_path / "RESEARCH_FINDINGS.md").read_text() + "\nmedian nan\n")
        e["source"]["raw_run_id"] = "OTHER"
        exp_p.write_text(json.dumps(e))
        rep = check_consistency(tmp / "processed", tmp / "artifacts", tmp_path, publish_files=[leak])
        failed = set(rep.loc[rep["status"] == "fail", "check_id"])
        assert {"C03", "C08", "C11"} <= failed
    finally:
        exp_p.write_text(original)


def test_app_state_loads_ablation_and_provenance(env, tmp_path):
    import yaml
    from trialsight.app_data import load_app_state
    from trialsight.config import DEFAULT_CONFIG_PATH
    tmp, _ = env
    base = yaml.safe_load(open(DEFAULT_CONFIG_PATH, encoding="utf-8"))
    base["paths"].update({"db_path": str(tmp / "processed" / "t.duckdb"), "processed_dir": str(tmp / "processed"),
                          "artifacts_dir": str(tmp / "artifacts"), "dev_db_path": str(tmp_path / "d.duckdb"),
                          "dev_processed_dir": str(tmp_path / "dev")})
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(base))
    s = load_app_state(str(cfg))
    assert "registration_lag_days" in s.ablations and s.provenance["models raw run"] == "SYNTHETIC-TEST-DATA"
