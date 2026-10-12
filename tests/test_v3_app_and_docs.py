"""App data loading, Streamlit smoke tests, and documentation publishing (SYNTHETIC data)."""
import json
import os
from pathlib import Path

import pytest
import yaml

from trialsight import analysis, modeling
from trialsight.app_data import load_app_state
from trialsight.config import DEFAULT_CONFIG_PATH
from trialsight.docs_publish import publish
from v3_synthetic import write_synthetic_research_db

FAST = modeling.ModelingConfig(cv_folds=3, n_bootstrap=20, permutation_repeats=1, min_events_train=10,
                               min_events_test=10, fast=True)
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _config(tmp_path, research=True):
    base = yaml.safe_load(open(DEFAULT_CONFIG_PATH, encoding="utf-8"))
    base["paths"] = {k: str(tmp_path / v) for k, v in {
        "raw_dir": "raw", "db_path": "processed/t.duckdb", "processed_dir": "processed", "api_snapshot_dir": "snap",
        "dev_db_path": "processed_dev/t_dev.duckdb", "dev_processed_dir": "processed_dev",
        "artifacts_dir": "artifacts", "dev_artifacts_dir": "artifacts_dev"}.items()}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def research_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    cfg = _config(tmp)
    db = tmp / "processed" / "t.duckdb"
    db.parent.mkdir(parents=True)
    write_synthetic_research_db(db, n=500)
    analysis.run_eda(db, tmp / "artifacts" / "eda", n_boot=20)
    modeling.run_modeling(db, tmp / "artifacts" / "models", cfg=FAST)
    return tmp, cfg


def test_app_state_research(research_env):
    tmp, cfg = research_env
    s = load_app_state(str(cfg))
    assert s.mode == "research" and len(s.tables["cohort"]) == 500
    assert "metrics_test" in s.models and "eda_associations" in s.eda and not s.warnings


def test_app_state_unavailable_and_development(tmp_path):
    cfg = _config(tmp_path)
    assert load_app_state(str(cfg)).mode == "unavailable"
    dev = tmp_path / "processed_dev" / "t_dev.duckdb"
    dev.parent.mkdir(parents=True)
    write_synthetic_research_db(dev, n=100, output_state="development", status="development_not_for_research")
    assert load_app_state(str(cfg)).mode == "unavailable"                          # opt-in required
    assert load_app_state(str(cfg), allow_development=True).mode == "development"


def test_app_hides_artifacts_from_another_run(research_env, tmp_path):
    tmp, cfg = research_env
    exp_path = tmp / "artifacts" / "models" / "experiment.json"
    original = exp_path.read_text()
    exp = json.loads(original)
    exp["source"]["raw_run_id"] = "SOME-OTHER-RUN"
    exp_path.write_text(json.dumps(exp))
    try:
        s = load_app_state(str(cfg))
        assert not s.models and any("not the loaded build" in w for w in s.warnings)
    finally:
        exp_path.write_text(original)


def _apptest(cfg, monkeypatch):
    pytest.importorskip("streamlit")   # app extra (pulls in PyArrow); core tests run without it
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("TRIALSIGHT_CONFIG", str(cfg))
    at = AppTest.from_file(str(APP), default_timeout=120)
    return at


def test_streamlit_app_renders_research_build(research_env, monkeypatch):
    tmp, cfg = research_env
    at = _apptest(cfg, monkeypatch).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 7
    assert any(m.label == "Classification cohort" and m.value == "500" for m in at.metric)
    assert any("smoke mode" in w.value for w in at.warning)           # fast artifacts are flagged in the UI


def test_streamlit_app_unavailable_shows_error(tmp_path, monkeypatch):
    at = _apptest(_config(tmp_path), monkeypatch).run()
    assert not at.exception and at.error and "No accepted research build" in at.error[0].value


def test_publish_refuses_smoke_and_development(research_env, tmp_path):
    tmp, _ = research_env
    with pytest.raises(ValueError, match="fast"):
        publish(tmp / "artifacts", tmp_path)


def test_publish_writes_docs_from_artifacts(research_env, tmp_path):
    tmp, _ = research_env
    exp_path = tmp / "artifacts" / "models" / "experiment.json"
    original = exp_path.read_text()
    exp = json.loads(original)
    exp["config"]["fast"] = False             # exercise the generator; the data remain synthetic test data
    exp_path.write_text(json.dumps(exp))
    try:
        paths = publish(tmp / "artifacts", tmp_path)
    finally:
        exp_path.write_text(original)
    names = sorted(p.name for p in paths)
    assert names == ["EVALUATION_REPORT.md", "MODEL_CARD.md", "RESEARCH_FINDINGS.md"]
    text = (tmp_path / "EVALUATION_REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC-TEST-DATA" in text and "Failure analysis" in text
