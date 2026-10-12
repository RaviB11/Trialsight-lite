"""Issue 7: an incomplete sample can never become the research population or back validated results."""
import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from trialsight.cli import main
from trialsight.config import ApiConfig, Config, PathsConfig, QueryConfig
from trialsight.pipeline import build_database, development_config
from trialsight.raw_store import fetch_to_raw
from trialsight.storage import ResearchPopulationError, list_tables, read_table, require_research_population
from test_resolution_rule import CFG_RULE
from v2_helpers import client_for, responses


def _cfg(tmp_path):
    return Config(api=ApiConfig(), query=QueryConfig(cond="x"), cohort=CFG_RULE,
                  paths=PathsConfig(raw_dir=tmp_path / "raw", db_path=tmp_path / "processed" / "t.duckdb",
                                    processed_dir=tmp_path / "processed", api_snapshot_dir=tmp_path / "snap",
                                    dev_db_path=tmp_path / "processed_dev" / "t_dev.duckdb",
                                    dev_processed_dir=tmp_path / "processed_dev"))


def _sample_run(tmp_path, fake_session, fake_response, studies):
    """Mimics the one-page sample: page 1 saved, pagination truncated (--max-pages 1)."""
    pages = [studies[:6], studies[6:]]
    s = fake_session(responses(fake_response, pages, len(studies)))
    return fetch_to_raw(client_for(s, page_size=6), {"q": "x"}, tmp_path / "raw", max_pages=1)


def _verified_run(tmp_path, fake_session, fake_response, studies, params=None):
    # v3: fetch with the same cohort-defining query as the config (Phase 1.3 query-consistency check)
    clean = [st for i, st in enumerate(studies) if i != 6]   # drop the fixture's conflicting duplicate
    s = fake_session(responses(fake_response, [clean], len(clean)))
    return fetch_to_raw(client_for(s, page_size=100), params or QueryConfig(cond="x").to_params(), tmp_path / "raw")


def test_sample_cannot_be_built_as_research(tmp_path, fake_session, fake_response, synthetic_studies):
    run = _sample_run(tmp_path, fake_session, fake_response, synthetic_studies)
    with pytest.raises(ValueError, match="not research-grade"):
        build_database(run, _cfg(tmp_path), allow_incomplete=True, require_research_grade=True)


def test_sample_development_build_is_labelled_and_isolated(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    run = _sample_run(tmp_path, fake_session, fake_response, synthetic_studies)
    info = build_database(run, development_config(cfg), allow_incomplete=True, development=True)
    assert info["population_status"] == "development_not_for_research"
    assert info["verification"]["status"] == "incomplete"
    assert not (tmp_path / "processed").exists()               # research outputs untouched
    dev_db = cfg.paths.dev_db_path
    with pytest.raises(ResearchPopulationError, match="development build"):
        require_research_population(dev_db)
    report = (tmp_path / "processed_dev" / "cohort_validation_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Cohort validation report\n> **DEVELOPMENT BUILD - NOT A RESEARCH POPULATION.**")
    assert read_table(dev_db, "build_metadata").loc[0, "population_status"] == "development_not_for_research"


def test_development_flag_wins_even_for_verified_run(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    run = _verified_run(tmp_path, fake_session, fake_response, synthetic_studies)
    build_database(run, development_config(cfg), development=True)
    with pytest.raises(ResearchPopulationError):
        require_research_population(cfg.paths.dev_db_path)


def test_verified_run_with_validated_cohort_passes_guard(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    run = _verified_run(tmp_path, fake_session, fake_response, synthetic_studies)
    info = build_database(run, cfg, require_research_grade=True)
    assert info["population_status"] == "research_verified" and info["cohort_validated"] is True
    row = require_research_population(cfg.paths.db_path)
    assert row["raw_run_id"] == info["raw_run_id"]
    checks = read_table(cfg.paths.db_path, "cohort_validation").set_index("check_id")
    assert (checks.loc[checks.severity == "critical", "status"] == "pass").all()
    for t in ("threshold_sensitivity", "oncology_capture_summary", "phase_audit", "date_audit", "build_metadata"):
        assert t in list_tables(cfg.paths.db_path)
        assert (tmp_path / "processed" / "parquet" / f"{t}.parquet").exists()   # DuckDB-native export still used


def test_unvalidated_research_build_is_blocked(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    cfg = replace(cfg, cohort=replace(cfg.cohort, target_statuses={"COMPLETED": 0, "SUSPENDED": 1}))
    run = _verified_run(tmp_path, fake_session, fake_response, synthetic_studies)
    info = build_database(run, cfg, require_research_grade=True)
    assert info["population_status"] == "research_verified" and info["cohort_validated"] is False
    # v3 (Phase 1.5): a failed research build is quarantined, never left at the accepted path
    assert info["output_state"] == "rejected" and not cfg.paths.db_path.exists()
    rejected = Path(info["output_dir"])
    assert "processed_rejected" in str(rejected) and (rejected / "INVALID_DO_NOT_USE.txt").exists()
    with pytest.raises(ResearchPopulationError, match="validation"):
        require_research_population(rejected / "t.duckdb")


def test_guard_fails_closed_for_legacy_or_missing_db(tmp_path):
    with pytest.raises(ResearchPopulationError, match="No database"):
        require_research_population(tmp_path / "nope.duckdb")
    legacy = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(legacy)) as con:
        con.execute("CREATE TABLE cohort AS SELECT 1 AS x")
    with pytest.raises(ResearchPopulationError, match="no build_metadata"):
        require_research_population(legacy)


def test_development_config_refuses_shared_paths(tmp_path):
    cfg = _cfg(tmp_path)
    clash = replace(cfg, paths=replace(cfg.paths, dev_db_path=cfg.paths.db_path))
    with pytest.raises(ValueError, match="must differ"):
        development_config(clash)


# ---------------------------------------------------------------- CLI level
def _write_cli_config(tmp_path):
    base = yaml.safe_load(open(__import__("trialsight.config", fromlist=["x"]).DEFAULT_CONFIG_PATH, encoding="utf-8"))
    base["cohort"]["resolution_rule"].update({"missing_year_policy": "skip", "min_studies_per_year": 1})
    base["paths"] = {k: str(tmp_path / v) for k, v in {
        "raw_dir": "raw", "db_path": "processed/t.duckdb", "processed_dir": "processed",
        "api_snapshot_dir": "snap", "dev_db_path": "processed_dev/t_dev.duckdb",
        "dev_processed_dir": "processed_dev"}.items()}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


def test_cli_sample_flow(tmp_path, fake_session, fake_response, synthetic_studies, capsys):
    cfg_path = _write_cli_config(tmp_path)
    run = _sample_run(tmp_path, fake_session, fake_response, synthetic_studies)
    assert main(["--config", str(cfg_path), "verify", "--run-dir", str(run)]) == 3
    assert main(["--config", str(cfg_path), "build", "--run-dir", str(run)]) == 1          # research mode refuses
    assert not (tmp_path / "processed").exists()
    assert main(["--config", str(cfg_path), "build", "--allow-incomplete", "--run-dir", str(run)]) == 0  # old flag
    out = capsys.readouterr()
    assert "DEVELOPMENT BUILD - NOT A RESEARCH POPULATION" in out.err
    assert (tmp_path / "processed_dev" / "t_dev.duckdb").exists() and not (tmp_path / "processed").exists()


def test_cli_verified_flow(tmp_path, fake_session, fake_response, synthetic_studies, capsys):
    cfg_path = _write_cli_config(tmp_path)
    from trialsight.config import load_config
    run = _verified_run(tmp_path, fake_session, fake_response, synthetic_studies,
                        params=load_config(cfg_path).query.to_params())
    assert main(["--config", str(cfg_path), "verify"]) == 0
    assert main(["--config", str(cfg_path), "build"]) == 0
    info = json.loads((tmp_path / "processed" / "build_info.json").read_text(encoding="utf-8"))
    assert info["population_status"] == "research_verified" and info["cohort_validated"] is True
