from pathlib import Path

from trialsight.api_client import CTGovClient
from trialsight.config import ApiConfig, Config, PathsConfig, QueryConfig
from trialsight.feature_registry import classify
from trialsight.pipeline import build_database
from trialsight.raw_store import fetch_to_raw
from trialsight.storage import list_tables, read_table
from test_quality_and_cohort import CFG as COHORT_CFG
from test_resolution_rule import CFG_RULE


def test_end_to_end_build(tmp_path, fake_session, fake_response, synthetic_studies):
    api = ApiConfig(base_url="https://example.test/api/v2", page_size=100, requests_per_minute=6000)
    session = fake_session([
        fake_response(payload={"apiVersion": "test"}),
        fake_response(payload={"totalCount": len(synthetic_studies), "studies": synthetic_studies}),
    ])
    client = CTGovClient(api, session=session, sleep=lambda s: None, clock=lambda: 0.0)
    run = fetch_to_raw(client, {"query.cond": "x"}, tmp_path / "raw")

    cfg = Config(api=api, query=QueryConfig(), cohort=CFG_RULE,
                 paths=PathsConfig(raw_dir=tmp_path / "raw", db_path=tmp_path / "p" / "t.duckdb",
                                   processed_dir=tmp_path / "p", api_snapshot_dir=tmp_path / "snap"))
    info = build_database(run, cfg)

    assert info["n_studies_raw"] == 11 and info["n_cohort"] == 2
    assert info["n_eligible_all_statuses"] == 4
    assert info["analysis_window"] == {"start_year": 2008, "end_year": 2014, "selected_by": "resolution_rule"}
    assert info["unclassified_columns"] == []
    tables = set(list_tables(cfg.paths.db_path))
    assert {"studies", "mesh", "cohort", "cohort_attrition", "dq_summary", "status_by_start_year",
            "resolution_by_start_year", "eligible_population"} <= tables
    cohort = read_table(cfg.paths.db_path, "cohort")
    assert cohort["target_terminated"].sum() == 1
    for f in ("dq_summary.csv", "cohort_attrition.csv", "resolution_by_start_year.csv", "build_info.json",
              "parquet/cohort.parquet", "parquet/eligible_population.parquet"):
        assert (Path(cfg.paths.processed_dir) / f).exists()


def test_every_cohort_column_is_classified(synthetic_studies):
    from trialsight.cohort import build_cohort
    from trialsight.flatten import flatten_studies
    t = flatten_studies(synthetic_studies)
    cohort = build_cohort(t["studies"], t["mesh"], COHORT_CFG).cohort
    assert [c for c in cohort.columns if classify(c) == "unclassified"] == []


def test_shipped_config_loads_with_rule():
    from trialsight.config import load_config
    cfg = load_config()
    assert (cfg.cohort.start_date_min, cfg.cohort.start_date_max) == ("2008-01-01", "2025-12-31")
    assert cfg.cohort.resolution_rule.min_resolved_share == 0.90
    assert "RANGE[2008-01-01,2025-12-31]" in cfg.query.to_params()["filter.advanced"]
