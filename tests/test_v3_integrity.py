"""Phase 1 adversarial tests: reconciliation cannot bypass critical failures; recount failures
are explicit; corrupt files fail gracefully; query consistency; safe research outputs."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import requests

from trialsight.config import ApiConfig, Config, PathsConfig, QueryConfig
from trialsight.pipeline import build_database, canonical_query, query_consistency
from trialsight.raw_store import (count_tolerance, fetch_to_raw, iter_raw_studies, read_manifest,
                                  reconcile_run, verify_run)
from trialsight.storage import ResearchPopulationError, read_table, require_research_population
from test_resolution_rule import CFG_RULE
from v2_helpers import client_for, mini, responses

REASON = "Reviewed the difference against the registry; acceptable registry churn."


def _ids(n, start=0):
    return [mini(f"NCT{i:08d}") for i in range(start, start + n)]


def _run(root, fake_session, fake_response, pages, total, page_size=1000, extra=(), **kw):
    s = fake_session(responses(fake_response, pages, total, extra))
    return fetch_to_raw(client_for(s, page_size), {"query.cond": "x"}, root, **kw)


# ------------------------------------------------------------- 1.1 reconciliation
def test_one_record_when_api_reports_1000_cannot_be_reconciled(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(1)], 1000)
    v = verify_run(run)
    assert v.status == "unverified" and not v.research_grade
    assert any("difference -999" in c for c in v.critical_issues)
    with pytest.raises(ValueError, match="critical integrity failures"):
        reconcile_run(run, REASON)
    note = reconcile_run(run, REASON, documentation_only=True)          # notes allowed for documentation...
    assert note["grants_research_grade"] is False
    assert not verify_run(run).research_grade                           # ...but grant nothing


def test_forged_reconciliation_cannot_bypass_critical_failure(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(1)], 1000)
    v = verify_run(run)
    forged = {"grants_research_grade": True, "verification_snapshot": v.snapshot,
              "manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()}
    (run / "reconciliation.json").write_text(json.dumps(forged), encoding="utf-8")
    v2 = verify_run(run)
    assert not v2.reconciliation_valid and not v2.research_grade
    assert any("critical integrity failures cannot be reconciled" in r for r in v2.reasons)


def test_small_gap_within_tolerance_is_reconcilable(tmp_path, fake_session, fake_response):
    assert count_tolerance(1000) == 3 and count_tolerance(35637) == 17
    run = _run(tmp_path, fake_session, fake_response, [_ids(998)], 1000)
    v = verify_run(run)
    assert not v.critical_issues and v.reconcilable_issues and not v.research_grade
    rec = reconcile_run(run, REASON)
    assert rec["grants_research_grade"] is True and verify_run(run).research_grade


def test_gap_just_beyond_tolerance_is_critical(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(996)], 1000)   # gap 4 > tolerance 3
    with pytest.raises(ValueError, match="critical"):
        reconcile_run(run, REASON)


@pytest.mark.parametrize("pages,total", [
    ([[mini("NCT00000001"), mini("NCT00000002", "a")], [mini("NCT00000002", "b")]], 3),  # conflicting copies
    ([_ids(1), _ids(1, 5)], 2),                                                           # short non-final page
])
def test_conflicts_and_pagination_failures_fail_closed(tmp_path, fake_session, fake_response, pages, total):
    run = _run(tmp_path, fake_session, fake_response, pages, total, page_size=2)
    assert verify_run(run).critical_issues
    with pytest.raises(ValueError, match="critical"):
        reconcile_run(run, REASON)


# ------------------------------------------------------------- 1.2 final recount
@pytest.mark.parametrize("failure", [requests.Timeout("read timed out"), "http500"])
def test_recount_failure_is_recorded_not_treated_as_success(tmp_path, fake_session, fake_response, failure):
    bad = fake_response(500, text="server error") if failure == "http500" else failure
    run = _run(tmp_path, fake_session, fake_response, [_ids(2)], 2, extra=[bad], recount_at_end=True)
    m = read_manifest(run)
    assert m["recount_requested"] is True and "total_count_at_end_error" in m
    v = verify_run(run)
    assert v.recount_status == "failed" and v.status == "unverified" and not v.research_grade
    assert any("recount FAILED" in r for r in v.reconcilable_issues)
    reconcile_run(run, REASON)                       # counts matched; the missing check is documented
    assert verify_run(run).research_grade


def test_recount_mismatch_beyond_tolerance_is_critical(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(2)], 2,
               extra=[fake_response(payload={"totalCount": 50, "studies": []})], recount_at_end=True)
    v = verify_run(run)
    assert v.recount_status == "mismatch" and any("changed during download" in c for c in v.critical_issues)


def test_recount_matched(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(2)], 2,
               extra=[fake_response(payload={"totalCount": 2, "studies": []})], recount_at_end=True)
    v = verify_run(run)
    assert (v.recount_status, v.status, v.research_grade) == ("matched", "verified", True)


def test_recount_requested_but_missing_result(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [_ids(2)], 2)
    m = read_manifest(run)
    m["recount_requested"] = True
    (run / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert verify_run(run).recount_status == "missing" and not verify_run(run).research_grade


# ------------------------------------------------------------- 1.4 corrupted files
def _good_run(tmp_path, fake_session, fake_response):
    return _run(tmp_path, fake_session, fake_response, [_ids(2)], 2)


def test_truncated_gzip_reported(tmp_path, fake_session, fake_response):
    run = _good_run(tmp_path, fake_session, fake_response)
    page = run / "page_0001.json.gz"
    page.write_bytes(page.read_bytes()[:10])
    v = verify_run(run)
    assert v.status == "corrupt" and any("corrupt gzip" in c for c in v.critical_issues)
    with pytest.raises(ValueError, match="corrupt gzip"):
        list(iter_raw_studies(run))


def test_invalid_json_with_matching_checksum_reported(tmp_path, fake_session, fake_response):
    run = _good_run(tmp_path, fake_session, fake_response)
    blob = b'{"studies": [ not json'
    (run / "page_0001.json.gz").write_bytes(gzip.compress(blob))
    m = read_manifest(run)
    m["pages"][0]["sha256"] = hashlib.sha256(blob).hexdigest()
    (run / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    v = verify_run(run)
    assert v.status == "corrupt" and any("invalid JSON" in c for c in v.critical_issues)


@pytest.mark.parametrize("damage", ["missing_page", "missing_manifest", "garbage_manifest", "garbage_reconciliation"])
def test_damaged_runs_return_explicit_failures(tmp_path, fake_session, fake_response, damage):
    run = _good_run(tmp_path, fake_session, fake_response)
    if damage == "missing_page":
        (run / "page_0001.json.gz").unlink()
    elif damage == "missing_manifest":
        (run / "manifest.json").unlink()
    elif damage == "garbage_manifest":
        (run / "manifest.json").write_text("{not json", encoding="utf-8")
    else:
        (run / "reconciliation.json").write_text("{not json", encoding="utf-8")
    v = verify_run(run)                                # must not raise
    if damage == "garbage_reconciliation":
        assert v.status == "verified" and not v.reconciliation_valid
    else:
        assert v.status == "corrupt" and not v.research_grade and v.critical_issues


# ------------------------------------------------------------- 1.3 query consistency
def test_canonical_query_ignores_order_whitespace_and_paging():
    a = {"format": "json", "query.cond": "cancer OR neoplasm",
         "filter.advanced": "AREA[StudyType]INTERVENTIONAL AND ( AREA[Phase]PHASE2 OR AREA[Phase]PHASE3 )",
         "pageSize": 1000, "countTotal": "true"}
    b = {"filter.advanced": "AREA[StudyType]INTERVENTIONAL AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)",
         "query.cond": "cancer  OR\n neoplasm"}
    assert canonical_query(a) == canonical_query(b) and query_consistency(a, b) == []


def test_changed_cohort_filter_is_detected():
    a = {"filter.advanced": "AREA[StartDate]RANGE[2008-01-01,2025-12-31]"}
    b = {"filter.advanced": "AREA[StartDate]RANGE[2015-01-01,2025-12-31]"}
    assert len(query_consistency(a, b)) == 1 and "2015" in query_consistency(a, b)[0]


def _cfg(tmp_path, cond="x"):
    return Config(api=ApiConfig(), query=QueryConfig(cond=cond), cohort=CFG_RULE,
                  paths=PathsConfig(raw_dir=tmp_path / "raw", db_path=tmp_path / "processed" / "t.duckdb",
                                    processed_dir=tmp_path / "processed", api_snapshot_dir=tmp_path / "snap"))


def _verified_fixture_run(tmp_path, fake_session, fake_response, studies, cond="x"):
    clean = [st for i, st in enumerate(studies) if i != 6]
    s = fake_session(responses(fake_response, [clean], len(clean)))
    return fetch_to_raw(client_for(s, page_size=100), QueryConfig(cond=cond).to_params(), tmp_path / "raw")


def test_research_build_refuses_query_mismatch(tmp_path, fake_session, fake_response, synthetic_studies):
    run = _verified_fixture_run(tmp_path, fake_session, fake_response, synthetic_studies, cond="x")
    with pytest.raises(ValueError, match="different cohort-defining query"):
        build_database(run, _cfg(tmp_path, cond="cancer"), require_research_grade=True)
    assert not (tmp_path / "processed").exists()


# ------------------------------------------------------------- 1.5 safe outputs
def test_rejected_build_never_replaces_accepted_output(tmp_path, fake_session, fake_response, synthetic_studies):
    from dataclasses import replace
    cfg = _cfg(tmp_path)
    run = _verified_fixture_run(tmp_path, fake_session, fake_response, synthetic_studies)
    good = build_database(run, cfg, require_research_grade=True)
    assert good["output_state"] == "accepted" and Path(good["output_dir"]) == cfg.paths.processed_dir
    accepted_id = require_research_population(cfg.paths.db_path)["built_at_utc"]
    bad_cfg = replace(cfg, cohort=replace(cfg.cohort, target_statuses={"COMPLETED": 0, "SUSPENDED": 1}))
    bad = build_database(run, bad_cfg, require_research_grade=True)
    assert bad["output_state"] == "rejected"
    assert require_research_population(cfg.paths.db_path)["built_at_utc"] == accepted_id   # untouched
    assert not (tmp_path / ".processed_staging").exists()


def test_second_accepted_build_keeps_one_backup(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    run = _verified_fixture_run(tmp_path, fake_session, fake_response, synthetic_studies)
    build_database(run, cfg, require_research_grade=True)
    build_database(run, cfg, require_research_grade=True)
    assert (tmp_path / "processed_previous" / "t.duckdb").exists()
    assert require_research_population(cfg.paths.db_path)["output_state"] == "accepted"


def test_unstaged_library_build_is_not_accepted(tmp_path, fake_session, fake_response, synthetic_studies):
    cfg = _cfg(tmp_path)
    run = _verified_fixture_run(tmp_path, fake_session, fake_response, synthetic_studies)
    info = build_database(run, cfg)                      # library default: not staged
    assert info["population_status"] == "research_verified" and info["output_state"] == "unstaged"
    assert read_table(cfg.paths.db_path, "build_metadata").loc[0, "output_state"] == "unstaged"
    with pytest.raises(ResearchPopulationError, match="unstaged"):
        require_research_population(cfg.paths.db_path)
