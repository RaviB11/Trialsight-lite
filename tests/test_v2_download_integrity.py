"""Issues 1 & 2: verified-complete gate, reconciliation, duplicates, pagination anomalies."""
import gzip
import json

import pytest

from trialsight.api_client import CTGovAPIError
from trialsight.pipeline import build_database, dedupe_exact
from trialsight.raw_store import fetch_to_raw, read_manifest, reconcile_run, verify_run
from v2_helpers import client_for, mini, responses

REASON = "Registry updated during pagination; difference reviewed and accepted."


def _run(tmp_path, fake_session, fake_response, pages, total, page_size=2, **kw):
    session = fake_session(responses(fake_response, pages, total, kw.pop("extra", ())))
    return fetch_to_raw(client_for(session, page_size), {"q": "x"}, tmp_path, **kw)


def test_matching_counts_are_verified(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001"), mini("NCT00000002")], [mini("NCT00000003")]], 3)
    v = verify_run(run)
    assert (v.status, v.research_grade, v.reasons) == ("verified", True, [])


def test_count_mismatch_is_not_verified(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001"), mini("NCT00000002")], [mini("NCT00000003")]], 5)
    v = verify_run(run)
    assert read_manifest(run)["complete"] is True          # pagination finished...
    assert v.status == "unverified" and not v.research_grade  # ...but that is not "verified complete"
    assert any("totalCount=5" in r for r in v.reasons)


def test_build_research_mode_rejects_count_mismatch(tmp_path, fake_session, fake_response, synthetic_studies):
    from test_pipeline import COHORT_CFG
    from trialsight.config import ApiConfig, Config, PathsConfig, QueryConfig
    run = _run(tmp_path / "raw", fake_session, fake_response, [synthetic_studies[:2]], 99, page_size=10)
    cfg = Config(api=ApiConfig(), query=QueryConfig(), cohort=COHORT_CFG,
                 paths=PathsConfig(raw_dir=tmp_path / "raw", db_path=tmp_path / "p" / "t.duckdb",
                                   processed_dir=tmp_path / "p", api_snapshot_dir=tmp_path / "s"))
    with pytest.raises(ValueError, match="not research-grade"):
        build_database(run, cfg, require_research_grade=True)
    assert not (tmp_path / "p" / "t.duckdb").exists()


def test_reconciliation_requires_written_reason(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 2)
    with pytest.raises(ValueError, match="at least 20 characters"):
        reconcile_run(run, "ok")


def test_reconciliation_grants_research_grade_and_is_bound_to_manifest(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 2)
    rec = reconcile_run(run, REASON)
    assert rec["reason"] == REASON and rec["consequences"]
    v = verify_run(run)
    assert v.status == "unverified" and v.reconciliation_valid and v.research_grade
    m = read_manifest(run)
    m["total_count_reported"] = 1   # edit manifest so counts *appear* to match
    (run / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    v2 = verify_run(run)
    assert not v2.reconciliation_valid and not v2.research_grade
    assert any("disagrees with the checksummed page-1 payload" in r for r in v2.reasons)


def test_manifest_total_cannot_override_page_payload(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 2)
    m = read_manifest(run)
    m["total_count_reported"] = 1
    (run / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    v = verify_run(run)
    assert v.reported_count == 2 and v.status == "unverified" and not v.research_grade


@pytest.mark.parametrize("case", ["incomplete", "verified"])
def test_reconciliation_refused(tmp_path, fake_session, fake_response, case):
    if case == "incomplete":
        run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001"), mini("NCT00000002")], [mini("NCT00000003")]], 3, max_pages=1)
        match = "incomplete"
    else:
        run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 1)
        match = "already verified"
    with pytest.raises(ValueError, match=match):
        reconcile_run(run, REASON)


def test_total_count_changing_during_download_is_flagged(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 1,
               extra=[fake_response(payload={"totalCount": 2, "studies": []})], recount_at_end=True)
    v = verify_run(run)
    assert v.count_at_end == 2 and v.status == "unverified"
    assert any("changed during download" in r for r in v.reasons)


def test_identical_duplicate_across_pages(tmp_path, fake_session, fake_response):
    b = mini("NCT00000002")
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001"), b], [b]], 3)
    v = verify_run(run)
    assert v.retrieved_records == 3 and v.unique_nct_ids == 2      # record count alone would look fine
    assert (v.exact_duplicate_records, v.conflicting_duplicate_ids) == (1, 0)
    assert v.status == "unverified" and v.duplicate_id_examples == ["NCT00000002"]


def test_conflicting_duplicate_across_pages(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response,
               [[mini("NCT00000001"), mini("NCT00000002", "v1")], [mini("NCT00000002", "v2")]], 3)
    v = verify_run(run)
    assert (v.exact_duplicate_records, v.conflicting_duplicate_ids) == (0, 1)


def test_dedupe_exact_keeps_conflicting_copies():
    a, b1, b2 = mini("NCT00000001"), mini("NCT00000002", "v1"), mini("NCT00000002", "v2")
    out, removed = dedupe_exact([a, a, b1, b2])
    assert removed == 1 and out == [a, b1, b2]


@pytest.mark.parametrize("pages,expected", [
    ([[mini("NCT00000001")], [mini("NCT00000002")]], "non-final page has 1 studies"),
    ([[], [mini("NCT00000002")]], "empty page but nextPageToken present"),
    ([[mini("NCT00000001"), {"protocolSection": {}}]], "without nctId"),
])
def test_pagination_anomalies_detected(tmp_path, fake_session, fake_response, pages, expected):
    total = sum(len(p) for p in pages)
    run = _run(tmp_path, fake_session, fake_response, pages, total)
    v = verify_run(run)
    assert any(expected in a for a in v.page_anomalies), v.page_anomalies
    assert v.status == "unverified" and not v.research_grade


def test_runaway_pagination_is_stopped(tmp_path, fake_session, fake_response):
    pages = [[mini(f"NCT{i:08d}"), mini(f"NCT{i + 50:08d}")] for i in range(20)]
    run_root = tmp_path
    with pytest.raises(CTGovAPIError, match="exceeded"):
        _run(run_root, fake_session, fake_response, pages, 2)   # totalCount=2 -> limit 7 pages
    run = next(run_root.iterdir())
    assert read_manifest(run)["complete"] is False
    assert verify_run(run).status == "incomplete"


def test_corrupt_page_reported_not_raised(tmp_path, fake_session, fake_response):
    run = _run(tmp_path, fake_session, fake_response, [[mini("NCT00000001")]], 1)
    (run / "page_0001.json.gz").write_bytes(gzip.compress(b'{"studies": []}'))
    v = verify_run(run)
    assert v.status == "corrupt" and not v.research_grade and not v.checksums_ok
