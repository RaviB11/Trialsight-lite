import json

import pytest
import requests

from trialsight.api_client import CTGovAPIError, CTGovClient, RateLimiter
from trialsight.config import ApiConfig
from trialsight.raw_store import fetch_to_raw, iter_raw_studies, latest_complete_run, read_manifest

CFG = ApiConfig(base_url="https://example.test/api/v2", page_size=2, requests_per_minute=6000, max_retries=2)


def _client(session, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return CTGovClient(CFG, session=session, sleep=sleeps.append, clock=lambda: 0.0)


def test_rate_limiter_waits():
    t = {"now": 0.0}
    slept = []
    rl = RateLimiter(60, clock=lambda: t["now"], sleep=lambda s: (slept.append(s), t.update(now=t["now"] + s)))
    rl.wait(); rl.wait()
    assert slept == [1.0]


def test_pagination_follows_tokens(fake_session, fake_response):
    s = fake_session([
        fake_response(payload={"totalCount": 3, "studies": [{"a": 1}, {"a": 2}], "nextPageToken": "t1"}),
        fake_response(payload={"studies": [{"a": 3}]}),
    ])
    pages = list(_client(s).iter_study_pages({"query.cond": "x"}))
    assert [p[0] for p in pages] == [1, 2]
    assert s.calls[0]["params"]["countTotal"] == "true" and "pageToken" not in s.calls[0]["params"]
    assert s.calls[1]["params"]["pageToken"] == "t1"
    assert all(c["params"]["pageSize"] == 2 for c in s.calls)


def test_pagination_loop_detected(fake_session, fake_response):
    s = fake_session([fake_response(payload={"studies": [], "nextPageToken": "same"})] * 3)
    with pytest.raises(CTGovAPIError, match="loop"):
        list(_client(s).iter_study_pages({}))


def test_retry_on_429_honours_retry_after(fake_session, fake_response):
    sleeps = []
    s = fake_session([fake_response(429, headers={"Retry-After": "7"}), fake_response(payload={"ok": True})])
    assert _client(s, sleeps).get_json("version") == {"ok": True}
    assert 7.0 in sleeps


def test_retry_on_network_error_then_give_up(fake_session):
    s = fake_session([requests.ConnectionError("boom")] * 3)
    with pytest.raises(CTGovAPIError, match="Giving up"):
        _client(s).get_json("version")
    assert len(s.calls) == 3


def test_client_error_not_retried(fake_session, fake_response):
    s = fake_session([fake_response(400, text="bad filter")])
    with pytest.raises(CTGovAPIError, match="400"):
        _client(s).get_json("studies")
    assert len(s.calls) == 1


def _pages_session(fake_session, fake_response, studies):
    return fake_session([
        fake_response(payload={"apiVersion": "test", "dataTimestamp": "test"}),
        fake_response(payload={"totalCount": len(studies), "studies": studies[:6], "nextPageToken": "p2"}),
        fake_response(payload={"studies": studies[6:]}),
    ])


def test_fetch_to_raw_roundtrip(tmp_path, fake_session, fake_response, synthetic_studies):
    run = fetch_to_raw(_client(_pages_session(fake_session, fake_response, synthetic_studies)),
                       {"query.cond": "x"}, tmp_path)
    m = read_manifest(run)
    assert m["complete"] and m["n_studies"] == len(synthetic_studies) and len(m["pages"]) == 2
    assert "count_mismatch_note" not in m
    assert list(iter_raw_studies(run)) == synthetic_studies
    assert latest_complete_run(tmp_path) == run


def test_truncated_run_is_incomplete_and_rejected(tmp_path, fake_session, fake_response, synthetic_studies):
    run = fetch_to_raw(_client(_pages_session(fake_session, fake_response, synthetic_studies)),
                       {}, tmp_path, max_pages=1)
    assert read_manifest(run)["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_raw_studies(run))
    with pytest.raises(FileNotFoundError):
        latest_complete_run(tmp_path)


def test_checksum_tamper_detected(tmp_path, fake_session, fake_response, synthetic_studies):
    import gzip
    run = fetch_to_raw(_client(_pages_session(fake_session, fake_response, synthetic_studies)), {}, tmp_path)
    page = run / "page_0001.json.gz"
    page.write_bytes(gzip.compress(json.dumps({"studies": []}).encode()))
    with pytest.raises(ValueError, match="Checksum"):
        list(iter_raw_studies(run))
