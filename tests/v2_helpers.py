"""Shared helpers for v2 regression tests. All records are SYNTHETIC test data."""
from trialsight.api_client import CTGovClient
from trialsight.config import ApiConfig


def mini(nct, title="t"):
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": title}}}


def client_for(session, page_size=2):
    cfg = ApiConfig(base_url="https://example.test/api/v2", page_size=page_size,
                    requests_per_minute=6000, max_retries=0)
    return CTGovClient(cfg, session=session, sleep=lambda s: None, clock=lambda: 0.0)


def responses(fake_response, pages, total, extra=()):
    """Version response, then one response per page; tokens added to all but the last page."""
    out = [fake_response(payload={"apiVersion": "test", "dataTimestamp": "test"})]
    for i, studies in enumerate(pages):
        payload = {"studies": studies}
        if i == 0:
            payload["totalCount"] = total
        if i < len(pages) - 1:
            payload["nextPageToken"] = f"t{i + 1}"
        out.append(fake_response(payload=payload))
    out.extend(extra)
    return out
