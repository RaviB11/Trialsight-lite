"""Immutable raw zone: API pages stored verbatim with a provenance manifest.

Each fetch creates data/raw/<run_id>/ containing gzip'd page payloads and a
manifest.json recording the endpoint, exact query parameters, retrieval times
(UTC), API version/data timestamp, reported totalCount, and a SHA-256 per page.
Downstream steps read only from a *complete* run, and verify checksums.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import logging
import math
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .api_client import CTGovAPIError, CTGovClient

log = logging.getLogger(__name__)
MANIFEST = "manifest.json"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fetch_to_raw(
    client: CTGovClient,
    params: dict[str, Any],
    raw_root: Path,
    max_pages: int | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
    recount_at_end: bool = False,
) -> Path:
    """Download all pages verbatim.

    ``manifest["complete"]`` means only that pagination finished without
    truncation or interruption. It does NOT mean the download is verified;
    research-grade status is decided by :func:`verify_run`, which recomputes
    counts, duplicates and page structure from the files on disk.
    """
    started = now()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(raw_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "source": "ClinicalTrials.gov API v2",
        "base_url": client.cfg.base_url,
        "endpoint": "studies",
        "query_params": params,
        "retrieved_at_utc_start": started.isoformat(),
        "api_version": None,
        "total_count_reported": None,
        "n_studies": 0,
        "page_size_requested": client.cfg.page_size,
        "pages": [],
        "fetch_anomalies": [],
        "complete": False,
    }
    page_size = client.cfg.page_size
    try:
        try:
            manifest["api_version"] = client.version()
        except CTGovAPIError as exc:  # non-fatal; recorded for transparency
            manifest["api_version_error"] = str(exc)

        truncated = False
        for page_no, req_params, payload in client.iter_study_pages(params):
            if page_no == 1:
                manifest["total_count_reported"] = payload.get("totalCount")
                if payload.get("totalCount") is None:
                    manifest["fetch_anomalies"].append("page 1: totalCount missing from response")
            studies = payload.get("studies", []) or []
            for msg in _page_anomalies(page_no, studies, bool(payload.get("nextPageToken")), page_size):
                manifest["fetch_anomalies"].append(msg)
                log.warning(msg)
            reported_total = manifest["total_count_reported"]
            if isinstance(reported_total, int):
                limit = math.ceil(reported_total / page_size) * 2 + 5
                if page_no > limit:
                    raise CTGovAPIError(
                        f"Pagination exceeded {limit} pages for totalCount={reported_total}; stopping to avoid a runaway loop."
                    )
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            name = f"page_{page_no:04d}.json.gz"
            (run_dir / name).write_bytes(gzip.compress(blob, mtime=0))
            manifest["pages"].append(
                {
                    "file": name,
                    "n_studies": len(studies),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "request_params": {k: v for k, v in req_params.items() if k != "pageToken"},
                    "used_page_token": "pageToken" in req_params,
                }
            )
            manifest["n_studies"] += len(studies)
            log.info("Page %d: %d studies (running total %d)", page_no, len(studies), manifest["n_studies"])
            if max_pages and page_no >= max_pages:
                truncated = True
                manifest["truncated_at_max_pages"] = max_pages
                break

        manifest["complete"] = not truncated
        if recount_at_end and manifest["complete"]:
            manifest["recount_requested"] = True
            try:
                manifest["total_count_at_end"] = client.count(params)
            except Exception as exc:  # recorded explicitly; never treated as a successful check
                manifest["total_count_at_end_error"] = f"{type(exc).__name__}: {exc}"
                log.warning("End-of-download recount failed: %s", manifest["total_count_at_end_error"])
        reported = manifest["total_count_reported"]
        if manifest["complete"] and reported is not None and reported != manifest["n_studies"]:
            manifest["count_mismatch_note"] = (
                f"API reported totalCount={reported} but {manifest['n_studies']} studies were "
                "retrieved; the registry may have changed during pagination."
            )
            log.warning(manifest["count_mismatch_note"])
    finally:
        manifest["retrieved_at_utc_end"] = now().isoformat()
        (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"No manifest in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_raw_studies(run_dir: Path, allow_incomplete: bool = False) -> Iterator[dict[str, Any]]:
    manifest = read_manifest(run_dir)
    if not manifest.get("complete") and not allow_incomplete:
        raise ValueError(
            f"Run {run_dir} is incomplete (truncated or interrupted). "
            "Re-fetch, or pass allow_incomplete=True for development only."
        )
    for page in manifest["pages"]:
        payload, err = _read_page(Path(run_dir), page)
        if err:
            raise ValueError(f"{err.replace('checksum mismatch:', 'Checksum mismatch for')} in {run_dir}")
        yield from payload.get("studies", []) or []


def latest_complete_run(raw_root: Path) -> Path:
    runs = []
    for d in sorted(Path(raw_root).glob("*")):
        if d.is_dir() and (d / MANIFEST).exists() and read_manifest(d).get("complete"):
            runs.append(d)
    if not runs:
        raise FileNotFoundError(f"No complete raw runs found in {raw_root}. Run `trialsight fetch` first.")
    return runs[-1]


# --------------------------------------------------------------------------
# Verification and explicit reconciliation (research-grade gate)
# --------------------------------------------------------------------------
RECONCILIATION = "reconciliation.json"
MIN_REASON_CHARS = 20


def _nct_id(study: Any) -> str | None:
    try:
        value = study["protocolSection"]["identificationModule"]["nctId"]
    except (KeyError, TypeError):
        return None
    return value or None


def _study_hash(study: Any) -> str:
    return hashlib.sha256(json.dumps(study, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _page_anomalies(page_no: int, studies: list, has_next: bool, page_size: int) -> list[str]:
    out = []
    n = len(studies)
    if has_next and n == 0:
        out.append(f"page {page_no}: empty page but nextPageToken present")
    elif has_next and n < page_size:
        out.append(f"page {page_no}: non-final page has {n} studies (< pageSize {page_size})")
    if n > page_size:
        out.append(f"page {page_no}: {n} studies exceeds pageSize {page_size}")
    missing = sum(1 for s in studies if not _nct_id(s))
    if missing:
        out.append(f"page {page_no}: {missing} record(s) without nctId")
    return out


def count_tolerance(reported: int | None) -> int:
    """Largest unique-count difference treated as registry churn during pagination
    (reconcilable). Larger gaps are missing/extra records and fail closed.
    Judgement value: max(3, 0.05% of totalCount) -> 17 for 35,637."""
    if not isinstance(reported, int) or reported < 0:
        return 0
    return max(3, int(reported * 0.0005))


@dataclass
class RunVerification:
    run_id: str
    status: str = "unverified"  # verified | unverified | incomplete | corrupt
    research_grade: bool = False
    reasons: list[str] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)      # never reconcilable
    reconcilable_issues: list[str] = field(default_factory=list)  # acceptable only with a valid reconciliation
    pagination_complete: bool = False
    checksums_ok: bool = True
    n_pages: int = 0
    reported_count: int | None = None
    count_at_end: int | None = None
    recount_status: str = "not_requested"  # not_requested | matched | mismatch | failed | missing
    count_tolerance: int = 0
    retrieved_records: int = 0
    unique_nct_ids: int = 0
    records_missing_nct_id: int = 0
    exact_duplicate_records: int = 0      # extra byte-identical copies of a record
    conflicting_duplicate_ids: int = 0    # same nctId with different content
    duplicate_id_examples: list[str] = field(default_factory=list)
    page_anomalies: list[str] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    reconciliation: dict | None = None
    reconciliation_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_page(run_dir: Path, page: dict) -> tuple[dict | None, str | None]:
    """Return (payload, error). Never raises for corrupt content."""
    name = page.get("file", "?")
    path = Path(run_dir) / name
    if not path.exists():
        return None, f"missing page file {name}"
    try:
        blob = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, zlib.error) as exc:
        return None, f"corrupt gzip in {name}: {type(exc).__name__}"
    if hashlib.sha256(blob).hexdigest() != page.get("sha256"):
        return None, f"checksum mismatch: {name}"
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"invalid JSON in {name}: {type(exc).__name__}"
    if not isinstance(payload, dict) or not isinstance(payload.get("studies", []), (list, type(None))):
        return None, f"unexpected payload structure in {name}"
    return payload, None


def verify_run(run_dir: Path) -> RunVerification:
    """Recompute integrity from the files on disk. Never raises for corrupt runs.

    research_grade is True only if there are NO critical issues and either no
    reconcilable issues (status ``verified``) or a valid reconciliation bound to
    this exact manifest and verification snapshot.
    """
    run_dir = Path(run_dir)
    v = RunVerification(run_id=run_dir.name)
    try:
        manifest = read_manifest(run_dir)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("pages", []), list):
            raise ValueError("manifest structure invalid")
    except (OSError, ValueError) as exc:  # missing, unreadable, invalid JSON (JSONDecodeError is a ValueError)
        v.status, v.checksums_ok = "corrupt", False
        v.critical_issues.append(f"manifest unreadable: {type(exc).__name__}: {exc}")
        v.reasons = list(v.critical_issues)
        v.snapshot = {"status": v.status, "critical_issues": v.critical_issues}
        return v

    v.run_id = manifest.get("run_id", run_dir.name)
    v.pagination_complete = bool(manifest.get("complete"))
    v.reported_count = manifest.get("total_count_reported")
    v.count_at_end = manifest.get("total_count_at_end")
    pages = manifest.get("pages", [])
    v.n_pages = len(pages)
    page_size = manifest.get("page_size_requested")

    copies: dict[str, int] = {}
    hashes: dict[str, set[str]] = {}
    page1_total: Any = None
    for i, page in enumerate(pages):
        payload, err = _read_page(run_dir, page if isinstance(page, dict) else {})
        if err:
            v.checksums_ok = False
            v.critical_issues.append(err)
            continue
        if i == 0:
            page1_total = payload.get("totalCount")
        studies = payload.get("studies", []) or []
        size = page_size or (page.get("request_params") or {}).get("pageSize")
        has_next = bool(payload.get("nextPageToken"))
        if isinstance(size, int):
            v.page_anomalies.extend(_page_anomalies(i + 1, studies, has_next, size))
        is_last = i == len(pages) - 1
        if is_last and has_next and v.pagination_complete:
            v.page_anomalies.append(f"page {i + 1}: final page still carries a nextPageToken")
        if not is_last and not has_next:
            v.page_anomalies.append(f"page {i + 1}: non-final page has no nextPageToken")
        for s in studies:
            v.retrieved_records += 1
            nid = _nct_id(s)
            if not nid:
                v.records_missing_nct_id += 1
                continue
            copies[nid] = copies.get(nid, 0) + 1
            hashes.setdefault(nid, set()).add(_study_hash(s))
    for msg in manifest.get("fetch_anomalies", []) or []:
        if msg not in v.page_anomalies:
            v.page_anomalies.append(msg)

    dup_ids = sorted(k for k, c in copies.items() if c > 1)
    v.unique_nct_ids = len(copies)
    v.conflicting_duplicate_ids = sum(1 for k in dup_ids if len(hashes[k]) > 1)
    v.exact_duplicate_records = sum(copies[k] - len(hashes[k]) for k in dup_ids)
    v.duplicate_id_examples = dup_ids[:10]

    if not v.checksums_ok:
        v.status = "corrupt"
    elif not v.pagination_complete:
        v.status = "incomplete"
        detail = manifest.get("truncated_at_max_pages")
        v.critical_issues.append(
            f"pagination incomplete (truncated at --max-pages {detail})" if detail
            else "pagination incomplete (download interrupted or failed)"
        )
    else:
        crit, rec = v.critical_issues, v.reconcilable_issues
        if pages and page1_total != v.reported_count:
            crit.append(f"manifest totalCount ({v.reported_count}) disagrees with the checksummed page-1 payload ({page1_total})")
            v.reported_count = page1_total
        rc = v.reported_count
        v.count_tolerance = tol = count_tolerance(rc)
        if not isinstance(rc, int):
            crit.append("API totalCount was not recorded")
        else:
            for label, n in (("unique nctIds", v.unique_nct_ids), ("records", v.retrieved_records)):
                gap = n - rc
                if gap == 0:
                    continue
                msg = f"retrieved {n} {label} but API reported totalCount={rc} (difference {gap:+d}, tolerance {tol})"
                (crit if abs(gap) > tol else rec).append(msg)
        # end-of-download recount: requested checks must have succeeded to count as verification
        if manifest.get("recount_requested") or v.count_at_end is not None or manifest.get("total_count_at_end_error"):
            if manifest.get("total_count_at_end_error"):
                v.recount_status = "failed"
                rec.append(f"end-of-download recount FAILED ({manifest['total_count_at_end_error']}); "
                           "the final count check did not run")
            elif v.count_at_end is None:
                v.recount_status = "missing"
                rec.append("end-of-download recount was requested but no result was recorded")
            elif v.count_at_end == rc:
                v.recount_status = "matched"
            else:
                v.recount_status = "mismatch"
                gap = v.count_at_end - (rc or 0)
                msg = f"totalCount changed during download ({rc} at start, {v.count_at_end} at end)"
                (crit if abs(gap) > tol else rec).append(msg)
        if v.conflicting_duplicate_ids:
            crit.append(f"{v.conflicting_duplicate_ids} nctIds appear with conflicting content")
        if v.exact_duplicate_records:
            rec.append(f"{v.exact_duplicate_records} extra byte-identical copies (collapsed during build)")
        if v.records_missing_nct_id:
            crit.append(f"{v.records_missing_nct_id} records without nctId")
        if v.page_anomalies:
            crit.append(f"{len(v.page_anomalies)} pagination anomalies: {v.page_anomalies[:3]}")
        v.status = "verified" if not (crit or rec) else "unverified"

    v.reasons = v.critical_issues + v.reconcilable_issues
    v.snapshot = {
        "status": v.status,
        "reported_count": v.reported_count,
        "count_at_end": v.count_at_end,
        "recount_status": v.recount_status,
        "retrieved_records": v.retrieved_records,
        "unique_nct_ids": v.unique_nct_ids,
        "conflicting_duplicate_ids": v.conflicting_duplicate_ids,
        "exact_duplicate_records": v.exact_duplicate_records,
        "records_missing_nct_id": v.records_missing_nct_id,
        "n_page_anomalies": len(v.page_anomalies),
        "critical_issues": list(v.critical_issues),
        "reconcilable_issues": list(v.reconcilable_issues),
    }

    rec_path = run_dir / RECONCILIATION
    if rec_path.exists():
        try:
            recon = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            recon = {"_unreadable": f"{type(exc).__name__}"}
        v.reconciliation = recon
        v.reconciliation_valid = (
            isinstance(recon, dict)
            and v.status == "unverified"
            and not v.critical_issues                       # notes can never bypass critical failures
            and recon.get("grants_research_grade") is True
            and recon.get("manifest_sha256") == _file_sha256(run_dir / MANIFEST)
            and recon.get("verification_snapshot") == v.snapshot
        )
        if not v.reconciliation_valid:
            why = ("critical integrity failures cannot be reconciled" if v.critical_issues
                   else "stale, tampered, documentation-only, or not applicable")
            v.reasons.append(f"reconciliation.json does not grant research grade ({why})")

    v.research_grade = not v.critical_issues and (
        v.status == "verified" or (v.status == "unverified" and v.reconciliation_valid)
    )
    return v


def reconcile_run(
    run_dir: Path, reason: str, now: Callable[[], dt.datetime] = _utc_now, documentation_only: bool = False
) -> dict[str, Any]:
    """Accept a complete run whose ONLY problems are reconcilable, with a written reason.

    Critical failures (corruption, incomplete pagination, conflicting records,
    missing records beyond tolerance, serious pagination anomalies) are refused:
    re-fetch instead. ``documentation_only=True`` records a note for any
    non-verified run but never grants research grade.
    """
    reason = (reason or "").strip()
    if len(reason) < MIN_REASON_CHARS:
        raise ValueError(f"A reconciliation reason of at least {MIN_REASON_CHARS} characters is required.")
    v = verify_run(run_dir)
    if v.status == "verified":
        raise ValueError("Run is already verified; no reconciliation is needed.")
    if v.critical_issues and not documentation_only:
        raise ValueError(
            f"Run is {v.status} with critical integrity failures that cannot be reconciled: "
            f"{v.critical_issues}. Re-fetch the data. (A documentation-only note can be recorded with "
            "--note-only; it never grants research grade.)"
        )
    grants = not v.critical_issues and not documentation_only
    consequences = []
    if v.reconcilable_issues and grants:
        consequences.append(
            f"The research population is this download as retrieved; unique records differ from the "
            f"API-reported totalCount ({v.reported_count}) by {abs(v.unique_nct_ids - (v.reported_count or 0))} "
            f"(tolerance {v.count_tolerance})."
        )
    if v.exact_duplicate_records:
        consequences.append(f"{v.exact_duplicate_records} byte-identical duplicate copies are collapsed during build.")
    if v.recount_status in ("failed", "missing"):
        consequences.append("The end-of-download recount did not complete; that check is recorded as not performed.")
    rec = {
        "run_id": v.run_id,
        "reconciled_at_utc": now().isoformat(),
        "reason": reason,
        "grants_research_grade": grants,
        "documentation_only": not grants,
        "problems_acknowledged": v.critical_issues + v.reconcilable_issues,
        "consequences": consequences,
        "manifest_sha256": _file_sha256(Path(run_dir) / MANIFEST),
        "verification_snapshot": v.snapshot,
    }
    (Path(run_dir) / RECONCILIATION).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def latest_run(raw_root: Path) -> Path:
    """Most recent run with a manifest, complete or not (development use only)."""
    runs = [d for d in sorted(Path(raw_root).glob("*")) if d.is_dir() and (d / MANIFEST).exists()]
    if not runs:
        raise FileNotFoundError(f"No raw runs found in {raw_root}. Run `trialsight fetch` first.")
    return runs[-1]
