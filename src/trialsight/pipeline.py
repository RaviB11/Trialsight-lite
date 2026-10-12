"""Raw run -> normalised tables -> data-quality report -> cohort -> DuckDB + Parquet."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import __version__
from . import diagnostics as dx
from .cohort import build_cohort
from .config import Config
from .feature_registry import classify
from .flatten import flatten_studies
from .quality import run_quality_checks
from .raw_store import RunVerification, _nct_id, _study_hash, iter_raw_studies, read_manifest, verify_run
from .report import write_validation_report
from .storage import export_parquet, write_tables

log = logging.getLogger(__name__)

RESEARCH_STATUSES = ("research_verified", "research_reconciled")
DEVELOPMENT_STATUS = "development_not_for_research"


def population_status(v: RunVerification) -> str:
    if v.status == "verified":
        return "research_verified"
    if v.research_grade:
        return "research_reconciled"
    return DEVELOPMENT_STATUS


def development_config(cfg: Config) -> Config:
    """Same config, but outputs routed to separate development paths."""
    processed = Path(cfg.paths.processed_dir)
    dev_dir = cfg.paths.dev_processed_dir or processed.parent / (processed.name + "_dev")
    dev_db = cfg.paths.dev_db_path or Path(dev_dir) / "trialsight_dev.duckdb"
    if Path(dev_db).resolve() == Path(cfg.paths.db_path).resolve() or Path(dev_dir).resolve() == processed.resolve():
        raise ValueError("Development paths must differ from research paths.")
    return replace(cfg, paths=replace(cfg.paths, db_path=Path(dev_db), processed_dir=Path(dev_dir)))


# Parameters that do not change WHICH studies are returned.
NON_COHORT_PARAMS = {"format", "pageSize", "countTotal", "pageToken", "sort"}


def _canon_value(value: Any) -> str:
    text = " ".join(str(value).split())
    return re.sub(r"\s*([\[\](),])\s*", r"\1", text)


def canonical_query(params: dict[str, Any] | None) -> dict[str, str]:
    """Canonical form of cohort-defining query parameters (order/whitespace-insensitive)."""
    return {k: _canon_value(v) for k, v in sorted((params or {}).items())
            if k not in NON_COHORT_PARAMS and v not in (None, "")}


def query_consistency(raw_params: dict[str, Any] | None, config_params: dict[str, Any]) -> list[str]:
    """Differences between the query a raw run was fetched with and the current config."""
    a, b = canonical_query(raw_params), canonical_query(config_params)
    diffs = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            diffs.append(f"{key}: raw run={a.get(key)!r} vs config={b.get(key)!r}")
    return diffs


def dedupe_exact(studies: Iterable[dict]) -> tuple[list[dict], int]:
    """Collapse byte-identical copies of the same record. Conflicting copies are
    kept so blocking rule DQ01 can flag (and exclude) them."""
    seen: set[tuple[str | None, str]] = set()
    out, removed = [], 0
    for s in studies:
        key = (_nct_id(s), _study_hash(s))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(s)
    return out, removed


def build_database(
    run_dir: Path, cfg: Config, allow_incomplete: bool = False, require_research_grade: bool = False,
    development: bool = False,
) -> dict[str, Any]:
    """Build all tables. Every build is labelled with a population_status;
    only research_verified / research_reconciled builds of a validated cohort
    pass :func:`trialsight.storage.require_research_population`."""
    verification = verify_run(run_dir)
    pop_status = DEVELOPMENT_STATUS if development else population_status(verification)
    if require_research_grade and pop_status not in RESEARCH_STATUSES:
        raise ValueError(
            f"Raw run {verification.run_id} is not research-grade (status={verification.status}). "
            f"Reasons: {'; '.join(verification.reasons) or 'none recorded'}. "
            "Run `trialsight verify`; re-fetch, or record an explicit `trialsight reconcile` if the "
            "differences are understood. Use `build --development` for a non-research build."
        )
    manifest = read_manifest(run_dir)
    query_diffs = query_consistency(manifest.get("query_params"), cfg.query.to_params())
    if require_research_grade and query_diffs:
        raise ValueError(
            "The raw run was fetched with a different cohort-defining query than the current config, "
            f"so it cannot be accepted as the research population: {query_diffs}. Re-fetch with the current config."
        )
    final_dir, final_db = Path(cfg.paths.processed_dir), Path(cfg.paths.db_path)
    staging_dir = None
    if require_research_grade:
        try:
            rel_db = final_db.resolve().relative_to(final_dir.resolve())
        except ValueError as exc:
            raise ValueError("Research builds require db_path to be inside processed_dir.") from exc
        staging_dir = final_dir.parent / f".{final_dir.name}_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        cfg = replace(cfg, paths=replace(cfg.paths, processed_dir=staging_dir, db_path=staging_dir / rel_db))
    raw, n_exact_dupes = dedupe_exact(iter_raw_studies(run_dir, allow_incomplete=allow_incomplete))
    tables = flatten_studies(raw)
    studies = tables["studies"]
    if studies.empty:
        raise ValueError(f"Raw run {run_dir} contains no studies.")

    dq_summary, dq_violations = run_quality_checks(studies)
    result = build_cohort(studies, tables["mesh"], cfg.cohort)
    cohort = result.cohort
    if cohort.empty:
        log.warning("Primary cohort is empty; check cohort_attrition and resolution_by_start_year.")

    unclassified = [c for c in cohort.columns if classify(c) == "unclassified"]
    if unclassified:
        log.warning("Columns missing from feature registry: %s", unclassified)

    # ---- diagnostics (describe consequences; never alter inclusion) ----
    eligible = result.eligible_population
    dcfg = cfg.diagnostics
    onc_summary, onc_review, onc_top = dx.oncology_capture(
        result.pre_oncology_population, tables["conditions"], tables["mesh"],
        dcfg.cancer_text_terms, dcfg.review_sample_size)
    retrieved_at = pd.Timestamp(manifest["retrieved_at_utc_start"]) if manifest.get("retrieved_at_utc_start") else None
    date_checks = dx.date_audit(studies, eligible, cohort, cfg.cohort, retrieved_at)
    enum_status, enum_table = dx.enum_check(cfg.cohort, Path(cfg.paths.api_snapshot_dir) / "enums.json")
    checks = dx.validation_checks(
        verification=verification.to_dict(), eligible=eligible, cohort=cohort,
        resolution=result.resolution_by_start_year, end_year=result.analysis_end_year,
        cfg=cfg.cohort, enum_status=enum_status, date_checks=date_checks, query_diffs=query_diffs)
    validated = dx.cohort_validated(checks)
    if staging_dir is not None:
        output_state = "accepted" if (validated and pop_status in RESEARCH_STATUSES) else "rejected"
    else:
        output_state = "development" if pop_status == DEVELOPMENT_STATUS else "unstaged"

    built_at = dt.datetime.now(dt.timezone.utc).isoformat()
    config_fingerprint = hashlib.sha256(
        json.dumps({"query": asdict(cfg.query), "cohort": asdict(cfg.cohort)}, sort_keys=True, default=str).encode()
    ).hexdigest()
    build_metadata = pd.DataFrame([{
        "population_status": pop_status,
        "cohort_validated": validated,
        "output_state": output_state,
        "query_differences": json.dumps(query_diffs),
        "raw_run_id": verification.run_id,
        "verification_status": verification.status,
        "research_grade": verification.research_grade,
        "verification_json": json.dumps(verification.to_dict(), default=str),
        "exact_duplicate_records_collapsed": n_exact_dupes,
        "analysis_start_year": pd.Timestamp(cfg.cohort.start_date_min).year,
        "analysis_end_year": result.analysis_end_year,
        "n_cohort": len(cohort),
        "config_sha256": config_fingerprint,
        "package_version": __version__,
        "built_at_utc": built_at,
    }])

    all_tables = {
        **tables,
        "dq_summary": dq_summary,
        "dq_violations": dq_violations,
        "cohort": cohort,
        "cohort_attrition": result.attrition,
        "status_by_start_year": result.status_by_start_year,
        "resolution_by_start_year": result.resolution_by_start_year,
        "eligible_population": eligible,
        "threshold_sensitivity": dx.threshold_sensitivity(eligible, cfg.cohort, dcfg.thresholds),
        "terminated_share_by_year": dx.terminated_share_by_year(eligible, cfg.cohort.target_statuses),
        "selection_profile": dx.selection_profile(eligible, result.analysis_end_year, cfg.cohort, dcfg.profile_columns),
        "oncology_capture_summary": onc_summary,
        "oncology_review_sample": onc_review,
        "oncology_text_only_mesh_terms": onc_top,
        "phase_audit": dx.phase_audit(studies, cfg.cohort),
        "date_audit": date_checks,
        "start_type_by_status": dx.start_type_by_status(eligible),
        "enum_check": enum_table,
        "cohort_validation": checks,
        "build_metadata": build_metadata,
    }
    write_tables(cfg.paths.db_path, all_tables)

    processed = Path(cfg.paths.processed_dir)
    export_parquet(cfg.paths.db_path, list(all_tables), processed / "parquet")
    for name in ("dq_summary", "cohort_attrition", "status_by_start_year", "resolution_by_start_year",
                 "threshold_sensitivity", "terminated_share_by_year", "selection_profile",
                 "oncology_capture_summary", "oncology_review_sample", "oncology_text_only_mesh_terms",
                 "phase_audit", "date_audit", "start_type_by_status", "enum_check", "cohort_validation"):
        all_tables[name].to_csv(processed / f"{name}.csv", index=False)

    build_info = {
        "population_status": pop_status,
        "cohort_validated": validated,
        "output_state": output_state,
        "query_differences": query_diffs,
        "raw_run_id": manifest["run_id"],
        "raw_retrieved_at_utc": manifest.get("retrieved_at_utc_start"),
        "api_version": manifest.get("api_version"),
        "query_params": manifest.get("query_params"),
        "verification": verification.to_dict(),
        "exact_duplicate_records_collapsed": n_exact_dupes,
        "n_studies_raw": int(len(studies)),
        "n_eligible_all_statuses": int(len(eligible)),
        "n_cohort": int(len(cohort)),
        "analysis_window": {
            "start_year": pd.Timestamp(cfg.cohort.start_date_min).year,
            "end_year": result.analysis_end_year,
            "selected_by": "resolution_rule" if cfg.cohort.resolution_rule else "fixed config window",
        },
        "enum_check": enum_status,
        "cohort_config": asdict(cfg.cohort),
        "config_sha256": config_fingerprint,
        "built_at_utc": built_at,
        "unclassified_columns": unclassified,
    }
    (processed / "build_info.json").write_text(json.dumps(build_info, indent=2, default=str), encoding="utf-8")
    build_info["output_dir"] = _display_path(processed)
    (processed / "build_info.json").write_text(json.dumps(build_info, indent=2, default=str), encoding="utf-8")
    write_validation_report(processed / "cohort_validation_report.md", build_info, all_tables)
    if staging_dir is not None:
        build_info["output_dir"] = _display_path(_finalise_research_output(staging_dir, final_dir, output_state, built_at))
        (Path(build_info["output_dir"]) / "build_info.json").write_text(
            json.dumps(build_info, indent=2, default=str), encoding="utf-8")
    return build_info


def _display_path(path: Path) -> str:
    """Project-relative path when possible, so outputs never embed a local username."""
    from .config import PROJECT_ROOT
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _finalise_research_output(staging: Path, final_dir: Path, state: str, built_at: str) -> Path:
    """Promote an accepted staging build; quarantine a rejected one. Accepted outputs
    are never overwritten by a build that failed validation."""
    if state == "accepted":
        if final_dir.exists():
            backup = final_dir.parent / f"{final_dir.name}_previous"
            if backup.exists():
                shutil.rmtree(backup)
            final_dir.rename(backup)
        staging.rename(final_dir)
        return final_dir
    stamp = re.sub(r"[^0-9A-Za-z]", "", built_at)[:15]
    dest = final_dir.parent / f"{final_dir.name}_rejected" / stamp
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(dest)
    (dest / "INVALID_DO_NOT_USE.txt").write_text(
        "This research build FAILED validation and is quarantined. It must not be used for analysis.\n"
        "See cohort_validation_report.md for the failed checks.\n", encoding="utf-8")
    return dest
