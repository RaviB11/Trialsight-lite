"""Plain-Markdown cohort validation report (no extra dependencies)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEV_BANNER = (
    "> **DEVELOPMENT BUILD - NOT A RESEARCH POPULATION.** This build comes from a raw download that is\n"
    "> incomplete or unverified. Its counts, analysis window and proportions describe only this download and\n"
    "> must not be reported as findings or used to train or evaluate models for reported results.\n"
)
UNVALIDATED_BANNER = (
    "> **Research-grade download, but the cohort is NOT validated.** At least one critical check failed;\n"
    "> see Validation checks. Modelling is blocked until this is resolved.\n"
)


MIN_EVENTS_STABLE = 20   # subgroups with fewer events in either class are flagged as unstable


def fmt_cell(v) -> str:
    """Display value: undefined -> 'n/a'; no negative zero."""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        if v == 0:
            v = 0.0
        return f"{v:g}" if abs(v) >= 1e-4 or v == 0 else f"{v:.2e}"
    return str(v)


def flag_stability(df: pd.DataFrame, pos: str = "n_positive", n: str = "n") -> pd.DataFrame:
    """Add n_negative and a stability flag (fewer than MIN_EVENTS_STABLE events in either class)."""
    out = df.copy()
    out["n_negative"] = out[n] - out[pos]
    ok = (out[pos] >= MIN_EVENTS_STABLE) & (out["n_negative"] >= MIN_EVENTS_STABLE)
    out["stability"] = ok.map({True: "ok", False: f"UNSTABLE (<{MIN_EVENTS_STABLE} events in a class)"})
    return out


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df.head(max_rows).astype(object).map(fmt_cell)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in the CSV/DuckDB table)_")
    return "\n".join(lines) + "\n"


def write_validation_report(path: Path, info: dict[str, Any], t: dict[str, pd.DataFrame]) -> Path:
    v = info["verification"]
    status = info["population_status"]
    parts = ["# Cohort validation report\n"]
    if status == "development_not_for_research":
        parts.append(DEV_BANNER)
    elif not info["cohort_validated"]:
        parts.append(UNVALIDATED_BANNER)
    parts.append(
        f"\n- Population status: **{status}**\n- Cohort validated: **{info['cohort_validated']}**\n"
        f"- Raw run: `{info['raw_run_id']}` retrieved {info['raw_retrieved_at_utc']}\n"
        f"- Download verification: **{v['status']}** (research_grade={v['research_grade']})\n"
        f"- API totalCount at start: {v['reported_count']}; at end: {v['count_at_end']}; "
        f"records retrieved: {v['retrieved_records']}; unique nctIds: {v['unique_nct_ids']}\n"
        f"- Duplicates: {v['conflicting_duplicate_ids']} conflicting IDs, {v['exact_duplicate_records']} identical extra "
        f"copies (collapsed in build: {info['exact_duplicate_records_collapsed']})\n"
        f"- Pagination anomalies: {len(v['page_anomalies'])}\n"
        f"- Analysis window: {info['analysis_window']['start_year']}-{info['analysis_window']['end_year']} "
        f"({info['analysis_window']['selected_by']})\n"
        f"- Eligible (all statuses): {info['n_eligible_all_statuses']}; primary cohort: {info['n_cohort']}\n"
        f"- Config fingerprint: `{info['config_sha256'][:16]}`; built {info['built_at_utc']}\n"
    )
    if v["reasons"]:
        parts.append("\nVerification reasons:\n" + "".join(f"- {r}\n" for r in v["reasons"]))
    sections = [
        ("Validation checks", "cohort_validation"),
        ("Cohort attrition", "cohort_attrition"),
        ("Resolution by start year (window evidence)", "resolution_by_start_year"),
        ("Threshold and ambiguous-status sensitivity", "threshold_sensitivity"),
        ("Terminated share by start year (censoring signal)", "terminated_share_by_year"),
        ("Oncology capture under strict MeSH matching", "oncology_capture_summary"),
        ("Top MeSH terms: cancer text present, Neoplasms MeSH absent", "oncology_text_only_mesh_terms"),
        ("Phase categories returned by the API", "phase_audit"),
        ("Date-field audit", "date_audit"),
        ("Start-date type by status (eligible)", "start_type_by_status"),
        ("Config tokens vs API enums", "enum_check"),
        ("Data-quality rules", "dq_summary"),
    ]
    for title, name in sections:
        parts.append(f"\n## {title}\n\n" + md_table(t.get(name)))
    parts.append(
        "\n## Notes\n\n- Diagnostics describe consequences of the cohort definition; they do not change it.\n"
        "- Oncology capture can only measure records the API query returned.\n"
        "- `oncology_review_sample.csv` lists excluded records with cancer-related condition text for manual review.\n"
    )
    Path(path).write_text("".join(parts), encoding="utf-8")
    return Path(path)
