"""Cohort diagnostics. These functions only DESCRIBE consequences of the cohort
definition; none of them changes inclusion criteria.

- threshold_sensitivity     : how resolution threshold / ambiguous handling move the window and cohort
- terminated_share_by_year  : right-censoring signal across all eligible start years
- selection_profile         : in-window cohort vs in-window studies excluded by status
- oncology_capture          : what strict MeSH matching excludes, incl. cancer text without MeSH
- phase_audit / date_audit  : API phase and date fields versus the intended definitions
- enum_check                : config status/phase tokens versus the API's own enums snapshot
- validation_checks         : pass/fail gate used to label a cohort as validated
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .cohort import resolution_by_start_year, select_analysis_end_year, wilson_interval
from .config import CohortConfig, ResolutionRule

CRITICAL, WARNING = "critical", "warning"


def _rule(cfg: CohortConfig) -> ResolutionRule:
    return cfg.resolution_rule or ResolutionRule()


def _year_range(cfg: CohortConfig) -> tuple[int, int]:
    return pd.Timestamp(cfg.start_date_min).year, pd.Timestamp(cfg.start_date_max).year


def _target_counts(df: pd.DataFrame, target_statuses: dict[str, int]) -> tuple[int, int]:
    mapped = df["overall_status"].map(target_statuses)
    return int((mapped == 0).sum()), int((mapped == 1).sum())


# ----------------------------------------------------------------- issue 4
def threshold_sensitivity(
    eligible: pd.DataFrame, cfg: CohortConfig, thresholds: list[float],
    handlings: tuple[str, ...] = ("exclude", "count_as_unresolved"),
) -> pd.DataFrame:
    base = _rule(cfg)
    rows = []
    for handling in handlings:
        for thr in sorted(set(thresholds) | {base.min_resolved_share}):
            rule = replace(base, min_resolved_share=thr, ambiguous_handling=handling)
            res = resolution_by_start_year(eligible, rule, year_range=_year_range(cfg))
            end = select_analysis_end_year(res)
            within = eligible[eligible["start_year"].le(end)] if end is not None else eligible.iloc[0:0]
            n0, n1 = _target_counts(within, cfg.target_statuses)
            lo, hi = wilson_interval(pd.Series([n1]), pd.Series([n0 + n1]))
            status = within["overall_status"]
            rows.append({
                "ambiguous_handling": handling,
                "threshold": thr,
                "is_configured_rule": bool(thr == base.min_resolved_share and handling == base.ambiguous_handling),
                "end_year": end,
                "n_years_in_window": (end - _year_range(cfg)[0] + 1) if end is not None else 0,
                "cohort_n": n0 + n1,
                "n_completed": n0,
                "n_terminated": n1,
                "terminated_share": round(n1 / (n0 + n1), 4) if (n0 + n1) else None,
                "terminated_ci_low": lo.iloc[0],
                "terminated_ci_high": hi.iloc[0],
                "in_window_unresolved": int(status.isin(rule.unresolved_statuses).sum()),
                "in_window_ambiguous": int(status.isin(rule.ambiguous_statuses).sum()),
                "in_window_other_non_target": int(
                    len(within) - (n0 + n1) - status.isin(rule.unresolved_statuses).sum()
                    - status.isin(rule.ambiguous_statuses).sum()
                ),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- issue 3
def terminated_share_by_year(eligible: pd.DataFrame, target_statuses: dict[str, int]) -> pd.DataFrame:
    cols = ["start_year", "n_completed", "n_terminated", "terminated_share", "ci_low", "ci_high"]
    if eligible.empty:
        return pd.DataFrame(columns=cols)
    mapped = eligible["overall_status"].map(target_statuses)
    t = pd.DataFrame({"start_year": eligible["start_year"].astype(int).values,
                      "c": (mapped == 0).values, "t": (mapped == 1).values})
    g = t.groupby("start_year").agg(n_completed=("c", "sum"), n_terminated=("t", "sum")).reset_index()
    n = g["n_completed"] + g["n_terminated"]
    g["terminated_share"] = (g["n_terminated"] / n.where(n > 0).astype(float)).round(4)
    g["ci_low"], g["ci_high"] = wilson_interval(g["n_terminated"], n)
    return g[cols]


def selection_profile(
    eligible: pd.DataFrame, end_year: int | None, cfg: CohortConfig, columns: list[str]
) -> pd.DataFrame:
    """Within the analysis window: distribution of key fields by status group."""
    cols = ["column", "value", "group", "n", "share_within_group"]
    if end_year is None or eligible.empty:
        return pd.DataFrame(columns=cols)
    rule = _rule(cfg)
    w = eligible[eligible["start_year"].le(end_year)].copy()
    s = w["overall_status"]
    w["_group"] = "other_non_target"
    w.loc[s.isin(rule.unresolved_statuses), "_group"] = "unresolved"
    w.loc[s.isin(rule.ambiguous_statuses), "_group"] = "ambiguous"
    w.loc[s.isin(list(cfg.target_statuses)), "_group"] = "cohort"
    out = []
    for col in [c for c in columns if c in w.columns]:
        vals = w[col].astype(object).where(w[col].notna(), "(missing)")
        counts = pd.crosstab(vals, w["_group"])
        shares = counts / counts.sum()
        for value in counts.index:
            for group in counts.columns:
                out.append({"column": col, "value": str(value), "group": group,
                            "n": int(counts.loc[value, group]),
                            "share_within_group": round(float(shares.loc[value, group]), 4)})
    return pd.DataFrame(out, columns=cols)


# ----------------------------------------------------------------- issue 5
def cancer_text_regex(terms: list[str]) -> re.Pattern:
    alternatives = "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))
    return re.compile(rf"\b({alternatives})\b", re.IGNORECASE)


def oncology_capture(
    pre_oncology: pd.DataFrame, conditions: pd.DataFrame, mesh: pd.DataFrame,
    text_terms: list[str], review_sample_size: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify the pre-oncology population by MeSH match and condition/keyword text.

    Returns (summary, review_sample, top_mesh_terms_for_text_only_records).
    Only records the API query returned can be measured; studies the
    ``query.cond`` search never returned are invisible here.
    """
    sum_cols = ["category", "included_by_current_rule", "n", "share_of_pre_oncology",
                "n_completed", "n_terminated", "n_other_status"]
    review_cols = ["nct_id", "category", "brief_title", "overall_status", "start_year",
                   "matched_terms", "conditions", "mesh_terms"]
    if pre_oncology.empty:
        return (pd.DataFrame(columns=sum_cols), pd.DataFrame(columns=review_cols),
                pd.DataFrame(columns=["mesh_term", "n"]))

    ids = set(pre_oncology["nct_id"])
    pattern = cancer_text_regex(text_terms)
    cond = conditions[conditions["nct_id"].isin(ids)].copy()
    cond["matched"] = cond["term"].fillna("").astype(str).str.findall(pattern)
    matched_terms = (
        cond.explode("matched").dropna(subset=["matched"])
        .assign(matched=lambda d: d["matched"].str.lower())
        .groupby("nct_id")["matched"].agg(lambda x: "|".join(sorted(set(x))))
    )
    cond_text = cond.groupby("nct_id")["term"].agg(lambda x: " | ".join(map(str, x)))
    m = mesh[mesh["nct_id"].isin(ids) & mesh["kind"].isin(["mesh", "ancestor"])]
    has_mesh = set(m["nct_id"])
    mesh_terms = m[m["kind"] == "mesh"].groupby("nct_id")["term"].agg(lambda x: " | ".join(map(str, x)))

    df = pre_oncology[["nct_id", "brief_title", "overall_status", "start_year", "is_oncology_mesh"]].copy()
    df["cancer_text"] = df["nct_id"].isin(matched_terms.index)
    df["has_any_mesh"] = df["nct_id"].isin(has_mesh)
    df["category"] = "excluded: no cancer text, no MeSH annotations"
    df.loc[~df["is_oncology_mesh"] & ~df["cancer_text"] & df["has_any_mesh"],
           "category"] = "excluded: no cancer text, non-neoplasm MeSH only"
    df.loc[~df["is_oncology_mesh"] & df["cancer_text"] & df["has_any_mesh"],
           "category"] = "excluded: cancer text, MeSH present but not Neoplasms"
    df.loc[~df["is_oncology_mesh"] & df["cancer_text"] & ~df["has_any_mesh"],
           "category"] = "excluded: cancer text, no MeSH annotations"
    df.loc[df["is_oncology_mesh"] & df["cancer_text"], "category"] = "included: Neoplasms MeSH + cancer text"
    df.loc[df["is_oncology_mesh"] & ~df["cancer_text"], "category"] = "included: Neoplasms MeSH, no cancer text"

    rows = []
    for cat, g in df.groupby("category"):
        st = g["overall_status"]
        rows.append({
            "category": cat, "included_by_current_rule": cat.startswith("included"), "n": len(g),
            "share_of_pre_oncology": round(len(g) / len(df), 4),
            "n_completed": int(st.eq("COMPLETED").sum()), "n_terminated": int(st.eq("TERMINATED").sum()),
            "n_other_status": int((~st.isin(["COMPLETED", "TERMINATED"])).sum()),
        })
    summary = pd.DataFrame(rows, columns=sum_cols).sort_values("category").reset_index(drop=True)

    review = df[~df["is_oncology_mesh"] & df["cancer_text"]].copy()
    review["matched_terms"] = review["nct_id"].map(matched_terms)
    review["conditions"] = review["nct_id"].map(cond_text)
    review["mesh_terms"] = review["nct_id"].map(mesh_terms)
    review = review.sort_values("nct_id").head(review_sample_size)[
        ["nct_id", "category", "brief_title", "overall_status", "start_year",
         "matched_terms", "conditions", "mesh_terms"]
    ].reset_index(drop=True)

    text_only_ids = set(df.loc[df["category"] == "excluded: cancer text, MeSH present but not Neoplasms", "nct_id"])
    top = (
        m[(m["kind"] == "mesh") & m["nct_id"].isin(text_only_ids)]
        .groupby("term").size().sort_values(ascending=False).head(25)
        .rename("n").reset_index().rename(columns={"term": "mesh_term"})
    )
    return summary, review, top


# ----------------------------------------------------------------- issue 6
def phase_audit(studies: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    s = studies.copy()
    s["phases"] = s["phases"].astype(object).where(s["phases"].notna(), "(none)")
    g = s.groupby(["phases", "phase_group"], dropna=False).agg(
        n=("nct_id", "size"),
        n_interventional=("study_type", lambda x: int(x.eq("INTERVENTIONAL").sum())),
    ).reset_index()
    g["included_by_definition"] = g["phase_group"].isin(cfg.phase_groups)
    g["note"] = ""
    g.loc[~g["included_by_definition"] & g["phases"].str.contains("PHASE2|PHASE3", regex=True),
          "note"] = "Returned by API phase filter (contains PHASE2/PHASE3) but excluded by the local Phase II/III definition"
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def date_audit(studies: pd.DataFrame, eligible: pd.DataFrame, cohort: pd.DataFrame,
               cfg: CohortConfig, retrieved_at: pd.Timestamp | None) -> pd.DataFrame:
    lo, hi = pd.Timestamp(cfg.start_date_min), pd.Timestamp(cfg.start_date_max)
    n_all = len(studies)
    rows: list[dict[str, Any]] = []

    def add(scope, check, n, denom, detail=""):
        rows.append({"scope": scope, "check": check, "n": int(n), "denominator": int(denom),
                     "share": round(n / denom, 4) if denom else None, "detail": detail})

    sd = studies["start_date"]
    add("retrieved", "start date missing", (sd.isna() & studies["start_date_precision"].ne("invalid")).sum(), n_all)
    add("retrieved", "start date unparseable", studies["start_date_precision"].eq("invalid").sum(), n_all)
    add("retrieved", "start date outside configured window (API RANGE filter not honoured?)",
        (sd.notna() & ~sd.between(lo, hi)).sum(), n_all)
    add("retrieved", "study type not INTERVENTIONAL (API filter not honoured?)",
        studies["study_type"].ne("INTERVENTIONAL").sum(), n_all)
    for prec in ("day", "month", "year"):
        add("eligible", f"start date precision = {prec}", eligible["start_date_precision"].eq(prec).sum(), len(eligible),
            "month/year precision is anchored to the first day of the period")
    est = eligible["start_date_type"].eq("ESTIMATED")
    add("eligible", "start date type ESTIMATED", est.sum(), len(eligible),
        "anticipated start; start year may not reflect when the study actually began")
    if retrieved_at is not None:
        ts = pd.Timestamp(retrieved_at).tz_localize(None) if pd.Timestamp(retrieved_at).tzinfo else pd.Timestamp(retrieved_at)
        add("eligible", "start date ESTIMATED but before retrieval date",
            (est & eligible["start_date"].lt(ts)).sum(), len(eligible), "stale record: anticipated date already passed")
    add("cohort", "start date type ESTIMATED among COMPLETED/TERMINATED",
        cohort["start_date_type"].eq("ESTIMATED").sum(), len(cohort), "terminal studies should normally have an ACTUAL start")
    add("cohort", "completion date type ESTIMATED among COMPLETED/TERMINATED",
        cohort["completion_date_type"].eq("ESTIMATED").sum(), len(cohort))
    add("cohort", "start date type missing", cohort["start_date_type"].isna().sum(), len(cohort))
    return pd.DataFrame(rows)


def start_type_by_status(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame(columns=["overall_status", "start_date_type", "n"])
    t = eligible.assign(start_date_type=eligible["start_date_type"].astype(object).where(
        eligible["start_date_type"].notna(), "(missing)"))
    return t.groupby(["overall_status", "start_date_type"]).size().rename("n").reset_index()


def _collect_enum_strings(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value" and isinstance(v, str):
                out.add(v)
            _collect_enum_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                out.add(item)
            else:
                _collect_enum_strings(item, out)


def enum_check(cfg: CohortConfig, enums_path: Path) -> tuple[str, pd.DataFrame]:
    """Check config tokens exist in the API's own enums snapshot (tolerant parser)."""
    rule = _rule(cfg)
    tokens: list[tuple[str, str]] = [(t, "target_statuses") for t in cfg.target_statuses]
    tokens += [(t, "resolved_statuses") for t in rule.resolved_statuses]
    tokens += [(t, "unresolved_statuses") for t in rule.unresolved_statuses]
    tokens += [(t, "ambiguous_statuses") for t in rule.ambiguous_statuses]
    phase_tokens = sorted({p for g in cfg.phase_groups for p in
                           ({"PHASE2_3": ["PHASE2", "PHASE3"]}.get(g, [g]))})
    tokens += [(t, "phase_groups") for t in phase_tokens]
    tokens += [("INTERVENTIONAL", "study_type"), ("ACTUAL", "date/enrollment type"), ("ESTIMATED", "date/enrollment type")]
    df = pd.DataFrame(tokens, columns=["token", "config_source"]).drop_duplicates()
    if not Path(enums_path).exists():
        df["present_in_api_enums"] = None
        return "not_checked: enums snapshot not found (run `trialsight inspect`)", df
    found: set[str] = set()
    _collect_enum_strings(json.loads(Path(enums_path).read_text(encoding="utf-8")), found)
    if not found:
        df["present_in_api_enums"] = None
        return "not_checked: enums snapshot format not recognised", df
    df["present_in_api_enums"] = df["token"].isin(found)
    missing = df.loc[~df["present_in_api_enums"], "token"].tolist()
    return ("pass" if not missing else f"fail: tokens not in API enums: {missing}"), df


# ----------------------------------------------------------------- gate (issue 7)
def validation_checks(
    *, verification: dict, eligible: pd.DataFrame, cohort: pd.DataFrame, resolution: pd.DataFrame,
    end_year: int | None, cfg: CohortConfig, enum_status: str, date_checks: pd.DataFrame,
    query_diffs: list[str] | None = None,
) -> pd.DataFrame:
    rule = _rule(cfg)
    rows = []

    def add(cid, severity, description, ok, detail="", not_checked=False):
        status = "not_checked" if not_checked else ("pass" if ok else ("fail" if severity == CRITICAL else "warn"))
        rows.append({"check_id": cid, "severity": severity, "description": description,
                     "status": status, "detail": detail})

    add("V01", CRITICAL, "Raw download is research-grade (verified, or explicitly reconciled)",
        verification.get("research_grade", False),
        f"status={verification.get('status')}; reasons={verification.get('reasons')}")
    add("V02", CRITICAL, "Raw page checksums valid", verification.get("checksums_ok", False))
    add("V03", CRITICAL, "Resolution rule selected an analysis window", end_year is not None, f"end_year={end_year}")
    in_win = resolution[resolution["start_year"].le(end_year)] if end_year is not None else resolution.iloc[0:0]
    min_n = int(in_win["n_denominator"].min()) if len(in_win) else 0
    add("V04", WARNING, "Every year in the window has >= min_studies_per_year resolvable studies",
        len(in_win) > 0 and min_n >= rule.min_studies_per_year,
        f"smallest yearly denominator={min_n}, min_studies_per_year={rule.min_studies_per_year}")
    years = sorted(eligible["start_year"].dropna().astype(int).unique())
    if end_year is not None and years:
        missing = [y for y in range(years[0], end_year + 1) if y not in set(years)]
    else:
        missing = []
    add("V05", WARNING, "No empty calendar years inside the window", not missing, f"missing={missing}")
    other = eligible[~eligible["overall_status"].isin(
        rule.resolved_statuses + rule.unresolved_statuses + rule.ambiguous_statuses)]
    add("V06", CRITICAL, "Every eligible status is mapped to a resolution group", other.empty,
        f"unmapped={sorted(other['overall_status'].astype(str).unique().tolist())}")
    add("V07", CRITICAL, "Cohort is non-empty and contains both outcome classes",
        len(cohort) > 0 and cohort["target_terminated"].nunique() == 2, f"n={len(cohort)}")
    add("V08", WARNING, "Config status/phase tokens exist in API enums snapshot",
        enum_status == "pass", enum_status, not_checked=enum_status.startswith("not_checked"))

    def date_n(check):
        hit = date_checks.loc[date_checks["check"] == check, "n"]
        return int(hit.iloc[0]) if len(hit) else 0
    outside = date_n("start date outside configured window (API RANGE filter not honoured?)")
    nonint = date_n("study type not INTERVENTIONAL (API filter not honoured?)")
    add("V09", WARNING, "API filters honoured (no out-of-window start dates, all interventional)",
        outside == 0 and nonint == 0, f"out_of_window={outside}, non_interventional={nonint}")
    add("V10", CRITICAL, "Raw run's cohort-defining query matches the current configuration",
        not query_diffs, f"differences={query_diffs or []}")
    return pd.DataFrame(rows)


def cohort_validated(checks: pd.DataFrame) -> bool:
    critical = checks[checks["severity"] == CRITICAL]
    return bool(len(critical)) and bool(critical["status"].eq("pass").all())
