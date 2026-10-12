"""Documented data-quality rules.

Rules FLAG records; they never silently delete them. Cohort construction then
excludes records only for rules marked ``blocking=True``, and that exclusion is
recorded in the attrition table. Rule IDs are referenced in DATA_DICTIONARY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "WITHDRAWN"}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str  # "error" | "warning" | "info"
    blocking: bool
    description: str
    check: Callable[[pd.DataFrame], pd.Series]  # boolean mask of violations


def _invalid_date(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda s: s[f"{col}_precision"].eq("invalid")


RULES: list[Rule] = [
    Rule("DQ01", "error", True, "Duplicate nct_id (all copies flagged)",
         lambda s: s["nct_id"].duplicated(keep=False)),
    Rule("DQ02", "error", True, "nct_id missing or not matching ^NCT\\d{8}$",
         lambda s: ~s["nct_id"].fillna("").astype(str).str.fullmatch(r"NCT\d{8}")),
    Rule("DQ03", "error", True, "overall_status missing",
         lambda s: s["overall_status"].isna()),
    Rule("DQ04", "warning", False, "start_date missing",
         lambda s: s["start_date"].isna() & s["start_date_precision"].ne("invalid")),
    Rule("DQ05", "warning", False, "start_date present but unparseable", _invalid_date("start_date")),
    Rule("DQ06", "warning", False, "completion_date present but unparseable", _invalid_date("completion_date")),
    Rule("DQ07", "warning", False, "completion_date earlier than start_date",
         lambda s: (s["completion_date"] < s["start_date"]).fillna(False)),
    Rule("DQ08", "warning", False, "primary_completion_date later than completion_date",
         lambda s: (s["primary_completion_date"] > s["completion_date"]).fillna(False)),
    Rule("DQ09", "error", False, "enrollment_count negative",
         lambda s: (pd.to_numeric(s["enrollment_count"], errors="coerce") < 0).fillna(False)),
    Rule("DQ10", "warning", False, "COMPLETED study reports zero enrollment",
         lambda s: s["overall_status"].eq("COMPLETED")
         & pd.to_numeric(s["enrollment_count"], errors="coerce").eq(0)),
    Rule("DQ11", "warning", False, "enrollment_count present without enrollment_type",
         lambda s: s["enrollment_count"].notna() & s["enrollment_type"].isna()),
    Rule("DQ12", "warning", False, "Terminal status but completion date still typed ESTIMATED",
         lambda s: s["overall_status"].isin(TERMINAL_STATUSES) & s["completion_date_type"].eq("ESTIMATED")),
    Rule("DQ13", "warning", False, "Interventional study with no phase recorded",
         lambda s: s["study_type"].eq("INTERVENTIONAL") & s["phase_group"].eq("NONE")),
    Rule("DQ16", "warning", False, "Minimum eligible age exceeds maximum eligible age",
         lambda s: (s["min_age_years"] > s["max_age_years"]).fillna(False)),
    Rule("DQ14", "info", False, "First submitted after start date (possible retrospective registration; "
         "month-precision dates make small lags unreliable)",
         lambda s: (pd.to_numeric(s["registration_lag_days"], errors="coerce") > 0).fillna(False)),
    Rule("DQ15", "info", False, "TERMINATED without a why_stopped explanation",
         lambda s: s["overall_status"].eq("TERMINATED") & s["why_stopped"].isna()),
]


def run_quality_checks(studies: pd.DataFrame, rules: list[Rule] = RULES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary, violations). violations has one row per (nct_id, rule)."""
    summary_rows, violation_frames = [], []
    n = len(studies)
    for rule in rules:
        mask = rule.check(studies).fillna(False).astype(bool)
        count = int(mask.sum())
        summary_rows.append(
            {
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "blocking": rule.blocking,
                "description": rule.description,
                "n_flagged": count,
                "pct_flagged": round(100 * count / n, 2) if n else 0.0,
            }
        )
        if count:
            violation_frames.append(
                pd.DataFrame({"nct_id": studies.loc[mask, "nct_id"].values, "rule_id": rule.rule_id})
            )
    summary = pd.DataFrame(summary_rows)
    violations = (
        pd.concat(violation_frames, ignore_index=True)
        if violation_frames
        else pd.DataFrame(columns=["nct_id", "rule_id"])
    )
    return summary, violations


def blocking_row_mask(studies: pd.DataFrame, rules: list[Rule] = RULES) -> pd.Series:
    """Row-level mask of records violating any blocking rule (works with duplicate IDs)."""
    mask = pd.Series(False, index=studies.index)
    for rule in rules:
        if rule.blocking:
            mask |= rule.check(studies).fillna(False).astype(bool)
    return mask
