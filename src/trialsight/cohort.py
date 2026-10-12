"""Cohort construction with a CONSORT-style attrition table.

The outcome-status filter is applied LAST, so the population immediately before
it (all eligible studies regardless of status) can be tabulated by start year.
That table is the primary evidence for selection/right-censoring bias.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from .config import CohortConfig, ResolutionRule
from .quality import blocking_row_mask

log = logging.getLogger(__name__)

RESOLUTION_COLUMNS = [
    "start_year", "n_total", "n_resolved", "n_unresolved", "n_ambiguous", "n_other",
    "resolved_share", "ambiguous_share", "meets_threshold",
]


class CohortResult(NamedTuple):
    cohort: pd.DataFrame                    # primary modelling cohort (target statuses only)
    attrition: pd.DataFrame                 # CONSORT-style step table
    status_by_start_year: pd.DataFrame      # eligible population, all statuses, all years
    resolution_by_start_year: pd.DataFrame  # evidence used to choose the analysis window
    eligible_population: pd.DataFrame       # all statuses, full window (for survival analysis)
    analysis_end_year: int | None           # None if no rule configured or no year qualified
    pre_oncology_population: pd.DataFrame | None = None  # passed every filter except oncology MeSH


def flag_oncology(studies: pd.DataFrame, mesh: pd.DataFrame, terms: list[str]) -> pd.Series:
    wanted = {t.lower() for t in terms}
    if mesh.empty:
        return pd.Series(False, index=studies.index)
    hits = mesh.loc[mesh["term"].fillna("").str.lower().isin(wanted), "nct_id"].unique()
    return studies["nct_id"].isin(hits)


EXTRA_RESOLUTION_COLUMNS = ["n_denominator", "resolved_share_ci_low", "resolved_share_ci_high", "decision_reason"]


def wilson_interval(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """Wilson score interval for k successes out of n (NaN where n == 0)."""
    k = k.astype(float)
    n = n.astype(float)
    nz = n.where(n > 0)
    p = k / nz
    denom = 1 + z**2 / nz
    centre = (p + z**2 / (2 * nz)) / denom
    half = z * np.sqrt(p * (1 - p) / nz + z**2 / (4 * nz**2)) / denom
    return (centre - half).clip(0, 1).round(4), (centre + half).clip(0, 1).round(4)


def resolution_by_start_year(
    eligible: pd.DataFrame, rule: ResolutionRule, year_range: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Per start year, how many eligible studies have reached a final status.

    ``year_range`` is used only when ``rule.missing_year_policy == "break"``:
    calendar years in the range with no studies are added as failing rows.
    """
    cols = RESOLUTION_COLUMNS + EXTRA_RESOLUTION_COLUMNS
    if eligible.empty and not (rule.missing_year_policy == "break" and year_range):
        return pd.DataFrame(columns=cols)
    status = eligible["overall_status"]
    flags = pd.DataFrame({
        "start_year": eligible["start_year"].astype(int).values,
        "resolved": status.isin(rule.resolved_statuses).values,
        "unresolved": status.isin(rule.unresolved_statuses).values,
        "ambiguous": status.isin(rule.ambiguous_statuses).values,
    })
    flags["other"] = ~(flags["resolved"] | flags["unresolved"] | flags["ambiguous"])
    t = (
        flags.groupby("start_year")
        .agg(n_total=("resolved", "size"), n_resolved=("resolved", "sum"),
             n_unresolved=("unresolved", "sum"), n_ambiguous=("ambiguous", "sum"), n_other=("other", "sum"))
        .sort_index()
    )
    if rule.missing_year_policy == "break" and year_range:
        t = t.reindex(range(year_range[0], year_range[1] + 1), fill_value=0)
        t.index.name = "start_year"
    t = t.reset_index()
    for c in ("n_total", "n_resolved", "n_unresolved", "n_ambiguous", "n_other"):
        t[c] = t[c].astype(int)

    denom = t["n_resolved"] + t["n_unresolved"]
    if rule.ambiguous_handling == "count_as_unresolved":
        denom = denom + t["n_ambiguous"]
    t["n_denominator"] = denom.astype(int)
    t["resolved_share"] = (t["n_resolved"] / denom.where(denom > 0).astype(float)).round(4)
    t["ambiguous_share"] = (t["n_ambiguous"] / t["n_total"].where(t["n_total"] > 0).astype(float)).round(4)
    t["resolved_share_ci_low"], t["resolved_share_ci_high"] = wilson_interval(t["n_resolved"], denom)

    reason = pd.Series("meets_threshold", index=t.index, dtype=object)
    reason[t["resolved_share"].lt(rule.min_resolved_share)] = "below_threshold"
    reason[(denom > 0) & (denom < int(rule.min_studies_per_year))] = "insufficient_n"
    reason[denom == 0] = "no_resolvable_studies"
    reason[t["n_total"] == 0] = "no_studies"
    t["decision_reason"] = reason
    t["meets_threshold"] = reason.eq("meets_threshold")
    return t[cols].reset_index(drop=True)


def select_analysis_end_year(resolution: pd.DataFrame) -> int | None:
    """Last year of the initial consecutive run of qualifying start years.

    Rows are taken in ascending year order. With missing_year_policy="skip",
    empty calendar years have no row and are ignored; with "break", they are
    present as failing rows and end the window. Returns None if the earliest
    row already fails.
    """
    end = None
    for year, ok in zip(resolution["start_year"], resolution["meets_threshold"]):
        if not bool(ok):
            break
        end = int(year)
    return end


def build_cohort(studies: pd.DataFrame, mesh: pd.DataFrame, cfg: CohortConfig) -> CohortResult:
    df = studies.copy()
    df["is_oncology_mesh"] = flag_oncology(df, mesh, cfg.oncology_mesh_terms)
    df["_dq_blocking"] = blocking_row_mask(df)

    attrition: list[dict] = [{"step": 0, "criterion": "Retrieved from API query", "n_remaining": len(df), "n_excluded": 0}]

    def apply(criterion: str, keep: Callable[[pd.DataFrame], pd.Series]) -> None:
        nonlocal df
        before = len(df)
        df = df[keep(df).fillna(False).astype(bool)]
        attrition.append(
            {"step": len(attrition), "criterion": criterion, "n_remaining": len(df), "n_excluded": before - len(df)}
        )

    lo, hi = pd.Timestamp(cfg.start_date_min), pd.Timestamp(cfg.start_date_max)
    apply("Passes blocking data-quality rules", lambda d: ~d["_dq_blocking"])
    apply("Study type INTERVENTIONAL", lambda d: d["study_type"].eq("INTERVENTIONAL"))
    apply(f"Phase group in {cfg.phase_groups}", lambda d: d["phase_group"].isin(cfg.phase_groups))
    apply(f"Start date within {cfg.start_date_min}..{cfg.start_date_max}",
          lambda d: d["start_date"].between(lo, hi))
    pre_oncology = df.drop(columns=["_dq_blocking"]).reset_index(drop=True)
    if cfg.require_oncology_mesh:
        apply(f"Condition mapped to MeSH {cfg.oncology_mesh_terms}", lambda d: d["is_oncology_mesh"])

    status_by_year = (
        pd.crosstab(df["start_year"], df["overall_status"], dropna=False)
        if len(df) else pd.DataFrame()
    )
    if len(status_by_year):
        status_by_year.columns.name = None
        status_by_year = status_by_year.reset_index()

    eligible = df.drop(columns=["_dq_blocking"]).reset_index(drop=True)

    resolution = pd.DataFrame(columns=RESOLUTION_COLUMNS)
    end_year: int | None = None
    rule = cfg.resolution_rule
    if rule is not None:
        resolution = resolution_by_start_year(df, rule, year_range=(lo.year, hi.year))
        end_year = select_analysis_end_year(resolution)
        resolution["in_analysis_window"] = (
            resolution["start_year"].le(end_year) if end_year is not None else False
        )
        if end_year is None:
            log.warning("Resolution rule: earliest start year is below %.0f%% resolved; cohort will be empty. "
                        "Inspect resolution_by_start_year.", 100 * rule.min_resolved_share)
            apply(f"Resolution rule: no start year reached {rule.min_resolved_share:.0%} resolved",
                  lambda d: pd.Series(False, index=d.index))
        else:
            apply(f"Start year <= {end_year} (each year >= {rule.min_resolved_share:.0%} resolved)",
                  lambda d: d["start_year"].le(end_year))

    statuses = list(cfg.target_statuses)
    apply(f"Overall status in {statuses} (outcome-defined)", lambda d: d["overall_status"].isin(statuses))

    df = df.drop(columns=["_dq_blocking"])
    df["target_terminated"] = df["overall_status"].map(cfg.target_statuses).astype("Int64")
    return CohortResult(
        cohort=df.reset_index(drop=True),
        attrition=pd.DataFrame(attrition),
        status_by_start_year=status_by_year,
        resolution_by_start_year=resolution,
        eligible_population=eligible,
        analysis_end_year=end_year,
        pre_oncology_population=pre_oncology,
    )
