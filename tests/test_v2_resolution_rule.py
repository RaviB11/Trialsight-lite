"""Issue 3: missing calendar years, ambiguous statuses, small-sample effects; criteria unchanged."""
import math

import pandas as pd
import pytest

from trialsight.cohort import resolution_by_start_year, select_analysis_end_year, wilson_interval
from trialsight.config import ResolutionRule, load_config


def _df(spec):
    """spec: {year: {status: count}} -> long DataFrame of synthetic studies."""
    rows = [(y, s) for y, d in spec.items() for s, n in d.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["start_year", "overall_status"])


GOOD = {"COMPLETED": 40, "TERMINATED": 10}          # 100% resolved, n=50


def test_missing_year_breaks_window_only_under_break_policy():
    df = _df({2008: GOOD, 2010: GOOD})               # 2009 has no studies
    skip = resolution_by_start_year(df, ResolutionRule(missing_year_policy="skip"), year_range=(2008, 2010))
    brk = resolution_by_start_year(df, ResolutionRule(missing_year_policy="break"), year_range=(2008, 2010))
    assert select_analysis_end_year(skip) == 2010    # original behaviour silently spans the gap
    assert select_analysis_end_year(brk) == 2008
    row = brk.set_index("start_year").loc[2009]
    assert row.n_total == 0 and row.decision_reason == "no_studies" and not row.meets_threshold


def test_trailing_empty_years_are_listed_under_break():
    brk = resolution_by_start_year(_df({2008: GOOD}), ResolutionRule(missing_year_policy="break"), (2008, 2011))
    assert brk["start_year"].tolist() == [2008, 2009, 2010, 2011]


def test_small_year_is_insufficient_not_qualifying():
    df = _df({2008: {"COMPLETED": 5}, 2009: GOOD})   # 5/5 resolved = 100%, but tiny
    t = resolution_by_start_year(df, ResolutionRule(min_studies_per_year=30))
    assert t.loc[0, "decision_reason"] == "insufficient_n" and select_analysis_end_year(t) is None
    assert select_analysis_end_year(resolution_by_start_year(df, ResolutionRule())) == 2009  # default n>=1


def test_ambiguous_handling_changes_share():
    df = _df({2010: {"COMPLETED": 9, "RECRUITING": 1, "UNKNOWN": 5}})
    excl = resolution_by_start_year(df, ResolutionRule(ambiguous_handling="exclude")).iloc[0]
    strict = resolution_by_start_year(df, ResolutionRule(ambiguous_handling="count_as_unresolved")).iloc[0]
    assert (excl.n_denominator, excl.resolved_share, bool(excl.meets_threshold)) == (10, 0.9, True)
    assert (strict.n_denominator, strict.resolved_share, bool(strict.meets_threshold)) == (15, 0.6, False)
    assert excl.ambiguous_share == strict.ambiguous_share == round(5 / 15, 4)


def test_wilson_interval_matches_closed_form():
    k, n, z = 9, 10, 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = wilson_interval(pd.Series([k, 0, 0]), pd.Series([n, 10, 0]))
    assert lo[0] == round(centre - half, 4) and hi[0] == round(centre + half, 4)
    assert lo[1] == 0.0 and pd.isna(lo[2]) and pd.isna(hi[2])


def test_ci_columns_present_and_ordered():
    t = resolution_by_start_year(_df({2010: {"COMPLETED": 27, "RECRUITING": 3}}), ResolutionRule())
    r = t.iloc[0]
    assert r.resolved_share_ci_low < r.resolved_share < r.resolved_share_ci_high


@pytest.mark.parametrize("kwargs", [{"ambiguous_handling": "drop"}, {"missing_year_policy": "fill"},
                                    {"min_studies_per_year": 0}])
def test_new_rule_options_validated(kwargs):
    with pytest.raises(ValueError):
        ResolutionRule(**kwargs)


def test_shipped_config_inclusion_criteria_unchanged():
    """Regression guard: v2 robustness work must not alter who is eligible."""
    c = load_config().cohort
    assert (c.start_date_min, c.start_date_max) == ("2008-01-01", "2025-12-31")
    assert c.phase_groups == ["PHASE2", "PHASE2_3", "PHASE3"]
    assert c.target_statuses == {"COMPLETED": 0, "TERMINATED": 1}
    assert c.require_oncology_mesh is True and c.oncology_mesh_terms == ["Neoplasms"]
    r = c.resolution_rule
    assert r.min_resolved_share == 0.90 and r.ambiguous_handling == "exclude"
    assert r.resolved_statuses == ["COMPLETED", "TERMINATED", "WITHDRAWN"]
    assert r.ambiguous_statuses == ["UNKNOWN"]
    # fail-closed robustness settings (can only shrink the window)
    assert (r.missing_year_policy, r.min_studies_per_year) == ("break", 30)
