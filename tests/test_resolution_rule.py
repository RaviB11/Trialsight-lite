from dataclasses import replace

import pandas as pd
import pytest

from trialsight.cohort import build_cohort, resolution_by_start_year, select_analysis_end_year
from trialsight.config import ResolutionRule
from trialsight.flatten import flatten_studies
from test_quality_and_cohort import CFG

CFG_RULE = replace(CFG, start_date_max="2025-12-31", resolution_rule=ResolutionRule(min_resolved_share=0.9))


def _df(rows):
    return pd.DataFrame(rows, columns=["start_year", "overall_status"])


def test_unknown_excluded_from_share_but_reported():
    df = _df([(2010, "COMPLETED")] * 9 + [(2010, "RECRUITING")] + [(2010, "UNKNOWN")] * 5)
    t = resolution_by_start_year(df, ResolutionRule())
    row = t.iloc[0]
    assert (row.n_total, row.n_resolved, row.n_unresolved, row.n_ambiguous) == (15, 9, 1, 5)
    assert row.resolved_share == pytest.approx(0.9)
    assert row.ambiguous_share == pytest.approx(5 / 15, abs=1e-4)
    assert bool(row.meets_threshold)  # exactly at threshold counts


def test_withdrawn_counts_as_resolved():
    t = resolution_by_start_year(_df([(2011, "WITHDRAWN"), (2011, "TERMINATED")]), ResolutionRule())
    assert t.iloc[0].resolved_share == 1.0


def test_year_with_only_unknown_fails():
    t = resolution_by_start_year(_df([(2012, "UNKNOWN")]), ResolutionRule())
    assert pd.isna(t.iloc[0].resolved_share) and not bool(t.iloc[0].meets_threshold)


def test_window_stops_at_first_failing_year_without_gaps():
    t = pd.DataFrame({"start_year": [2008, 2009, 2010, 2011], "meets_threshold": [True, True, False, True]})
    assert select_analysis_end_year(t) == 2009  # 2011 passes but is after a gap
    t.loc[0, "meets_threshold"] = False
    assert select_analysis_end_year(t) is None


@pytest.mark.parametrize("kwargs", [
    {"min_resolved_share": 0}, {"min_resolved_share": 1.5},
    {"resolved_statuses": ["COMPLETED"], "unresolved_statuses": ["COMPLETED"]},
])
def test_rule_validation(kwargs):
    with pytest.raises(ValueError):
        ResolutionRule(**kwargs)


def test_cohort_with_rule(synthetic_studies):
    # Eligible (pre-outcome): 2012 TERMINATED; 2014 WITHDRAWN + COMPLETED; 2016 RECRUITING.
    t = flatten_studies(synthetic_studies)
    r = build_cohort(t["studies"], t["mesh"], CFG_RULE)
    assert r.analysis_end_year == 2014
    assert r.resolution_by_start_year["start_year"].tolist() == [2012, 2014, 2016]
    assert r.resolution_by_start_year["in_analysis_window"].tolist() == [True, True, False]
    assert r.attrition["n_remaining"].tolist() == [11, 8, 8, 7, 5, 4, 3, 2]
    assert "Start year <= 2014" in r.attrition["criterion"].iloc[-2]
    assert sorted(r.cohort["nct_id"]) == ["NCT99990002", "NCT99990009"]
    # eligible population keeps recent/unresolved studies for survival analysis
    assert sorted(r.eligible_population["nct_id"]) == ["NCT99990002", "NCT99990003", "NCT99990004", "NCT99990009"]
    assert "_dq_blocking" not in r.eligible_population.columns


def test_no_qualifying_year_gives_empty_cohort_not_crash(synthetic_studies):
    t = flatten_studies(synthetic_studies)
    strict = replace(CFG_RULE, start_date_min="2016-01-01")  # only the RECRUITING study remains eligible
    r = build_cohort(t["studies"], t["mesh"], strict)
    assert r.analysis_end_year is None and r.cohort.empty
    assert "no start year reached" in r.attrition["criterion"].iloc[-2]
