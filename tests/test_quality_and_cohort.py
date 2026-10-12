from trialsight.cohort import build_cohort, flag_oncology
from trialsight.config import CohortConfig
from trialsight.flatten import flatten_studies
from trialsight.quality import run_quality_checks

CFG = CohortConfig(
    start_date_min="2008-01-01", start_date_max="2017-12-31",
    phase_groups=["PHASE2", "PHASE2_3", "PHASE3"],
    target_statuses={"COMPLETED": 0, "TERMINATED": 1},
    require_oncology_mesh=True, oncology_mesh_terms=["Neoplasms"],
)


def _flagged(summary, rule_id):
    return int(summary.loc[summary.rule_id == rule_id, "n_flagged"].iloc[0])


def test_quality_rules(synthetic_studies):
    tables = flatten_studies(synthetic_studies)
    summary, violations = run_quality_checks(tables["studies"])
    assert _flagged(summary, "DQ01") == 2          # both copies of the duplicate
    assert _flagged(summary, "DQ02") == 1          # 'NCT123'
    assert _flagged(summary, "DQ05") == 1          # '2013-13'
    assert _flagged(summary, "DQ14") == 1          # registered after start
    assert _flagged(summary, "DQ15") == 1          # NCT99990007 terminated w/o reason
    assert _flagged(summary, "DQ16") == len(synthetic_studies)  # fixture: min 18y > max 6mo
    assert set(violations.columns) == {"nct_id", "rule_id"}


def test_flag_oncology(synthetic_studies):
    tables = flatten_studies(synthetic_studies)
    flags = flag_oncology(tables["studies"], tables["mesh"], ["neoplasms"])  # case-insensitive
    by_id = dict(zip(tables["studies"]["nct_id"], flags))
    assert by_id["NCT99990005"] is False or by_id["NCT99990005"] == False  # noqa: E712
    assert by_id["NCT99990002"]


def test_cohort_attrition_and_target(synthetic_studies):
    tables = flatten_studies(synthetic_studies)
    result = build_cohort(tables["studies"], tables["mesh"], CFG)
    cohort, attrition, status_by_year = result.cohort, result.attrition, result.status_by_start_year
    assert result.analysis_end_year is None and result.resolution_by_start_year.empty  # no rule configured
    assert attrition["n_remaining"].tolist() == [11, 8, 8, 7, 5, 4, 2]
    assert (attrition["n_excluded"].iloc[1:] == -attrition["n_remaining"].diff().iloc[1:]).all()
    assert sorted(cohort["nct_id"]) == ["NCT99990002", "NCT99990009"]
    assert dict(zip(cohort["nct_id"], cohort["target_terminated"])) == {"NCT99990002": 1, "NCT99990009": 0}
    # pre-outcome population (4 studies) is tabulated by start year, incl. non-terminal statuses
    assert int(status_by_year.drop(columns="start_year").to_numpy().sum()) == 4
    assert {"WITHDRAWN", "RECRUITING"} <= set(status_by_year.columns)
