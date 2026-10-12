"""Issues 4-6: threshold sensitivity, oncology capture, phase/date/enum audits. Synthetic data only."""
import json

import pandas as pd
import pytest

from trialsight import diagnostics as dx
from trialsight.config import DEFAULT_CANCER_TERMS, CohortConfig, ResolutionRule

CFG = CohortConfig(
    start_date_min="2008-01-01", start_date_max="2011-12-31",
    phase_groups=["PHASE2", "PHASE2_3", "PHASE3"], target_statuses={"COMPLETED": 0, "TERMINATED": 1},
    resolution_rule=ResolutionRule(min_resolved_share=0.90, missing_year_policy="break", min_studies_per_year=1),
)


def _eligible():
    """Per year 100 studies; resolved share 95%, 92%, 85%, 70%; terminated = 20 of resolved; 5 UNKNOWN extra."""
    rows = []
    for year, resolved in [(2008, 95), (2009, 92), (2010, 85), (2011, 70)]:
        rows += [(year, "TERMINATED")] * 20 + [(year, "COMPLETED")] * (resolved - 20)
        rows += [(year, "RECRUITING")] * (100 - resolved) + [(year, "UNKNOWN")] * 5
    df = pd.DataFrame(rows, columns=["start_year", "overall_status"])
    df["phase_group"] = "PHASE2"
    df["lead_sponsor_class"] = ["INDUSTRY" if i % 2 else "OTHER" for i in range(len(df))]
    return df


def test_threshold_sensitivity_end_years_and_monotone_cohort():
    t = dx.threshold_sensitivity(_eligible(), CFG, [0.80, 0.85, 0.90, 0.95])
    ex = t[t.ambiguous_handling == "exclude"].set_index("threshold")
    assert ex.loc[0.80, "end_year"] == 2010 and ex.loc[0.90, "end_year"] == 2009 and ex.loc[0.95, "end_year"] == 2008
    assert ex.loc[0.90, "cohort_n"] == 95 + 92 and ex.loc[0.90, "n_terminated"] == 40
    assert list(ex["cohort_n"]) == sorted(ex["cohort_n"], reverse=True)   # stricter -> never larger
    assert t["is_configured_rule"].sum() == 1
    assert ex.loc[0.90, "terminated_ci_low"] < ex.loc[0.90, "terminated_share"] < ex.loc[0.90, "terminated_ci_high"]
    assert ex.loc[0.90, "in_window_unresolved"] == 8 + 5 and ex.loc[0.90, "in_window_ambiguous"] == 10


def test_threshold_sensitivity_strict_ambiguous_shrinks_window():
    t = dx.threshold_sensitivity(_eligible(), CFG, [0.90])
    strict = t[(t.ambiguous_handling == "count_as_unresolved") & (t.threshold == 0.90)].iloc[0]
    assert strict.end_year == 2008          # 2009: 92/105 = 0.876 < 0.90 once UNKNOWN counts


def test_terminated_share_by_year_and_selection_profile():
    tsy = dx.terminated_share_by_year(_eligible(), CFG.target_statuses).set_index("start_year")
    assert tsy.loc[2011, "terminated_share"] == round(20 / 70, 4)
    prof = dx.selection_profile(_eligible(), 2009, CFG, ["lead_sponsor_class", "not_a_column"])
    assert set(prof["group"]) == {"cohort", "unresolved", "ambiguous"}
    assert set(prof["column"]) == {"lead_sponsor_class"}
    sums = prof.groupby("group")["share_within_group"].sum()
    assert all(abs(s - 1) < 1e-3 for s in sums)


def _onc_frames():
    pre = pd.DataFrame({
        "nct_id": ["A", "B", "C", "D", "E", "F", "G"],
        "brief_title": ["t"] * 7,
        "overall_status": ["COMPLETED", "TERMINATED", "COMPLETED", "TERMINATED", "COMPLETED", "COMPLETED", "COMPLETED"],
        "start_year": [2010] * 7,
        "is_oncology_mesh": [True, True, False, False, False, False, False],
    })
    cond = pd.DataFrame({"nct_id": ["A", "B", "C", "D", "E", "F", "G"], "kind": ["condition"] * 7,
                         "term": ["Breast Cancer", "Polyps", "Non-Small Cell Lung Carcinoma",
                                  "Non-Hodgkin Lymphoma", "Diabetes", "Healthy", "Precancerous Lesion"]})
    mesh = pd.DataFrame({"nct_id": ["A", "B", "D", "E", "G"], "kind": ["ancestor", "ancestor", "mesh", "mesh", "mesh"],
                         "mesh_id": ["x"] * 5, "term": ["Neoplasms", "Neoplasms", "Some Disease", "Diabetes Mellitus",
                                                        "Lesion"]})
    return pre, cond, mesh


def test_oncology_capture_categories():
    pre, cond, mesh = _onc_frames()
    summary, review, top = dx.oncology_capture(pre, cond, mesh, DEFAULT_CANCER_TERMS)
    s = summary.set_index("category")["n"].to_dict()
    assert s == {
        "included: Neoplasms MeSH + cancer text": 1,          # A
        "included: Neoplasms MeSH, no cancer text": 1,         # B
        "excluded: cancer text, no MeSH annotations": 1,       # C  <- the key measurement
        "excluded: cancer text, MeSH present but not Neoplasms": 1,  # D
        "excluded: no cancer text, non-neoplasm MeSH only": 2,  # E, G ("Precancerous" is not a word match)
        "excluded: no cancer text, no MeSH annotations": 1,     # F
    }
    assert review["nct_id"].tolist() == ["C", "D"]
    assert review.set_index("nct_id").loc["D", "matched_terms"] == "lymphoma"
    assert top["mesh_term"].tolist() == ["Some Disease"]
    assert summary["n"].sum() == len(pre)


def test_oncology_capture_empty():
    pre, cond, mesh = _onc_frames()
    summary, review, top = dx.oncology_capture(pre.iloc[0:0], cond, mesh, ["cancer"])
    assert summary.empty and list(review.columns)[0] == "nct_id"


def _studies():
    return pd.DataFrame({
        "nct_id": ["A", "B", "C", "D"],
        "phases": ["PHASE2", "PHASE1|PHASE2", "PHASE2|PHASE3", None],
        "phase_group": ["PHASE2", "PHASE1|PHASE2", "PHASE2_3", "NONE"],
        "study_type": ["INTERVENTIONAL", "INTERVENTIONAL", "INTERVENTIONAL", "OBSERVATIONAL"],
        "start_date": pd.to_datetime(["2010-01-01", "2026-01-01", "2011-05-01", None]),  # CFG window 2008-2011
        "start_date_precision": ["day", "month", "month", None],
        "start_date_type": ["ACTUAL", "ESTIMATED", "ESTIMATED", None],
        "completion_date_type": ["ACTUAL", "ESTIMATED", "ESTIMATED", None],
        "overall_status": ["COMPLETED", "NOT_YET_RECRUITING", "TERMINATED", "COMPLETED"],
    })


def test_phase_audit_flags_api_returned_but_locally_excluded():
    pa = dx.phase_audit(_studies(), CFG).set_index("phases")
    assert bool(pa.loc["PHASE2", "included_by_definition"]) and bool(pa.loc["PHASE2|PHASE3", "included_by_definition"])
    assert not bool(pa.loc["PHASE1|PHASE2", "included_by_definition"])
    assert "excluded by the local Phase II/III definition" in pa.loc["PHASE1|PHASE2", "note"]
    assert pa.loc["(none)", "n_interventional"] == 0


def test_date_audit_detects_filter_and_type_problems():
    s = _studies()
    eligible = s.iloc[[0, 2]]
    cohort = s.iloc[[0, 2]]
    da = dx.date_audit(s, eligible, cohort, CFG, pd.Timestamp("2026-10-12T00:00:00+00:00")).set_index("check")
    assert da.loc["start date outside configured window (API RANGE filter not honoured?)", "n"] == 1  # B (2026)
    assert da.loc["study type not INTERVENTIONAL (API filter not honoured?)", "n"] == 1                # D
    assert da.loc["start date missing", "n"] == 1
    assert da.loc["start date ESTIMATED but before retrieval date", "n"] == 1                          # C
    assert da.loc["start date type ESTIMATED among COMPLETED/TERMINATED", "n"] == 1


def test_enum_check(tmp_path):
    good = {"_note": "SYNTHETIC enums snapshot for tests", "enums": [
        {"type": "Status", "values": [{"value": v} for v in
         ["COMPLETED", "TERMINATED", "WITHDRAWN", "NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION",
          "ACTIVE_NOT_RECRUITING", "SUSPENDED", "UNKNOWN"]]},
        {"type": "Phase", "values": [{"value": "PHASE2"}, {"value": "PHASE3"}]},
        {"type": "Misc", "values": ["INTERVENTIONAL", "ACTUAL", "ESTIMATED"]}]}
    p = tmp_path / "enums.json"
    p.write_text(json.dumps(good))
    status, table = dx.enum_check(CFG, p)
    assert status == "pass" and table["present_in_api_enums"].all()
    good["enums"][1]["values"] = [{"value": "PHASE_2"}]
    p.write_text(json.dumps(good))
    status, _ = dx.enum_check(CFG, p)
    assert status.startswith("fail") and "PHASE2" in status
    assert dx.enum_check(CFG, tmp_path / "missing.json")[0].startswith("not_checked")
    p.write_text("{}")
    assert dx.enum_check(CFG, p)[0].startswith("not_checked")
