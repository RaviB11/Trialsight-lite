import datetime as dt

import pandas as pd
import pytest

from trialsight.flatten import flatten_studies, flatten_study, parse_age_years, parse_ct_date, phase_group


@pytest.mark.parametrize("raw,expected", [
    ("2015-03-12", (dt.date(2015, 3, 12), "day")),
    ("2015-03", (dt.date(2015, 3, 1), "month")),
    ("2015", (dt.date(2015, 1, 1), "year")),
    ("2013-13", (None, "invalid")),
    ("March 2015", (None, "invalid")),
    ("", (None, None)),
    (None, (None, None)),
])
def test_parse_ct_date(raw, expected):
    assert parse_ct_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("18 Years", 18.0), ("6 Months", 0.5), ("1 Year", 1.0), ("N/A", None), (None, None),
])
def test_parse_age_years(raw, expected):
    assert parse_age_years(raw) == expected


def test_phase_group():
    assert phase_group(["PHASE3", "PHASE2"]) == "PHASE2_3"
    assert phase_group(["PHASE2"]) == "PHASE2"
    assert phase_group([]) == "NONE"
    assert phase_group(["PHASE1", "PHASE2"]) == "PHASE1|PHASE2"


def test_flatten_core_fields(synthetic_studies):
    row = flatten_study(synthetic_studies[0])["studies"]
    assert row["nct_id"] == "NCT99990001"
    assert row["overall_status"] == "COMPLETED"
    assert row["phase_group"] == "PHASE3"
    assert row["n_masked_roles"] == 2
    assert row["n_countries"] == 2 and row["has_us_site"] is True
    assert row["n_arms"] == 2 and row["n_interventions"] == 2
    assert row["intervention_types"] == "DRUG|OTHER"
    assert row["start_date"] == dt.date(2011, 3, 15)
    assert row["registration_lag_days"] == 0


def test_month_precision_and_registration_lag(synthetic_studies):
    row = flatten_study(synthetic_studies[1])["studies"]
    assert row["start_date_precision"] == "month"
    assert row["start_date"] == dt.date(2012, 5, 1)
    assert row["registration_lag_days"] == (dt.date(2012, 8, 1) - dt.date(2012, 5, 1)).days
    assert row["why_stopped"] == "Slow accrual"


def test_missing_modules_do_not_crash():
    flat = flatten_study({"protocolSection": {"identificationModule": {"nctId": "NCT00000000"}}})
    assert flat["studies"]["nct_id"] == "NCT00000000"
    assert flat["studies"]["phase_group"] == "NONE"
    assert flat["studies"]["n_locations"] == 0


def test_no_contact_person_fields_extracted(synthetic_studies):
    tables = flatten_studies(synthetic_studies)
    all_cols = {c for df in tables.values() for c in df.columns}
    assert not any(k in all_cols for k in ("email", "phone", "contact_name", "official_name"))


def test_flatten_tables_shapes(synthetic_studies):
    tables = flatten_studies(synthetic_studies)
    assert len(tables["studies"]) == len(synthetic_studies)
    assert pd.api.types.is_datetime64_any_dtype(tables["studies"]["start_date"])
    mesh = tables["mesh"]
    assert set(mesh["kind"]) == {"mesh", "ancestor", "branch"}
    assert (tables["sponsors"]["role"] == "lead").sum() == len(synthetic_studies)


def test_empty_input_keeps_child_columns():
    tables = flatten_studies([])
    assert list(tables["mesh"].columns) == ["nct_id", "kind", "mesh_id", "term"]
