"""SYNTHETIC cohort generator for testing analysis/modelling MACHINERY only.
Values are random with an injected signal so models have something to fit.
Nothing produced from this data is a research result."""
import datetime as dt

import numpy as np
import pandas as pd

from trialsight.storage import write_tables


def synthetic_cohort(n=600, seed=7):
    rng = np.random.default_rng(seed)
    phase = rng.choice(["PHASE2", "PHASE3", "PHASE2_3"], n, p=[0.6, 0.3, 0.1])
    sponsor = rng.choice(["INDUSTRY", "OTHER", "NIH", "OTHER_GOV"], n, p=[0.4, 0.45, 0.1, 0.05])
    alloc = rng.choice(["RANDOMIZED", "NA", "NON_RANDOMIZED", None], n, p=[0.5, 0.35, 0.1, 0.05])
    start = pd.to_datetime("2008-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D")
    n_arms = rng.integers(1, 5, n)
    crit = rng.integers(200, 6000, n).astype(float)
    crit[rng.random(n) < 0.05] = np.nan
    logit = -1.4 + 0.8 * (sponsor == "INDUSTRY") + 0.5 * (phase == "PHASE3") - 0.15 * n_arms + rng.normal(0, 0.7, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "nct_id": [f"NCT9{i:07d}" for i in range(n)], "brief_title": [f"Synthetic study {i}" for i in range(n)],
        "overall_status": np.where(y == 1, "TERMINATED", "COMPLETED"), "target_terminated": y,
        "start_date": start, "start_year": start.year, "phase_group": phase, "phases": phase,
        "allocation": alloc, "intervention_model": rng.choice(["PARALLEL", "SINGLE_GROUP"], n),
        "primary_purpose": rng.choice(["TREATMENT", "SUPPORTIVE_CARE", "PREVENTION"], n, p=[0.8, 0.15, 0.05]),
        "masking": rng.choice(["NONE", "DOUBLE", "QUADRUPLE"], n), "lead_sponsor_class": sponsor,
        "sex": rng.choice(["ALL", "FEMALE", "MALE"], n, p=[0.8, 0.15, 0.05]),
        "healthy_volunteers": rng.choice([False, True], n, p=[0.95, 0.05]),
        "has_dmc": rng.choice([True, False, None], n), "is_fda_regulated_drug": rng.choice([True, False, None], n),
        "is_fda_regulated_device": rng.choice([False, None], n), "n_masked_roles": rng.integers(0, 4, n),
        "n_conditions": rng.integers(1, 6, n), "n_arms": n_arms, "n_interventions": rng.integers(1, 5, n),
        "n_primary_outcomes": rng.integers(1, 4, n), "min_age_years": rng.choice([18.0, 12.0, np.nan], n, p=[0.85, 0.1, 0.05]),
        "max_age_years": rng.choice([np.nan, 75.0, 99.0], n), "eligibility_criteria_chars": crit,
        "registration_lag_days": rng.integers(-60, 400, n), "intervention_types": rng.choice(
            ["DRUG", "DRUG|OTHER", "BIOLOGICAL", "RADIATION|DRUG", "PROCEDURE"], n),
        "std_ages": rng.choice(["ADULT|OLDER_ADULT", "ADULT", "CHILD|ADULT|OLDER_ADULT"], n),
        "has_us_site": rng.choice([True, False], n), "n_locations": rng.integers(1, 80, n),
        "n_countries": rng.integers(1, 10, n), "n_collaborators": rng.integers(0, 3, n),
        "n_secondary_outcomes": rng.integers(0, 10, n), "start_date_type": "ACTUAL",
    })
    lag = df["registration_lag_days"].to_numpy()
    df["first_submit_date"] = df["start_date"] + pd.to_timedelta(lag, unit="D")
    df["primary_completion_date"] = df["start_date"] + pd.to_timedelta(rng.integers(100, 1500, n), unit="D")
    df["completion_date"] = df["primary_completion_date"] + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    return df


def write_synthetic_research_db(path, n=600, seed=7, output_state="accepted", status="research_verified"):
    cohort = synthetic_cohort(n, seed)
    terms = ["Lung Neoplasms", "Breast Neoplasms", "Leukemia", "Lymphoma"]
    mesh = pd.DataFrame({"nct_id": cohort["nct_id"], "kind": "ancestor", "mesh_id": "x",
                         "term": [terms[i % 4] for i in range(n)]})
    mesh = pd.concat([mesh, mesh.assign(term="Neoplasms")], ignore_index=True)
    conditions = pd.DataFrame({"nct_id": cohort["nct_id"], "kind": "condition",
                               "term": [terms[i % 4].replace("Neoplasms", "Cancer") for i in range(n)]})
    meta = pd.DataFrame([{"population_status": status, "cohort_validated": True, "output_state": output_state,
                          "raw_run_id": "SYNTHETIC-TEST-DATA", "built_at_utc": dt.datetime(2026, 10, 12).isoformat(),
                          "config_sha256": "synthetic", "analysis_start_year": 2008, "analysis_end_year": 2013}])
    attrition = pd.DataFrame({"step": [0, 1], "criterion": ["synthetic", "synthetic"], "n_remaining": [n, n],
                              "n_excluded": [0, 0]})
    write_tables(path, {"cohort": cohort, "mesh": mesh, "conditions": conditions, "build_metadata": meta,
                        "eligible_population": cohort, "cohort_attrition": attrition,
                        "cohort_validation": pd.DataFrame({"check_id": ["V01"], "severity": ["critical"],
                                                           "description": ["synthetic"], "status": ["pass"],
                                                           "detail": [""]})})
    return cohort
