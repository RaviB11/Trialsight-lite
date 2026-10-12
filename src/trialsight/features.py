"""Feature sets for analysis and modelling, derived strictly from the feature registry.

Prediction-time assumption: information plausibly known at or near study start.
Because only current record versions exist, every model measures RETROSPECTIVE
association, not prospective performance.
"""

from __future__ import annotations

import pandas as pd

from .feature_registry import classify

INTERVENTION_TYPES = ["DRUG", "BIOLOGICAL", "RADIATION", "PROCEDURE", "DEVICE", "BEHAVIORAL",
                      "DIETARY_SUPPLEMENT", "GENETIC", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER"]
AGE_GROUPS = ["CHILD", "ADULT", "OLDER_ADULT"]

PRIMARY_CATEGORICAL = ["phase_group", "allocation", "intervention_model", "primary_purpose", "masking",
                       "lead_sponsor_class", "sex", "healthy_volunteers", "has_dmc",
                       "is_fda_regulated_drug", "is_fda_regulated_device"]
PRIMARY_NUMERIC = ["n_masked_roles", "n_conditions", "n_arms", "n_interventions", "n_primary_outcomes",
                   "min_age_years", "max_age_years", "eligibility_criteria_chars", "registration_lag_days"]
DERIVED_FLAGS = [f"itype_{t}" for t in INTERVENTION_TYPES] + [f"age_{a}" for a in AGE_GROUPS]
SUSPECT_CATEGORICAL = ["has_us_site"]
SUSPECT_NUMERIC = ["n_locations", "n_countries", "n_collaborators", "n_secondary_outcomes"]
CENSORING_PROXY = ["start_year"]

ALLOWED_CLASSES = {"baseline_candidate"}
FORBIDDEN_CLASSES = {"target", "post_outcome", "identifier", "metadata", "unclassified"}
TARGET = "target_terminated"


class LeakageError(ValueError):
    """A feature set contains information not available at the prediction point."""


def source_column(feature: str) -> str:
    if feature.startswith("itype_"):
        return "intervention_types"
    if feature.startswith("age_"):
        return "std_ages"
    return feature


def availability(feature: str) -> str:
    return classify(source_column(feature))


def feature_set(name: str) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) columns for a named feature set."""
    cat, num = list(PRIMARY_CATEGORICAL), list(PRIMARY_NUMERIC) + list(DERIVED_FLAGS)
    if name == "primary":
        pass
    elif name == "with_time_varying":
        cat, num = cat + SUSPECT_CATEGORICAL, num + SUSPECT_NUMERIC
    elif name == "with_start_year":
        num = num + CENSORING_PROXY
    else:
        raise ValueError(f"Unknown feature set {name!r}")
    assert_no_leakage(cat + num, allow_time_varying=(name == "with_time_varying"))
    return cat, num


def assert_no_leakage(features: list[str], allow_time_varying: bool = False) -> None:
    allowed = ALLOWED_CLASSES | ({"time_varying_suspect"} if allow_time_varying else set())
    bad = {f: availability(f) for f in features if availability(f) not in allowed}
    if TARGET in features:
        bad[TARGET] = "target"
    if bad:
        raise LeakageError(f"Features not available at the prediction point (or not allowed here): {bad}")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicator columns from pipe-joined intervention types and standard age groups."""
    out = df.copy()
    itypes = out["intervention_types"].fillna("").astype(str).str.split("|")
    for t in INTERVENTION_TYPES:
        out[f"itype_{t}"] = itypes.apply(lambda xs, t=t: int(t in xs))
    ages = out["std_ages"].fillna("").astype(str).str.split("|")
    for a in AGE_GROUPS:
        out[f"age_{a}"] = ages.apply(lambda xs, a=a: int(a in xs))
    return out


def as_categorical_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Booleans/None -> stable string levels so missingness is an explicit category."""
    out = df.copy()
    for c in columns:
        out[c] = out[c].astype(object).where(out[c].notna(), "(missing)").astype(str)
    return out
