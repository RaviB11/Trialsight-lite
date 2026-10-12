"""Information-availability registry for every column in the `studies` table.

Prediction-time assumption (to be confirmed): the model represents information
plausibly known AT OR NEAR STUDY START. The API exposes only the CURRENT version
of each record, so even "baseline" fields may have been amended later. Classes:

- target              : outcome label source; never a feature.
- post_outcome        : known only after the outcome; always excluded (leakage).
- time_varying_suspect: design-time in principle, but plausibly updated in ways
                        correlated with outcome; used only in ablation /
                        sensitivity analyses, never in the headline model.
- baseline_candidate  : plausibly fixed at registration; eligible features.
- identifier          : IDs / free text / high-cardinality names; excluded.
- metadata            : provenance or parse-quality columns; excluded.
"""

from __future__ import annotations

FEATURE_CLASSES: dict[str, tuple[str, str]] = {
    "overall_status": ("target", "Source of the completed/terminated label."),
    "target_terminated": ("target", "Binary label derived from overall_status."),
    "why_stopped": ("post_outcome", "Termination explanation; exists because of the outcome."),
    "last_known_status": ("post_outcome", "Status bookkeeping after verification lapses."),
    "enrollment_count": ("post_outcome", "For terminal studies this is usually ACTUAL enrollment, which "
                         "mechanically reflects early stopping. Anticipated enrollment is overwritten."),
    "enrollment_type": ("post_outcome", "ACTUAL vs ESTIMATED reflects record lifecycle, not design."),
    "has_results": ("post_outcome", "Results posting happens after completion/termination."),
    "primary_completion_date": ("post_outcome", "Actual date for terminal studies."),
    "completion_date": ("post_outcome", "Actual date for terminal studies."),
    "primary_completion_date_type": ("post_outcome", "Lifecycle indicator."),
    "completion_date_type": ("post_outcome", "Lifecycle indicator."),
    "last_update_post_date": ("post_outcome", "Record maintenance after the outcome."),
    "status_verified_date_raw": ("post_outcome", "Record maintenance after the outcome."),
    "n_locations": ("time_varying_suspect", "Sites are added/removed over time; terminated studies may "
                    "never open planned sites."),
    "n_countries": ("time_varying_suspect", "Derived from locations."),
    "has_us_site": ("time_varying_suspect", "Derived from locations."),
    "n_collaborators": ("time_varying_suspect", "Collaborators can be added during conduct."),
    "n_secondary_outcomes": ("time_varying_suspect", "Outcome lists are often amended."),
    "phase_group": ("baseline_candidate", "Declared design phase."),
    "phases": ("baseline_candidate", "Raw phase list (use phase_group in models)."),
    "allocation": ("baseline_candidate", "Randomised vs non-randomised."),
    "intervention_model": ("baseline_candidate", "Parallel, single-group, crossover, etc."),
    "primary_purpose": ("baseline_candidate", "Treatment, supportive care, etc."),
    "masking": ("baseline_candidate", "Blinding level."),
    "n_masked_roles": ("baseline_candidate", "Count of masked roles."),
    "lead_sponsor_class": ("baseline_candidate", "INDUSTRY, NIH, OTHER, etc."),
    "has_dmc": ("baseline_candidate", "Oversight declared; usually set at registration (verify)."),
    "is_fda_regulated_drug": ("baseline_candidate", "Field introduced with the 2017 Final Rule; "
                              "high missingness for older records expected."),
    "is_fda_regulated_device": ("baseline_candidate", "As above."),
    "n_conditions": ("baseline_candidate", "Breadth of condition list."),
    "n_arms": ("baseline_candidate", "Amendable, but design-defining."),
    "n_interventions": ("baseline_candidate", "Amendable, but design-defining."),
    "intervention_types": ("baseline_candidate", "DRUG, BIOLOGICAL, RADIATION, etc."),
    "n_primary_outcomes": ("baseline_candidate", "Amendable; include in sensitivity check."),
    "healthy_volunteers": ("baseline_candidate", "Eligibility flag."),
    "sex": ("baseline_candidate", "Eligibility sex."),
    "min_age_years": ("baseline_candidate", "Parsed minimum age."),
    "max_age_years": ("baseline_candidate", "Parsed maximum age."),
    "std_ages": ("baseline_candidate", "CHILD / ADULT / OLDER_ADULT."),
    "eligibility_criteria_chars": ("baseline_candidate", "Length of criteria text (amendable)."),
    "start_year": ("baseline_candidate", "Calendar context; ALSO a censoring proxy - interpret carefully."),
    "registration_lag_days": ("baseline_candidate", "Registration timing relative to start."),
    "is_oncology_mesh": ("metadata", "Cohort filter, constant within cohort."),
    "study_type": ("metadata", "Cohort filter, constant within cohort."),
    "start_date": ("metadata", "Used for splits/filters; use start_year as feature."),
    "first_submit_date": ("metadata", "Used to derive registration_lag_days."),
    "first_post_date": ("metadata", "Registry posting date."),
    "nct_id": ("identifier", "Study identifier."),
    "brief_title": ("identifier", "Free text (possible future NLP features, with care)."),
    "lead_sponsor_name": ("identifier", "High cardinality; risk of memorising sponsors."),
}


def columns_in(cls: str) -> list[str]:
    return [c for c, (k, _) in FEATURE_CLASSES.items() if k == cls]


def classify(column: str) -> str:
    if column in FEATURE_CLASSES:
        return FEATURE_CLASSES[column][0]
    if column.endswith(("_precision", "_raw", "_type")) or column.endswith("_date"):
        return "metadata"
    return "unclassified"
