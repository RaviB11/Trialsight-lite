# Data Dictionary (draft - structure only; value distributions added after first real build)

Source: ClinicalTrials.gov API v2 `/studies`. `PS` = `protocolSection`.
Availability classes come from `src/trialsight/feature_registry.py`.

## `studies` (one row per record)
| Column | Source path | Transformation | Availability |
|---|---|---|---|
| nct_id | PS.identificationModule.nctId | none | identifier |
| overall_status | PS.statusModule.overallStatus | none | target |
| why_stopped | PS.statusModule.whyStopped | none | post_outcome |
| study_type | PS.designModule.studyType | none | metadata |
| phases / phase_group | PS.designModule.phases | sorted, pipe-joined; grouped to PHASE2 / PHASE3 / PHASE2_3 | baseline_candidate |
| allocation, intervention_model, primary_purpose | PS.designModule.designInfo.* | none | baseline_candidate |
| masking, n_masked_roles | ...designInfo.maskingInfo.masking / whoMasked | count of roles | baseline_candidate |
| enrollment_count, enrollment_type | PS.designModule.enrollmentInfo.count / type | none | post_outcome |
| lead_sponsor_name / _class | PS.sponsorCollaboratorsModule.leadSponsor | none | identifier / baseline_candidate |
| n_collaborators | ...collaborators | count | time_varying_suspect |
| has_dmc, is_fda_regulated_drug/_device | PS.oversightModule.* | none | baseline_candidate |
| n_conditions | PS.conditionsModule.conditions | count | baseline_candidate |
| n_arms, n_interventions, intervention_types | PS.armsInterventionsModule.* | counts; distinct types pipe-joined | baseline_candidate |
| n_primary_outcomes / n_secondary_outcomes | PS.outcomesModule.* | counts | baseline_candidate / time_varying_suspect |
| sex, healthy_volunteers, std_ages | PS.eligibilityModule.* | none / pipe-joined | baseline_candidate |
| min_age_years, max_age_years | ...minimumAge / maximumAge | "N Unit" converted to years | baseline_candidate |
| eligibility_criteria_chars | ...eligibilityCriteria | character length | baseline_candidate |
| n_locations, n_countries, has_us_site | PS.contactsLocationsModule.locations | counts; "United States" membership | time_varying_suspect |
| has_results | hasResults | none | post_outcome |
| {start, primary_completion, completion, first_post, last_update_post}_date | PS.statusModule.<x>DateStruct.date | parsed; partial dates anchored to day 1 | see registry |
| ..._date_precision / _raw / _type | same struct | day/month/year/invalid; original string; ACTUAL/ESTIMATED | metadata |
| first_submit_date | PS.statusModule.studyFirstSubmitDate | parsed | metadata |
| start_year | start_date | year | baseline_candidate (also a censoring proxy) |
| registration_lag_days | first_submit_date - start_date | days | baseline_candidate |

## Child tables
`conditions` (condition and keyword terms), `mesh` (derivedSection.conditionBrowseModule meshes, ancestors, browse branches), `interventions`, `sponsors` (lead + collaborators), `locations` (facility, city, state, country, site status). **Not extracted:** central contacts, overall officials, site contacts.

## Cohort tables
- `eligible_population` = `studies` passing blocking DQ rules, interventional, allowed phase group, start date 2008-01-01..2025-12-31, oncology MeSH; **all statuses**. Reserved for survival analysis.
- `resolution_by_start_year` = per start year: `n_total`, `n_resolved`, `n_unresolved`, `n_ambiguous` (UNKNOWN), `n_other`, `resolved_share` = resolved / (resolved + unresolved), `ambiguous_share` = ambiguous / total, `meets_threshold`, `in_analysis_window`.
- `cohort` = `eligible_population` restricted to start years within the data-selected window, then to target statuses, plus `target_terminated` (COMPLETED=0, TERMINATED=1).
- `cohort_attrition` = each filtering step with counts remaining and excluded.

## Diagnostic and metadata tables (v2)
| Table | Grain | Purpose |
|---|---|---|
| `build_metadata` | 1 row | population_status, cohort_validated, raw run id, verification status/JSON, exact duplicates collapsed, analysis window, config SHA-256, version, build time |
| `cohort_validation` | 1 row per check | V01-V09 with severity (critical/warning) and pass/fail/warn/not_checked |
| `resolution_by_start_year` | start year | adds `n_denominator`, Wilson `resolved_share_ci_low/high`, `decision_reason` (meets_threshold / below_threshold / insufficient_n / no_resolvable_studies / no_studies) |
| `threshold_sensitivity` | threshold x ambiguous handling | end year, years covered, cohort n, completed/terminated, terminated share + Wilson CI, in-window excluded counts |
| `terminated_share_by_year` | start year | completed, terminated, share, Wilson CI (all eligible years) |
| `selection_profile` | column x value x group | distribution within window for cohort / unresolved / ambiguous / other |
| `oncology_capture_summary` | category | pre-oncology records by MeSH match x condition/keyword cancer text |
| `oncology_review_sample` | record | excluded records with cancer text, for manual review (default first 200 by nct_id) |
| `oncology_text_only_mesh_terms` | MeSH term | most common MeSH terms where cancer text exists but Neoplasms MeSH does not |
| `phase_audit` | raw phase combination | counts, phase_group, included_by_definition, note for API-returned-but-excluded |
| `date_audit` | check | missing/invalid/out-of-window start dates, precision, ESTIMATED types |
| `start_type_by_status` | status x start type | eligible population |
| `enum_check` | config token | present in the API enums snapshot (`docs/api_snapshot/enums.json`) |

Cancer text = whole-word, case-insensitive match against `DEFAULT_CANCER_TERMS`
in `src/trialsight/config.py`, applied to condition and keyword terms only.

## Data-quality rules
See `src/trialsight/quality.py` (DQ01-DQ16) and the generated `dq_summary` table.

## Types and missing-value interpretation (`studies` / `cohort`)
| Field group | DuckDB type | Missing means |
|---|---|---|
| nct_id, brief_title, status, phase, design, sponsor class, sex | VARCHAR | not reported in the record |
| has_dmc, is_fda_regulated_drug/_device, healthy_volunteers, has_us_site, has_results | BOOLEAN | not reported (FDA fields were introduced in 2017, so older records are often missing) |
| counts (n_arms, n_interventions, n_conditions, n_*_outcomes, n_locations, n_countries, n_collaborators, n_masked_roles) | BIGINT | never missing; 0 means none listed |
| min_age_years, max_age_years | DOUBLE | no limit stated, or unparseable text |
| eligibility_criteria_chars | DOUBLE | criteria text absent |
| *_date | TIMESTAMP | not reported; see *_date_precision (day/month/year/invalid) |
| registration_lag_days | DOUBLE | start or first-submit date missing |
| enrollment_count | DOUBLE | not reported (post-outcome; never a feature) |
| target_terminated | INTEGER (nullable) | not in the cohort |

## Model feature encodings (`features.py`)
- `itype_<TYPE>` = 1 if the intervention type appears in `intervention_types` (0 if none listed).
- `age_<GROUP>` = 1 if the group appears in `std_ages`.
- Categorical missing values become the explicit level `(missing)`; numeric missing values are
  median-imputed with a missing-indicator column.

## Analysis artifacts
`artifacts/eda/`: eda_associations (all tests, effect sizes, p and BH q), eda_levels (rates with Wilson CIs),
eda_therapeutic_areas, eda_top_conditions, eda_by_start_year, eda_metadata.json, eda_summary.md,
eda_figures.html.
`artifacts/models/`: metrics_test, bootstrap, subgroups, calibration_bins, curves, logistic_coefficients,
permutation_importance, sensitivity, predictions_test (test set only), experiment.json, models/*.joblib,
evaluation_summary.md.
