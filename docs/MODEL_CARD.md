# Model card

_Generated from saved artifacts. Source: raw run `20261012T020557Z`, build 2026-10-12T02:11:44.836572+00:00, cohort fingerprint `2b8ddab1bf698355`._

## Purpose
Describe retrospective associations between registry characteristics and whether a Phase II/III oncology trial was completed or terminated. Positive class: TERMINATED.

## Training population
Train n=4893 (1109 terminated), test n=1648 (370 terminated); temporal cutoff 2013-01-01. Inclusion criteria: METHODOLOGY.md.

## Features
Categorical: phase_group, allocation, intervention_model, primary_purpose, masking, lead_sponsor_class, sex, healthy_volunteers, has_dmc, is_fda_regulated_drug, is_fda_regulated_device.

Numeric: n_masked_roles, n_conditions, n_arms, n_interventions, n_primary_outcomes, min_age_years, max_age_years, eligibility_criteria_chars, registration_lag_days, itype_DRUG, itype_BIOLOGICAL, itype_RADIATION, itype_PROCEDURE, itype_DEVICE, itype_BEHAVIORAL, itype_DIETARY_SUPPLEMENT, itype_GENETIC, itype_COMBINATION_PRODUCT, itype_DIAGNOSTIC_TEST, itype_OTHER, age_CHILD, age_ADULT, age_OLDER_ADULT.

Excluded: post-outcome fields (status, why_stopped, enrollment, completion dates, results), identifiers, time-varying-suspect fields (sensitivity only), start_year.

## Performance (test set)

- Discrimination is modest. The highest test ROC-AUC is 0.6017 (hist_gradient_boosting; 95% CI 0.568-0.635); learned models range 0.5879-0.6017. PR-AUC is 0.2819-0.2918 against a prevalence of 0.2245 (lift 1.26-1.3x).
- Every learned model's ROC-AUC and PR-AUC improvement over the prevalence baseline has a paired 95% CI above zero, but the improvements are small and the learned models' intervals overlap heavily.
- Calibrated models (slope 0.8-1.2) whose Brier score beats the baseline: logistic, random_forest. Poorly calibrated: lda, hist_gradient_boosting. A higher ROC-AUC without calibration is not a better model for probability estimates.
- False positives are high. At the training-derived max-F1 thresholds, learned models flag 0.614-0.688 of completed trials as terminated, i.e. 3.83-3.99 trials flagged per true termination. Max-F1 thresholds favour recall when the positive class is a minority; these scores are not suitable for flagging individual trials.

| model | roc_auc | roc_ci | pr_auc | pr_auc_lift_over_prevalence | brier | brier_better_than_prevalence | calibration_slope | well_calibrated | precision | recall | false_positive_rate | flagged_per_true_positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | 0.5 | 0.5-0.5 | 0.2245 | 1 | 0.2245 | False | n/a (constant prediction) | False | 0 | 0 | 0 | n/a |
| prevalence | 0.5 | 0.5-0.5 | 0.2245 | 1 | 0.1741 | False | n/a (constant prediction) | False | 0.2245 | 1 | 1 | 4.45 |
| logistic | 0.5932 | 0.562-0.625 | 0.2853 | 1.271 | 0.1709 | True | 0.918 | True | 0.2564 | 0.7865 | 0.6604 | 3.9 |
| lda | 0.5879 | 0.557-0.621 | 0.2819 | 1.256 | 0.1717 | False | 0.6831 | False | 0.252 | 0.7541 | 0.6479 | 3.97 |
| random_forest | 0.5924 | 0.561-0.624 | 0.2873 | 1.28 | 0.171 | True | 0.9565 | True | 0.2608 | 0.7486 | 0.6142 | 3.83 |
| hist_gradient_boosting | 0.6017 | 0.568-0.635 | 0.2918 | 1.3 | 0.1737 | False | 0.4498 | False | 0.2506 | 0.7946 | 0.6878 | 3.99 |

Reference models for probability estimates (calibrated and better than baseline on Brier): logistic, random_forest.

## Cohort-definition dependence

- In the accepted window, UNKNOWN-status studies are 0.116-0.179 of each year's eligible studies (1201 studies in total), and are excluded from the resolved-share denominator.
- Counting UNKNOWN as unresolved at the same 0.9 threshold, no start year qualifies; the most permissive strict variant tested (threshold 0.8) gives a window ending 2011. The window therefore depends on treating UNKNOWN as uninformative.
- Years whose Wilson 95% lower bound falls below 0.9 (borderline): [2014].
- If every in-window UNKNOWN study had in fact ended, the terminated share among in-window studies would lie between 0.191 (all completed) and 0.346 (all terminated), versus the observed 0.226.
- Across tested thresholds (UNKNOWN excluded), the cohort terminated share ranges 0.2261-0.2291, so the choice of threshold changes cohort size far more than outcome balance.

## Feature ablation

Dropping registration_lag_days (same temporal split, grids, seed and config):

- logistic: ROC-AUC 0.5932 -> 0.5862 (delta -0.0072, 95% CI -0.0165 to 0.0015); PR-AUC delta -0.0097 (-0.0235 to 0.0033); Brier delta 0.0007; calibration slope 0.918 -> 0.865.
- lda: ROC-AUC 0.5879 -> 0.5842 (delta -0.0037, 95% CI -0.0104 to 0.0027); PR-AUC delta -0.0098 (-0.0205 to 0.0001); Brier delta 0.0004; calibration slope 0.683 -> 0.653.
- random_forest: ROC-AUC 0.5924 -> 0.5826 (delta -0.0099, 95% CI -0.021 to 0.0015); PR-AUC delta -0.01 (-0.0242 to 0.0023); Brier delta 0.0008; calibration slope 0.956 -> 0.836.
- hist_gradient_boosting: ROC-AUC 0.6017 -> 0.5827 (delta -0.0188, 95% CI -0.0422 to 0.0008); PR-AUC delta -0.0051 (-0.0325 to 0.0213); Brier delta -0.0005; calibration slope 0.45 -> 0.456.

## Limitations
- Current-version records; no historical snapshots, so not prospective validation.
- Selection bias: the cohort conditions on reaching COMPLETED/TERMINATED within a resolved window; unresolved, UNKNOWN and WITHDRAWN studies are excluded.
- Sponsor-reported statuses; 'terminated' mixes safety, futility, accrual, business and early-success reasons.
- Modest discrimination and high false-positive rates (above).

## Intended use
Research and teaching about registry-level patterns.

## Inappropriate uses
Forecasting the fate of a specific trial; funding, regulatory, ethics or clinical decisions; any statement about patient risk.
