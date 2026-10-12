# Research findings

_Generated from saved artifacts. Source: raw run `20261012T020557Z`, build 2026-10-12T02:11:44.836572+00:00, cohort fingerprint `2b8ddab1bf698355`._

Cohort: n=6541, terminated 1479 (22.6%), start years 2008-2014. 36 tests; Benjamini-Hochberg FDR 5%; 21 significant.

## Observations

Every significant association is at most small (largest |effect| = 0.1425). With n in the thousands, tests detect effects too small to matter for any individual trial.

1. **n_locations** (numeric; time_varying_suspect) - rank-biserial r = -0.1425 (small), q = 2.93e-16. median terminated 2 vs completed 4; median difference -2 (95% CI -3 to -2). Evidence: `eda_associations.csv`. Time-varying field: may have been updated after the outcome.
2. **n_secondary_outcomes** (numeric; time_varying_suspect) - rank-biserial r = -0.1211 (small), q = 1.67e-11. median terminated 3 vs completed 4; median difference -1 (95% CI -1 to 0). Evidence: `eda_associations.csv`. Time-varying field: may have been updated after the outcome.
3. **n_arms** (numeric; baseline_candidate) - rank-biserial r = -0.071 (very small), q = 3.13e-05. median terminated 1 vs completed 2; median difference -1 (95% CI -1 to 0). Evidence: `eda_associations.csv`.
4. **phase_group** (categorical; baseline_candidate) - Cramer's V (bias-corrected) = 0.0683 (very small), q = 1.04e-06. PHASE2: 0.242 [0.231, 0.255] (n=4919); PHASE2_3: 0.227 [0.171, 0.295] (n=172); PHASE3: 0.171 [0.152, 0.191] (n=1450). Evidence: `eda_associations.csv`, `eda_levels.csv`.
5. **eligibility_criteria_chars** (numeric; baseline_candidate) - rank-biserial r = 0.0644 (very small), q = 0.000410169. median terminated 2613 vs completed 2307; median difference 306 (95% CI 153.5 to 475.025). Evidence: `eda_associations.csv`.
6. **has_dmc** (categorical; baseline_candidate) - Cramer's V (bias-corrected) = 0.0597 (very small), q = 2.34e-05. (missing): 0.159 [0.134, 0.188] (n=705); False: 0.22 [0.203, 0.237] (n=2359); True: 0.244 [0.23, 0.259] (n=3477). Evidence: `eda_associations.csv`, `eda_levels.csv`.
7. **max_age_years** (missingness; baseline_candidate) - Cramer's V = 0.0575 (very small), q = 1.74e-05. missing in 0.772 of terminated vs 0.709 of completed studies. Evidence: `eda_associations.csv`.
8. **lead_sponsor_class** (categorical; baseline_candidate) - Cramer's V (bias-corrected) = 0.0561 (very small), q = 0.000414299. FED: 0 [0, 0.434] (n=5, small n - unstable); INDIV: 0.333 [0.097, 0.7] (n=6, small n - unstable); INDUSTRY: 0.203 [0.186, 0.22] (n=2093); NETWORK: 0.151 [0.109, 0.204] (n=219); NIH: 0.207 [0.167, 0.254] (n=338); OTHER: 0.246 [0.233, 0.26] (n=3818); OTHER_GOV: 0.161 [0.09, 0.272] (n=62). Evidence: `eda_associations.csv`, `eda_levels.csv`.
9. **is_fda_regulated_drug** (categorical; baseline_candidate) - Cramer's V (bias-corrected) = 0.056 (very small), q = 4.69e-05. (missing): 0.235 [0.224, 0.246] (n=5641); False: 0.12 [0.08, 0.176] (n=175); True: 0.181 [0.154, 0.21] (n=725). Evidence: `eda_associations.csv`, `eda_levels.csv`.
10. **has_dmc** (missingness; baseline_candidate) - Cramer's V = 0.0545 (very small), q = 3.20e-05. missing in 0.076 of terminated vs 0.117 of completed studies. Evidence: `eda_associations.csv`.

## Right-censoring
Within the window the terminated share per start year ranges 0.211-0.238; for start years after 2016 it ranges 0.246-0.381, consistent with recent studies that ended early being over-represented among those already finished.

## Cohort window and UNKNOWN statuses

- In the accepted window, UNKNOWN-status studies are 0.116-0.179 of each year's eligible studies (1201 studies in total), and are excluded from the resolved-share denominator.
- Counting UNKNOWN as unresolved at the same 0.9 threshold, no start year qualifies; the most permissive strict variant tested (threshold 0.8) gives a window ending 2011. The window therefore depends on treating UNKNOWN as uninformative.
- Years whose Wilson 95% lower bound falls below 0.9 (borderline): [2014].
- If every in-window UNKNOWN study had in fact ended, the terminated share among in-window studies would lie between 0.191 (all completed) and 0.346 (all terminated), versus the observed 0.226.
- Across tested thresholds (UNKNOWN excluded), the cohort terminated share ranges 0.2261-0.2291, so the choice of threshold changes cohort size far more than outcome balance.

## Oncology definition
Strict MeSH matching excluded 2417 studies with cancer-related condition text but no MeSH annotations (0.084 of the pre-oncology pool, all years); among those with a final status, 0.233 were terminated. See the cohort validation report.

## Model evidence

- Discrimination is modest. The highest test ROC-AUC is 0.6017 (hist_gradient_boosting; 95% CI 0.568-0.635); learned models range 0.5879-0.6017. PR-AUC is 0.2819-0.2918 against a prevalence of 0.2245 (lift 1.26-1.3x).
- Every learned model's ROC-AUC and PR-AUC improvement over the prevalence baseline has a paired 95% CI above zero, but the improvements are small and the learned models' intervals overlap heavily.
- Calibrated models (slope 0.8-1.2) whose Brier score beats the baseline: logistic, random_forest. Poorly calibrated: lda, hist_gradient_boosting. A higher ROC-AUC without calibration is not a better model for probability estimates.
- False positives are high. At the training-derived max-F1 thresholds, learned models flag 0.614-0.688 of completed trials as terminated, i.e. 3.83-3.99 trials flagged per true termination. Max-F1 thresholds favour recall when the positive class is a minority; these scores are not suitable for flagging individual trials.

Dropping registration_lag_days (same temporal split, grids, seed and config):

- logistic: ROC-AUC 0.5932 -> 0.5862 (delta -0.0072, 95% CI -0.0165 to 0.0015); PR-AUC delta -0.0097 (-0.0235 to 0.0033); Brier delta 0.0007; calibration slope 0.918 -> 0.865.
- lda: ROC-AUC 0.5879 -> 0.5842 (delta -0.0037, 95% CI -0.0104 to 0.0027); PR-AUC delta -0.0098 (-0.0205 to 0.0001); Brier delta 0.0004; calibration slope 0.683 -> 0.653.
- random_forest: ROC-AUC 0.5924 -> 0.5826 (delta -0.0099, 95% CI -0.021 to 0.0015); PR-AUC delta -0.01 (-0.0242 to 0.0023); Brier delta 0.0008; calibration slope 0.956 -> 0.836.
- hist_gradient_boosting: ROC-AUC 0.6017 -> 0.5827 (delta -0.0188, 95% CI -0.0422 to 0.0008); PR-AUC delta -0.0051 (-0.0325 to 0.0213); Brier delta -0.0005; calibration slope 0.45 -> 0.456.

## Practical implications
Patterns describe registry records, not trial-design advice; the models should not be used to flag individual trials.

## Future research questions
- Do results hold with historical record versions?
- Time-to-event analysis including unresolved trials.
- Are termination reasons (why_stopped) associated with distinct design profiles?
