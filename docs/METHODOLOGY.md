# Methodology

Status: applied to the verified research run `20261012T020557Z` (cohort 6,541). Results are in
EVALUATION_REPORT.md and RESEARCH_FINDINGS.md, generated from saved artifacts.

## Research question
Can publicly available registration metadata reveal patterns associated with
completion versus termination of Phase II/III interventional oncology trials,
and how well can interpretable models describe these historical relationships?
The analysis is retrospective. The API returns only current record versions, so
no claim of prospective predictive validity is made.

## Population pipeline
1. **Download** - ClinicalTrials.gov API v2 `/studies`, condition search
   `cancer OR neoplasm`, interventional, phase filter containing PHASE2 or
   PHASE3, start date 2008-01-01..2025-12-31, all statuses.
2. **Verification gate** - `trialsight verify` recomputes integrity from disk.
   A run is research-grade only if retrieved and unique counts equal the
   API-reported totalCount (read from the checksummed page-1 payload), there are
   no duplicate IDs or pagination anomalies, and checksums are valid; or if a
   complete run has an explicit, written `trialsight reconcile` record bound to
   that exact manifest. Incomplete or corrupt runs cannot be reconciled.
3. **Eligibility** - passes blocking data-quality rules (DQ01-DQ03),
   interventional, phase group PHASE2 / PHASE2_3 / PHASE3, start date in range,
   condition mapped to MeSH `Neoplasms`.
4. **Analysis window** - from 2008 to the last year of the initial unbroken run
   of start years meeting the resolution rule (below).
5. **Outcome** - COMPLETED = 0, TERMINATED = 1. Other statuses are excluded from
   the classification cohort but kept in `eligible_population`.

## Resolution rule
Per start year, `resolved_share = resolved / denominator`, where resolved =
COMPLETED, TERMINATED, WITHDRAWN.

| Setting | Value | Effect |
|---|---|---|
| min_resolved_share | 0.90 | year qualifies at >= 90% resolved |
| ambiguous_handling | exclude | UNKNOWN left out of the denominator; reported as `ambiguous_share` |
| missing_year_policy | break | an empty calendar year ends the window (added in v2) |
| min_studies_per_year | 30 | a year with < 30 resolvable studies cannot qualify (added in v2) |

The two v2 settings are fail-closed: they can only shorten the window, never
lengthen it. They were added because (a) the original rule silently spanned
empty years, so a "2008-2014" window could include a year with no data, and
(b) shares from very small years are unstable. For a year at 90% resolved, the
95% Wilson interval computed by this code is [0.596, 0.982] at n=10,
[0.744, 0.965] at n=30, [0.826, 0.945] at n=100 and [0.880, 0.917] at n=1000.
n=30 is a floor against near-empty years, not a precision guarantee; every year
reports its Wilson interval. To restore v1 behaviour, set
`missing_year_policy: skip` and `min_studies_per_year: 1`.

## Known biases and how they are made visible
- **Right-censoring / outcome-dependent selection**: recent starts that already
  ended are disproportionately early stops. Visible in
  `terminated_share_by_year` and `resolution_by_start_year`.
- **Excluding unresolved and UNKNOWN studies inside the window**: they may
  differ systematically from the cohort (e.g. long-running trials, or
  terminated trials never updated). `selection_profile` compares phase, sponsor
  class, allocation and purpose across cohort / unresolved / ambiguous groups.
- **Query visibility**: studies the API condition search never returned cannot
  be measured by any local diagnostic.
- **Sample builds are not populations**: API result order is not random, so a
  one-page sample is not representative. The v1 sample (1,000 records, 715
  eligible, cohort 170, window 2008-2013, as reported from the developer's
  machine) is a development artefact only.

## Alternatives considered (not applied) and their consequences
These are documented so the choice is explicit; none was adopted to increase
the record count. The diagnostic that quantifies each is named.

| Alternative | Consequence | Diagnostic |
|---|---|---|
| Include PHASE1/PHASE2 trials (returned by the API phase filter) | More records, but mixes dose-finding designs with different stopping dynamics; changes the research question | `phase_audit` |
| Oncology = MeSH Neoplasms OR cancer condition text | Recovers records with missing or non-neoplasm MeSH; adds false positives (e.g. "tumor necrosis factor", supportive-care trials) and needs manual review | `oncology_capture_summary`, `oncology_review_sample` |
| Oncology = malignant MeSH subtree only | Removes benign neoplasms currently included; fewer records | not yet implemented |
| Broader `query.cond` (e.g. adding lymphoma, leukemia, myeloma, sarcoma, melanoma) | May surface studies the current search never returns; requires a new download | `trialsight count --cond "..."` |
| ambiguous_handling = count_as_unresolved | Treats unknown outcomes as unresolved; shorter window, fewer records, arguably more conservative | `threshold_sensitivity` |
| Threshold 0.80 / 0.85 / 0.95 | Lower: larger cohort, more censoring bias. Higher: smaller cohort, older trials only | `threshold_sensitivity` |
| Treat WITHDRAWN as terminated | Merges never-enrolled studies with stopped studies; changes the outcome definition | `status_by_start_year` |
| Exclude studies whose start date is ESTIMATED | Removes anticipated starts; changes resolution denominators | `date_audit`, `start_type_by_status` |
| Reconcile instead of re-fetching a mismatched download | Accepts a population that differs from the registry snapshot; must be justified in writing | `trialsight verify` |

## Cohort validation gate
`cohort_validation` lists checks V01-V09. Critical checks: V01 research-grade
download, V02 checksums, V03 window selected, V06 every status mapped to a
resolution group, V07 cohort non-empty with both classes. Modelling code must
call `trialsight.storage.require_research_population`, which fails closed on
development builds, legacy databases, and unvalidated cohorts.

## Integrity rules (v0.3.0)
A complete download is research-grade only when it has no **critical** issue: corrupt/missing/invalid pages
or manifest, incomplete pagination, conflicting duplicate records, records without nctId, a unique-record gap
larger than `max(3, 0.05% of totalCount)` (17 for 35,637), pagination anomalies, a manifest totalCount that
disagrees with the checksummed page 1, or an end-of-download recount that moved beyond the tolerance.
**Reconcilable** issues (a gap within tolerance, byte-identical duplicate copies, a failed or missing recount)
need an explicit written reconciliation. Critical issues cannot be reconciled; notes on them are
documentation-only. The raw run's canonicalised cohort-defining query must equal the current config (V10).
Research builds are staged and promoted only if every critical validation check passes; otherwise they are
quarantined in `data/processed_rejected/` with an INVALID marker. Downstream commands load data only through
`study_data.load_study_data`, which refuses anything except an accepted research build unless
`--development` is given.

## Statistical methods (exploratory analysis)
- Categorical predictors: chi-square test of independence (Fisher's exact test for 2x2 tables with any
  expected count < 5); effect size = bias-corrected Cramér's V. Missing values form an explicit level.
- Per-level termination rates with Wilson 95% intervals.
- Numeric predictors: Mann-Whitney U; effect size = rank-biserial r (positive = higher in terminated);
  median difference (terminated - completed) with a percentile bootstrap 95% CI.
- Missingness: missing-vs-outcome tests for every analysed field.
- Multiple testing: Benjamini-Hochberg FDR at 5% across every p-value in one EDA run.
- Therapeutic areas (MeSH ancestors) are overlapping labels, so they are reported descriptively without tests.
- `start_year` is analysed descriptively; within the window it partly reflects follow-up time.

## Modelling
- Outcome: TERMINATED = 1, COMPLETED = 0, within the validated cohort.
- Primary features: `feature_set("primary")` = registry fields classed `baseline_candidate`
  (design, sponsor class, eligibility, intervention-type and age-group indicators, counts). Excluded:
  post-outcome fields, identifiers, time-varying-suspect fields (site counts, collaborators, secondary
  outcomes) and `start_year`. A leakage check (`assert_no_leakage`) raises on any forbidden field.
- Preprocessing (scikit-learn pipelines fitted on training data only): median imputation with missing
  indicators and standardisation for numeric fields; constant imputation and one-hot encoding (rare levels
  grouped, unseen levels ignored) for categorical fields; zero-variance removal.
- Models: majority class; prevalence probability; L2 logistic regression; shrinkage LDA; random forest;
  histogram gradient boosting (kept only if it beats simpler models with non-overlapping uncertainty).
- Tuning: grid search with stratified K-fold CV on the training set (scoring PR-AUC); class_weight is a
  tuned option; no resampling anywhere.
- Thresholds for precision/recall/F1: maximise F1 on out-of-fold training predictions.

## Evaluation design
- Primary: temporal hold-out; the latest 25% of the cohort by start date is the test set. Each NCT ID
  appears once; train/test disjointness is asserted. Minimum 50 events of each class in training and 30 in
  test, otherwise the run stops with InsufficientDataError rather than reporting unstable metrics.
- Metrics: prevalence, precision, recall, F1, ROC-AUC, PR-AUC, Brier, log loss, calibration intercept/slope
  (reported as unavailable when the recalibration fit does not converge), quantile calibration bins,
  confusion matrix.
- Uncertainty: 1,000 bootstrap resamples of the test set; percentile 95% CIs and paired differences against
  the prevalence baseline on identical resamples.
- Subgroups: phase group, sponsor class, start year. Failure analysis: false-negative/false-positive rates by
  subgroup at the training-derived threshold.
- Sensitivity: stratified random split; adding time-varying-suspect fields; adding start_year.
- Interpretation: logistic coefficients (standardised/one-hot scale) and test-set permutation importance
  (PR-AUC drop), each tagged with its availability class.
- Reproducibility: fixed seeds, single-threaded fitting, saved models, a cohort fingerprint and a test-ID
  hash; `trialsight reproduce` recomputes every saved metric from the saved models.

## Extensions not yet run
Clustering, survival analysis and Bayesian modelling (Phase 5) are deferred until the primary experiment has
run on the validated cohort. `eligible_population` keeps unresolved studies for a future time-to-event
analysis (study completion, not recruitment completion, with unresolved studies censored).

## Reporting conventions (v0.4.0)
- Effect-size labels for |Cramér's V| and |rank-biserial r|: < 0.1 very small, < 0.3 small, < 0.5 moderate,
  otherwise large. These are rough conventions; with thousands of studies, significant effects can be trivially
  small.
- Undefined values are shown as `n/a` with the reason (e.g. calibration of a constant-score baseline, ROC-AUC
  in a subgroup with one class). Wilson bounds are clipped to [0, 1].
- Subgroups and levels with fewer than 20 terminated or completed studies (or fewer than 30 studies for EDA
  levels) are flagged UNSTABLE and should not be interpreted.
- Tuning selects hyperparameters by PR-AUC, which can choose class weighting; class-weighted models produce
  inflated probabilities, so their Brier scores and calibration are not comparable with unweighted models.

## UNKNOWN statuses and the window
UNKNOWN studies are excluded from the resolved-share denominator. In the verified run this choice is decisive:
counting them as unresolved, no start year reaches 90%. The cohort's terminated share is nonetheless stable
across thresholds (0.2261-0.2291), so the threshold mainly changes cohort size. Bounds on the in-window
terminated share, assuming every UNKNOWN study had ended, are reported in RESEARCH_FINDINGS.md.

## Feature ablation
`trialsight ablate --drop <feature>` re-runs the saved primary protocol (same temporal split verified by
test-ID hash, same grids, seed and configuration) without the named features, and compares against the saved
primary predictions on identical bootstrap resamples. It also reports registration timing relative to start
and (for diagnosis only) completion dates. The first target is `registration_lag_days`, the top permutation
feature for logistic regression and gradient boosting, because registration can occur after a study starts
or ends.
