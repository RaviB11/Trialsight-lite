# Technical Notes

## Environment constraint discovered (2026-10-12)
The development sandbox cannot resolve `clinicaltrials.gov`, and the web-fetch
tool is blocked by the site's robots.txt. Consequently:
- The pipeline was written against the **documented** API v2 structure and
  verified only with mocked HTTP responses and a synthetic fixture.
- The first real `inspect` / `fetch` / `build` run happens on the developer's
  machine. `inspect` saves the API's own metadata, search areas and enums to
  `docs/api_snapshot/` so field paths and filter syntax can be checked against
  the live schema before analysis.

## Schema assumptions to verify on first real run
| Assumption | Source of confidence | How to verify |
|---|---|---|
| Field paths `protocolSection.<module>.<field>` (e.g. `statusModule.whyStopped`) | AACT documentation maps fields to these API paths | `studies_metadata.json` |
| `AREA[StudyType]`, `AREA[Phase]`, `AREA[StartDate]RANGE[a,b]` filter syntax | Standard Essie syntax as documented | `search_areas.json`; `fetch --dry-run` returns a plausible count, not an error |
| Enrollment/date `type` values are `ACTUAL` / `ESTIMATED` | Third-party API documentation | `enums.json` |
| MeSH ancestors include the term `Neoplasms` for cancer trials | Derived section structure | Inspect `mesh` table after build |
| Country string for US sites is `United States` | Common usage | `SELECT DISTINCT country FROM locations` |

## Design decisions
- **Broad pull, local filtering.** The API query keeps all statuses so the
  outcome filter is applied last and its effect is measurable
  (`status_by_start_year`).
- **Flag, don't delete.** Quality rules flag; only rules marked `blocking`
  (duplicate ID, malformed ID, missing status) exclude records, and that
  exclusion appears in the attrition table.
- **Partial dates** (`YYYY-MM`) are anchored to day 1 and their precision stored,
  so day-level arithmetic on them is visibly approximate.
- **Immutable raw zone with checksums**, so any build is traceable to an exact
  retrieval.

## Decision log
**2026-10-12 - v0.4.0: validation of the first full research run and publication readiness.**
- Reviewed raw run `20261012T020557Z`: verified download (35,637/35,637, recount matched), accepted build,
  cohort 6,541; EDA, models and the three generated documents all reference the same run and build.
- Defects found in the generated outputs: RESEARCH_FINDINGS printed `median nan` for a missingness finding,
  repeated a level table under the wrong finding, called all effects "small-to-moderate" (largest |effect|
  0.14), and printed q-values below 1e-4 as 0; Wilson bounds could display as `-0.000`; unstable subgroups
  (n=1-2, one class) were unflagged; the PR plot drew a misleading diagonal for the constant baseline.
  All fixed; documents regenerated from the same artifacts.
- `build_info.json` stored an absolute Windows path including the user name; output paths are now
  project-relative, and `check-consistency` scans publishable files for local paths.
- Added: computed interpretation (modest discrimination, calibration, false-positive burden), UNKNOWN-status
  analysis with bounds, `trialsight ablate`, `trialsight check-consistency`, app provenance/ablation panels,
  README figure script. The ablation and the database fingerprint check (C10) need the research database and
  were not run in the development sandbox.

**2026-10-12 - v0.3.0: integrity hardening, analysis, modelling, app.**
- *Reconciliation could not be trusted with critical failures.* Issues are now split into critical (never
  reconcilable) and reconcilable. A forged reconciliation.json with a matching snapshot still grants nothing
  if any critical issue exists. Adversarial test: 1 record retrieved when the API reports 1,000.
- *Recount.* `fetch` records `recount_requested`, and any failure as `total_count_at_end_error` (exception
  type and message). A failed or missing recount is recorded as such and blocks `verified`.
- *Corrupt inputs.* `verify_run` returns explicit failures for truncated gzip, invalid JSON, missing pages,
  missing or garbage manifest, and unreadable reconciliation files; it never raises for these.
- *Query consistency.* Canonicalisation drops paging/format parameters and normalises whitespace around
  brackets and parentheses; any remaining difference blocks research builds and fails check V10.
- *Safe outputs.* Research builds are written to `data/.processed_staging/`, promoted to `data/processed/` only
  when accepted (one backup kept as `processed_previous`), otherwise moved to
  `data/processed_rejected/<timestamp>/` with `INVALID_DO_NOT_USE.txt`. The guard now also requires
  `output_state == "accepted"`; library builds that bypass staging are `unstaged` and refused.
- *v2 tests changed intentionally.* Three v2 guard tests fetched with a dummy query that did not match their
  config; the new consistency check correctly refused them. They now fetch with a matching query, and the
  unvalidated-build test expects quarantine. The 48 baseline test files are unchanged (SHA-256 verified).
- *PyArrow.* Streamlit hard-depends on PyArrow, so the app is an optional extra (`requirements-app.txt`).
  The pipeline, EDA and modelling were tested in an environment without PyArrow: 144 passed, 2 skipped (app).
- *Calibration.* statsmodels recalibration fits that fail to converge (small or separable samples) are
  reported as NaN instead of unreliable numbers.
- *Environment slip.* An editable install from an earlier zip check left the `trialsight` command pointing at
  an old copy; tests were unaffected (they import `src/`). Reinstalled from the workspace.
- *Python 3.11.* `numpy==2.5.4` was mistakenly pinned (it does not support 3.11); pinned 2.4.4. The full
  suite (146 tests) passes on Python 3.11.15 and 3.12.3.
- *Not done.* No real data processed here (sandbox cannot reach ClinicalTrials.gov). Phase 5 extensions
  and the optional LLM assistant are deferred until the primary experiment runs on validated data.

**2026-10-12 - v2 pre-download hardening.** Context, as reported from the
developer's Windows machine: 48 tests passed, live inspection succeeded, the
configured query reported totalCount 35,637, and a one-page development sample
(1,000 records) built 715 eligible / cohort 170 / window 2008-2013.
- *Verified vs complete.* `manifest.complete` still means only "pagination
  finished". `verify_run` recomputes integrity from disk and is the only source
  of research-grade status. Count mismatch, duplicates, pagination anomalies, a
  totalCount change during download, or a manifest/page-1 totalCount
  disagreement make a run `unverified` until re-fetched or explicitly
  reconciled (`reconciliation.json`, bound to the manifest hash and a
  verification snapshot; written reason of 20+ characters required).
- *Debugging discovery.* A regression test showed that editing the manifest's
  totalCount could make a mismatched run look verified. totalCount is now read
  from the checksummed page-1 payload and any disagreement is flagged.
- *Pagination.* Fetch records anomalies (empty or short non-final pages,
  oversized pages, records without nctId, missing totalCount) and stops if pages
  exceed `2 * ceil(totalCount / pageSize) + 5` (77 for 35,637 at 1,000/page).
  `fetch` re-queries totalCount at the end.
- *Duplicates.* Byte-identical copies are collapsed during build (counted in
  `build_metadata`); conflicting copies are kept so DQ01 excludes them.
- *Resolution rule.* Added `missing_year_policy`, `min_studies_per_year`,
  `ambiguous_handling` and Wilson intervals. Dataclass defaults reproduce v1;
  the shipped config enables `break` and `30` (fail-closed). See METHODOLOGY.md.
- *Diagnostics.* threshold_sensitivity, terminated_share_by_year,
  selection_profile, oncology_capture (+ review sample and top MeSH terms),
  phase_audit, date_audit, start_type_by_status, enum_check, cohort_validation,
  all in DuckDB, Parquet (DuckDB-native) and CSV, plus
  `cohort_validation_report.md`.
- *Population guard.* Every build carries `population_status`.
  `build --development` (and the old `--allow-incomplete`, now an alias)
  writes only to `data/processed_dev/` and is always labelled
  `development_not_for_research`. Default `build` refuses non-research-grade
  runs. `require_research_population()` fails closed for development builds,
  legacy databases without `build_metadata`, and unvalidated cohorts.
- *Behaviour change to note.* `build --allow-incomplete` previously wrote to
  `data/processed/`. It now writes to `data/processed_dev/`. The existing
  `data/processed/` from the v1 sample has no `build_metadata`, so the guard
  rejects it; archive it before the full build.
- *Library-level default.* `build_database()` called directly from Python does
  not refuse unverified runs by default (this preserves the unchanged baseline
  tests); it labels them `development_not_for_research`, and the guard blocks
  their use. The CLI is strict by default.
- *Not verifiable in the sandbox.* Whether the live API always returns exactly
  pageSize records on non-final pages. If not, the full run will show
  "non-final page" anomalies and stay unverified until reviewed.

**2026-10-12 - Parquet export moved to DuckDB's native writer; PyArrow removed.**
- `storage.export_parquet` now runs `COPY <table> TO '<path>' (FORMAT PARQUET,
  COMPRESSION ZSTD)` against the DuckDB file after tables are written, so each
  Parquet file mirrors the stored table exactly.
- PyArrow was used only for `DataFrame.to_parquet`; it is no longer a dependency.
- Verified in a clean virtual environment with pandas 3.0.2 and duckdb 1.5.6 and
  no PyArrow installed: `import pyarrow` fails, and all 48 tests pass, including
  the end-to-end build and Parquet round-trip tests.
- Caveat: pandas uses PyArrow-backed strings when PyArrow is present and plain
  Python strings when it is absent. Code here does not depend on either, but
  anyone reading the Parquet files with `pandas.read_parquet` will need PyArrow
  or fastparquet (or can read them with DuckDB instead).

**2026-10-12 (later) - Data-driven analysis window (supersedes the entry below).**
- Fetch / eligible population: start dates 2008-01-01..2025-12-31, all statuses.
  2008 follows the FDAAA registration requirements that took effect at the end of 2007.
- Primary classification cohort: start years from 2008 up to an end year chosen
  by `resolution_rule` (config). Per start year,
  `resolved_share = resolved / (resolved + unresolved)`, where resolved =
  COMPLETED, TERMINATED, WITHDRAWN and unresolved = not yet recruiting,
  recruiting, enrolling by invitation, active not recruiting, suspended.
  UNKNOWN is excluded from the share and reported as `ambiguous_share`.
- The window ends at the last year of the initial consecutive run of years with
  `resolved_share >= 0.90`, so it never has gaps. If the earliest year fails,
  the cohort is empty (with a logged warning) rather than silently widened.
- Outputs: `resolution_by_start_year` (the evidence), a dedicated attrition
  step, `analysis_window` in `build_info.json`, and `eligible_population`
  (all statuses, full window) reserved for survival analysis of recent starts.
- Known caveats: (1) 0.90 is a judgement threshold; a sensitivity check at other
  thresholds is planned. (2) Excluding UNKNOWN from the denominator may flatter
  older years if UNKNOWN is common there; `ambiguous_share` makes this visible.
  (3) The 2017 Final Rule falls inside the likely window; a pre/post-2017
  comparison is planned.

**2026-10-12 - Start-date window set to 2015-01-01..2025-12-31** (superseded above) (project owner's
decision; replaces the proposed 2008-2017).
- Benefit: records post-date the 2017 Final Rule era more often, so newer fields
  (e.g. FDA-regulated flags) should be better populated; findings are more current.
- Risk - right-censoring / outcome-dependent selection: at retrieval (Oct 2026),
  many trials started 2020-2025 will still be active. Among recent starts, those
  already in a terminal status are disproportionately early stops, so the
  terminated share is expected to be inflated for later start years, and
  `start_year` may act as a proxy for follow-up time rather than a real effect.
  A chronological train/test split inherits the same distortion.
- Additional context: 2020-2021 starts overlap the COVID-19 disruption period.
- Mitigations planned (to be decided after seeing real data, not assumed):
  1. Inspect `status_by_start_year` to quantify the unresolved share per year.
  2. Sensitivity analysis restricted to start years with a high resolved share.
  3. Treat `start_year` as a confounder/censoring proxy, not an interpretable driver.
  4. Consider survival analysis (time to termination, active trials censored),
     which handles unresolved trials directly instead of dropping them.

## Open decisions (need confirmation before analysis)
1. ~~Start-date window~~ - decided, see Decision log.
2. Treatment of WITHDRAWN (proposed: excluded from the primary target; reported descriptively).
3. Inclusion of combined PHASE2/PHASE3 (proposed: included as its own `phase_group`).
4. Oncology definition (proposed: API condition search, confirmed by MeSH ancestor `Neoplasms`; note this includes benign neoplasms).
