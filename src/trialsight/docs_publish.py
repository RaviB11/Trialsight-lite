"""Generate MODEL_CARD.md, EVALUATION_REPORT.md and RESEARCH_FINDINGS.md from saved artifacts.

Refuses development builds and fast/smoke runs, so documentation can only ever
contain numbers computed from an accepted, validated research population.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .report import MIN_EVENTS_STABLE, flag_stability, fmt_cell, md_table

RESEARCH = ("research_verified", "research_reconciled")


def _check_publishable(eda_meta: dict, exp: dict) -> None:
    problems = []
    if eda_meta.get("development") or eda_meta.get("population_status") not in RESEARCH:
        problems.append(f"EDA ran on {eda_meta.get('population_status')!r}")
    if exp.get("development") or exp["source"].get("population_status") not in RESEARCH:
        problems.append(f"models trained on {exp['source'].get('population_status')!r}")
    if exp["config"].get("fast"):
        problems.append("models were trained in --fast smoke mode")
    if eda_meta.get("raw_run_id") != exp["source"].get("raw_run_id"):
        problems.append("EDA and models come from different raw runs")
    if problems:
        raise ValueError("Refusing to publish documentation: " + "; ".join(problems))


def _r(df: pd.DataFrame, cols: list[str], nd: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)
    return out



BASELINES = ("majority", "prevalence")


def f(x, nd: int = 4) -> str:
    try:
        return fmt_cell(round(float(x), nd))
    except (TypeError, ValueError):
        return fmt_cell(x)


def magnitude(effect: float) -> str:
    """Rough conventions for |Cramer's V| and |rank-biserial r| (documented in METHODOLOGY.md)."""
    a = abs(effect)
    return "very small" if a < 0.1 else "small" if a < 0.3 else "moderate" if a < 0.5 else "large"


def load_processed(processed_dir: Path | None) -> dict:
    out: dict = {}
    if processed_dir and Path(processed_dir).exists():
        for name in ("resolution_by_start_year", "threshold_sensitivity", "terminated_share_by_year",
                     "oncology_capture_summary", "cohort_attrition"):
            path = Path(processed_dir) / f"{name}.csv"
            if path.exists():
                out[name] = pd.read_csv(path)
        bi = Path(processed_dir) / "build_info.json"
        if bi.exists():
            out["build_info"] = json.loads(bi.read_text(encoding="utf-8"))
    return out


def unknown_analysis(proc: dict) -> tuple[str, pd.DataFrame]:
    """How UNKNOWN statuses shape the window, and the bounds they imply for the termination rate."""
    ts, res = proc.get("threshold_sensitivity"), proc.get("resolution_by_start_year")
    if ts is None or res is None:
        return "_UNKNOWN-status analysis unavailable (cohort tables not supplied)._\n", pd.DataFrame()
    conf = ts[ts["is_configured_rule"].astype(str).str.lower() == "true"].iloc[0]
    thr, n, t, amb = float(conf["threshold"]), int(conf["cohort_n"]), int(conf["n_terminated"]), int(conf["in_window_ambiguous"])
    lo, hi = t / (n + amb), (t + amb) / (n + amb)
    strict = ts[(ts["ambiguous_handling"] == "count_as_unresolved") & (ts["threshold"] == thr)]
    strict_end = strict["end_year"].iloc[0] if len(strict) else None
    strict_any = ts[(ts["ambiguous_handling"] == "count_as_unresolved") & ts["end_year"].notna()]
    excl = ts[ts["ambiguous_handling"] == "exclude"]
    w = res[res["in_analysis_window"].astype(str).str.lower() == "true"].copy()
    w["strict_share"] = (w["n_resolved"] / (w["n_resolved"] + w["n_unresolved"] + w["n_ambiguous"])).round(4)
    w["borderline"] = w["resolved_share_ci_low"] < thr
    table = w[["start_year", "n_total", "n_resolved", "n_unresolved", "n_ambiguous", "resolved_share",
               "resolved_share_ci_low", "strict_share", "ambiguous_share", "borderline"]]
    border = w.loc[w["borderline"], "start_year"].astype(int).tolist()
    if len(strict_any):
        loosest = strict_any.sort_values("threshold").iloc[0]
        best_strict = (f"the most permissive strict variant tested (threshold {f(loosest['threshold'], 2)}) gives a "
                       f"window ending {int(loosest['end_year'])}")
    else:
        best_strict = "no strict variant tested qualifies any start year"
    text = (
        f"- In the accepted window, UNKNOWN-status studies are {f(w['ambiguous_share'].min(), 3)}-"
        f"{f(w['ambiguous_share'].max(), 3)} of each year's eligible studies ({amb} studies in total), and are "
        "excluded from the resolved-share denominator.\n"
        f"- Counting UNKNOWN as unresolved at the same {f(thr, 2)} threshold, "
        f"{'no start year qualifies' if pd.isna(strict_end) else f'the window would end in {int(strict_end)}'}; "
        f"{best_strict}. The window therefore depends on treating UNKNOWN as uninformative.\n"
        f"- Years whose Wilson 95% lower bound falls below {f(thr, 2)} (borderline): {border or 'none'}.\n"
        f"- If every in-window UNKNOWN study had in fact ended, the terminated share among in-window studies would lie "
        f"between {f(lo, 3)} (all completed) and {f(hi, 3)} (all terminated), versus the observed {f(t / n, 3)}.\n"
        f"- Across tested thresholds (UNKNOWN excluded), the cohort terminated share ranges "
        f"{f(excl['terminated_share'].min(), 4)}-{f(excl['terminated_share'].max(), 4)}, so the choice of threshold "
        "changes cohort size far more than outcome balance.\n")
    return text, table


def model_interpretation(metrics: pd.DataFrame, boot: pd.DataFrame) -> tuple[str, pd.DataFrame, list[str]]:
    m = metrics.set_index("model")
    prev = float(m.iloc[0]["prevalence"])
    learned = [x for x in m.index if x not in BASELINES]
    b = boot.set_index(["model", "metric"])
    rows = []
    for name in m.index:
        r = m.loc[name]
        fpr = r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) else float("nan")
        brier_better = name in learned and b.loc[(name, "brier"), "delta_ci_high"] < 0
        slope = r["calibration_slope"]
        cal_ok = pd.notna(slope) and 0.8 <= slope <= 1.2
        rows.append({"model": name, "roc_auc": r["roc_auc"], "roc_ci": f"{f(b.loc[(name, 'roc_auc'), 'ci_low'], 3)}-"
                     f"{f(b.loc[(name, 'roc_auc'), 'ci_high'], 3)}", "pr_auc": r["pr_auc"],
                     "pr_auc_lift_over_prevalence": round(r["pr_auc"] / prev, 3),
                     "brier": r["brier"], "brier_better_than_prevalence": bool(brier_better),
                     "calibration_slope": round(float(slope), 4) if pd.notna(slope) else "n/a (constant prediction)",
                     "well_calibrated": bool(cal_ok), "precision": r["precision"], "recall": r["recall"],
                     "false_positive_rate": round(fpr, 4) if fpr == fpr else "n/a",
                     "flagged_per_true_positive": round((r["tp"] + r["fp"]) / r["tp"], 2) if r["tp"] else "n/a"})
    t = pd.DataFrame(rows)
    best = m.loc[learned, "roc_auc"].idxmax()
    good = [x for x in learned if t.set_index("model").loc[x, "well_calibrated"]
            and t.set_index("model").loc[x, "brier_better_than_prevalence"]]
    poor = [x for x in learned if not t.set_index("model").loc[x, "well_calibrated"]]
    gl = t.set_index("model")
    text = (
        f"- Discrimination is modest. The highest test ROC-AUC is {f(m.loc[best, 'roc_auc'])} ({best}; 95% CI "
        f"{gl.loc[best, 'roc_ci']}); learned models range {f(m.loc[learned, 'roc_auc'].min())}-"
        f"{f(m.loc[learned, 'roc_auc'].max())}. PR-AUC is {f(m.loc[learned, 'pr_auc'].min())}-"
        f"{f(m.loc[learned, 'pr_auc'].max())} against a prevalence of {f(prev)} "
        f"(lift {f(gl.loc[learned, 'pr_auc_lift_over_prevalence'].min(), 2)}-{f(gl.loc[learned, 'pr_auc_lift_over_prevalence'].max(), 2)}x).\n"
        "- Every learned model's ROC-AUC and PR-AUC improvement over the prevalence baseline has a paired 95% CI above "
        "zero, but the improvements are small and the learned models' intervals overlap heavily.\n"
        f"- Calibrated models (slope 0.8-1.2) whose Brier score beats the baseline: {', '.join(good) or 'none'}. "
        f"Poorly calibrated: {', '.join(poor) or 'none'}. A higher ROC-AUC without calibration is not a better model "
        "for probability estimates.\n"
        f"- False positives are high. At the training-derived max-F1 thresholds, learned models flag "
        f"{f(gl.loc[learned, 'false_positive_rate'].astype(float).min(), 3)}-{f(gl.loc[learned, 'false_positive_rate'].astype(float).max(), 3)} "
        f"of completed trials as terminated, i.e. {f(gl.loc[learned, 'flagged_per_true_positive'].astype(float).min(), 2)}-"
        f"{f(gl.loc[learned, 'flagged_per_true_positive'].astype(float).max(), 2)} trials flagged per true termination. "
        "Max-F1 thresholds favour recall when the positive class is a minority; these scores are not suitable for "
        "flagging individual trials.\n")
    return text, t, good


def _ablations(a: Path, exp: dict) -> list[tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    out = []
    root = a / "ablation"
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not (d / "ablation.json").exists():
            continue
        meta = json.loads((d / "ablation.json").read_text(encoding="utf-8"))
        if meta.get("data_fingerprint") != exp["data_fingerprint"] or meta.get("development"):
            continue          # not from this research run
        out.append((meta, pd.read_csv(d / "comparison.csv"), pd.read_csv(d / "paired_deltas.csv"),
                    pd.read_csv(d / "registration_timing.csv")))
    return out


def ablation_text(meta, comp, deltas) -> str:
    lines = [f"Dropping {', '.join(meta['dropped_features'])} (same temporal split, grids, seed and config):\n\n"]
    for name in comp["model"].unique():
        if name in BASELINES:
            continue
        dd = deltas[(deltas["model"] == name)].set_index("metric")
        c = comp[comp["model"] == name].set_index("variant")
        lines.append(f"- {name}: ROC-AUC {f(c.loc['full (saved primary)', 'roc_auc'])} -> {f(c.loc['ablated', 'roc_auc'])} "
                     f"(delta {f(dd.loc['roc_auc', 'delta_ablated_minus_full'])}, 95% CI {f(dd.loc['roc_auc', 'ci_low'])} to "
                     f"{f(dd.loc['roc_auc', 'ci_high'])}); PR-AUC delta {f(dd.loc['pr_auc', 'delta_ablated_minus_full'])} "
                     f"({f(dd.loc['pr_auc', 'ci_low'])} to {f(dd.loc['pr_auc', 'ci_high'])}); Brier delta "
                     f"{f(dd.loc['brier', 'delta_ablated_minus_full'])}; calibration slope "
                     f"{f(c.loc['full (saved primary)', 'calibration_slope'], 3)} -> {f(c.loc['ablated', 'calibration_slope'], 3)}.\n")
    return "".join(lines)


def publish(artifacts_dir: Path, docs_dir: Path, processed_dir: Path | None = None) -> list[Path]:
    a = Path(artifacts_dir)
    eda_meta = json.loads((a / "eda" / "eda_metadata.json").read_text(encoding="utf-8"))
    exp = json.loads((a / "models" / "experiment.json").read_text(encoding="utf-8"))
    _check_publishable(eda_meta, exp)
    proc = load_processed(processed_dir)
    bi = proc.get("build_info")
    if bi is not None and bi.get("raw_run_id") != exp["source"].get("raw_run_id"):
        raise ValueError("Processed build and model artifacts come from different raw runs.")
    m = a / "models"
    metrics = pd.read_csv(m / "metrics_test.csv")
    boot = pd.read_csv(m / "bootstrap.csv")
    subgroups = flag_stability(pd.read_csv(m / "subgroups.csv"))
    coefs = pd.read_csv(m / "logistic_coefficients.csv")
    perm = pd.read_csv(m / "permutation_importance.csv")
    sens = pd.read_csv(m / "sensitivity.csv")
    preds = pd.read_csv(m / "predictions_test.csv")
    calib = pd.read_csv(m / "calibration_bins.csv")
    assoc = pd.read_csv(a / "eda" / "eda_associations.csv")
    levels = pd.read_csv(a / "eda" / "eda_levels.csv")
    src = (f"raw run `{exp['source'].get('raw_run_id')}`, build {exp['source'].get('built_at_utc')}, "
           f"cohort fingerprint `{exp['data_fingerprint'][:16]}`")
    interp, interp_table, good = model_interpretation(metrics, boot)
    unk_text, unk_table = unknown_analysis(proc)
    ablations = _ablations(a, exp)
    mcols = ["roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "prevalence", "threshold",
             "calibration_intercept", "calibration_slope", "cv_pr_auc_train"]
    met = _r(metrics[["model", "n", "n_positive"] + mcols + ["tn", "fp", "fn", "tp"]], mcols)

    fail_rows = []
    top = metrics[~metrics["model"].isin(BASELINES)].sort_values("pr_auc", ascending=False)["model"].iloc[0]
    for name in dict.fromkeys(["logistic", top]):
        thr = float(metrics.set_index("model").loc[name, "threshold"])
        pred = (preds[f"p_{name}"] >= thr).astype(int)
        for col in ("phase_group", "lead_sponsor_class"):
            for lvl, g in preds.assign(pred=pred).groupby(col):
                pos, neg = g[g["target_terminated"] == 1], g[g["target_terminated"] == 0]
                fail_rows.append({"model": name, "subgroup": col, "level": lvl, "n": len(g), "n_positive": len(pos),
                                  "false_negative_rate": round(float((pos["pred"] == 0).mean()), 4) if len(pos) else None,
                                  "false_positive_rate": round(float((neg["pred"] == 1).mean()), 4) if len(neg) else None})
    fail = flag_stability(pd.DataFrame(fail_rows))
    weighted = [n for n, bp in zip(sens["model"], sens.get("best_params", pd.Series(dtype=str)).astype(str))
                if "balanced" in bp]
    sens_note = (f"\nNote: in the sensitivity runs, tuning (scored on PR-AUC) selected class weighting for "
                 f"{', '.join(sorted(set(weighted)))}; class weighting inflates predicted probabilities, so those Brier "
                 "scores and calibration values are not comparable with the primary run.\n") if weighted else ""
    sg_note = (f"\nRows marked UNSTABLE have fewer than {MIN_EVENTS_STABLE} terminated or completed studies; their "
               "metrics are reported for completeness and should not be interpreted. n/a = undefined (one class only).\n")

    abl_md = ""
    for meta, comp, deltas, timing in ablations:
        cc = comp.copy()
        for col in cc.columns[2:]:
            cc[col] = pd.to_numeric(cc[col], errors="coerce").round(4)
        abl_md += ("\n## Feature ablation\n\n" + ablation_text(meta, comp, deltas) +
                   "\n" + md_table(cc) + "\nPaired differences (ablated minus full, identical resamples):\n\n" +
                   md_table(_r(deltas, ["delta_ablated_minus_full", "ci_low", "ci_high", "share_resamples_ablated_worse"]),
                            max_rows=40) + "\nRegistration timing (descriptive; completion dates used for diagnosis only):\n\n" +
                   md_table(timing))
    if not ablations:
        abl_md = "\n## Feature ablation\n\nNo ablation results from this research run were found.\n"

    eval_md = "".join([
        "# Evaluation report\n\n_Generated by `trialsight publish-docs` from saved artifacts. Source: ", src, "._\n\n",
        "## How to read these results\n\n", interp,
        "- Results describe retrospective associations in current-version registry records, within a cohort that "
        "conditions on reaching a final status. They are not evidence of prospective predictive validity.\n\n",
        "## Experiments\n\n",
        f"- Primary: temporal hold-out (test = studies starting on/after {exp['split']}); train n={exp['n_train']} "
        f"({exp['train_positive']} terminated), test n={exp['n_test']} ({exp['test_positive']} terminated).\n",
        "- Tuning: stratified K-fold CV on training data only (scoring: PR-AUC). Thresholds: max-F1 on training "
        "out-of-fold predictions. No resampling.\n",
        "- Sensitivity: stratified random split; adding time-varying-suspect fields; adding start_year.\n\n",
        "## Summary by model\n\n", md_table(_r(interp_table, ["roc_auc", "pr_auc", "brier", "precision", "recall"])),
        "\n## Test metrics\n\n", md_table(met),
        "\nn/a: calibration intercept/slope are undefined for constant-score baselines.\n",
        "\n## Bootstrap 95% CIs and paired differences vs the prevalence baseline\n\n",
        md_table(_r(boot, ["ci_low", "ci_high", "delta_vs_prevalence", "delta_ci_low", "delta_ci_high",
                           "share_resamples_better"]), max_rows=40),
        "\n## Calibration (quantile bins)\n\n", md_table(_r(calib, ["mean_predicted", "observed_rate"]), max_rows=70),
        "\n## Subgroup performance\n\n", md_table(_r(subgroups, ["prevalence", "roc_auc", "pr_auc", "brier", "recall",
                                                                  "precision"]), max_rows=80), sg_note,
        "\n## Failure analysis (error rates at the training-derived threshold)\n\n", md_table(fail), sg_note,
        "\n## Sensitivity experiments\n\n", md_table(_r(sens.drop(columns=["best_params"], errors="ignore"), mcols),
                                                     max_rows=20), sens_note,
        abl_md,
        "\n## Interpretation\n\nTop logistic coefficients (standardised / one-hot scale):\n\n",
        md_table(_r(coefs.head(20), ["coefficient", "odds_ratio"])),
        "\nPermutation importance on the test set (mean PR-AUC drop):\n\n",
        md_table(_r(perm.groupby("model").head(10), ["importance_mean_pr_auc_drop", "importance_std"]), max_rows=40),
        "\n`registration_lag_days` (days from start to first registration) ranks among the most important features. "
        "Registration can occur after a study has started, or even ended, so this field may carry information that "
        "was not available at study start; see the feature ablation. All importances are predictive associations, "
        "not causal effects.\n",
    ])

    card_md = "".join([
        "# Model card\n\n_Generated from saved artifacts. Source: ", src, "._\n\n",
        "## Purpose\nDescribe retrospective associations between registry characteristics and whether a Phase II/III "
        "oncology trial was completed or terminated. Positive class: TERMINATED.\n\n",
        f"## Training population\nTrain n={exp['n_train']} ({exp['train_positive']} terminated), test n={exp['n_test']} "
        f"({exp['test_positive']} terminated); temporal cutoff {exp['split']}. Inclusion criteria: METHODOLOGY.md.\n\n",
        "## Features\nCategorical: " + ", ".join(exp["features_categorical"]) + ".\n\nNumeric: " +
        ", ".join(exp["features_numeric"]) + ".\n\nExcluded: post-outcome fields (status, why_stopped, enrollment, "
        "completion dates, results), identifiers, time-varying-suspect fields (sensitivity only), start_year.\n\n",
        "## Performance (test set)\n\n", interp, "\n", md_table(_r(interp_table, ["roc_auc", "pr_auc", "brier",
                                                                                "precision", "recall"])),
        (f"\nReference models for probability estimates (calibrated and better than baseline on Brier): "
         f"{', '.join(good)}.\n" if good else "\n"),
        "\n## Cohort-definition dependence\n\n", unk_text,
        "".join("\n## Feature ablation\n\n" + ablation_text(mt, c, d) for mt, c, d, _ in ablations),
        "\n## Limitations\n- Current-version records; no historical snapshots, so not prospective validation.\n"
        "- Selection bias: the cohort conditions on reaching COMPLETED/TERMINATED within a resolved window; "
        "unresolved, UNKNOWN and WITHDRAWN studies are excluded.\n"
        "- Sponsor-reported statuses; 'terminated' mixes safety, futility, accrual, business and early-success reasons.\n"
        "- Modest discrimination and high false-positive rates (above).\n\n",
        "## Intended use\nResearch and teaching about registry-level patterns.\n\n",
        "## Inappropriate uses\nForecasting the fate of a specific trial; funding, regulatory, ethics or clinical "
        "decisions; any statement about patient risk.\n",
    ])

    sig = assoc[assoc["significant_fdr"].astype(str).str.lower() == "true"].copy()
    sig = sig.reindex(sig["effect_size"].abs().sort_values(ascending=False).index)
    maxeff = sig["effect_size"].abs().max() if len(sig) else float("nan")
    findings = ["# Research findings\n\n_Generated from saved artifacts. Source: ", src, "._\n\n",
                f"Cohort: n={eda_meta['n']}, terminated {eda_meta['n_terminated']} "
                f"({eda_meta['termination_rate']:.1%}), start years {eda_meta['year_min']}-{eda_meta['year_max']}. "
                f"{eda_meta['n_tests']} tests; Benjamini-Hochberg FDR 5%; {len(sig)} significant.\n\n",
                "## Observations\n\n",
                (f"Every significant association is at most {magnitude(maxeff)} (largest |effect| = {f(maxeff)}). "
                 "With n in the thousands, tests detect effects too small to matter for any individual trial.\n\n"
                 if len(sig) else "")]
    for i, r in enumerate(sig.head(10).itertuples(), 1):
        if r.family == "categorical":
            lv = levels[levels["column"] == r.column]
            detail = "; ".join(f"{x.level}: {f(x.termination_rate, 3)} [{f(x.ci_low, 3)}, {f(x.ci_high, 3)}] (n={x.n}"
                               f"{', small n - unstable' if x.n < 30 else ''})" for x in lv.itertuples())
            ev = "`eda_associations.csv`, `eda_levels.csv`"
        elif r.family == "numeric":
            detail = (f"median terminated {f(r.median_terminated)} vs completed {f(r.median_completed)}; median "
                      f"difference {f(r.median_diff)} (95% CI {f(r.median_diff_ci_low)} to {f(r.median_diff_ci_high)})")
            ev = "`eda_associations.csv`"
        else:
            detail = (f"missing in {f(r.missing_share_terminated, 3)} of terminated vs "
                      f"{f(r.missing_share_completed, 3)} of completed studies")
            ev = "`eda_associations.csv`"
        caution = (" Time-varying field: may have been updated after the outcome." if r.availability ==
                   "time_varying_suspect" else "")
        findings.append(f"{i}. **{r.column}** ({r.family}; {r.availability}) - {r.effect_measure} = {f(r.effect_size)} "
                        f"({magnitude(r.effect_size)}), q = {fmt_cell(float(r.q_value))}. {detail}. Evidence: {ev}.{caution}\n")
    if sig.empty:
        findings.append("No association was significant after FDR correction.\n")
    tsy = proc.get("terminated_share_by_year")
    if tsy is not None:
        inw = tsy[tsy["start_year"] <= eda_meta["year_max"]]
        late = tsy[tsy["start_year"] > eda_meta["year_max"] + 2]
        findings.append(
            f"\n## Right-censoring\nWithin the window the terminated share per start year ranges {f(inw['terminated_share'].min(), 3)}-"
            f"{f(inw['terminated_share'].max(), 3)}; for start years after {eda_meta['year_max'] + 2} it ranges "
            f"{f(late['terminated_share'].min(), 3)}-{f(late['terminated_share'].max(), 3)}, consistent with recent "
            "studies that ended early being over-represented among those already finished.\n")
    findings += ["\n## Cohort window and UNKNOWN statuses\n\n", unk_text]
    onc = proc.get("oncology_capture_summary")
    if onc is not None:
        o = onc.set_index("category")
        k = "excluded: cancer text, no MeSH annotations"
        if k in o.index:
            r = o.loc[k]
            res_n = int(r["n_completed"] + r["n_terminated"])
            findings.append(f"\n## Oncology definition\nStrict MeSH matching excluded {int(r['n'])} studies with cancer-related "
                            f"condition text but no MeSH annotations ({f(r['share_of_pre_oncology'], 3)} of the pre-oncology "
                            f"pool, all years); among those with a final status, {f(r['n_terminated'] / res_n, 3) if res_n else 'n/a'} "
                            "were terminated. See the cohort validation report.\n")
    findings += ["\n## Model evidence\n\n", interp,
                 "".join("\n" + ablation_text(mt, c, d) for mt, c, d, _ in ablations),
                 "\n## Practical implications\nPatterns describe registry records, not trial-design advice; the models "
                 "should not be used to flag individual trials.\n\n",
                 "## Future research questions\n- Do results hold with historical record versions?\n"
                 "- Time-to-event analysis including unresolved trials.\n"
                 "- Are termination reasons (why_stopped) associated with distinct design profiles?\n"]
    outs = []
    for name, text in (("EVALUATION_REPORT.md", eval_md), ("MODEL_CARD.md", card_md),
                       ("RESEARCH_FINDINGS.md", "".join(findings))):
        path = Path(docs_dir) / name
        path.write_text(text, encoding="utf-8")
        outs.append(path)
    return outs
