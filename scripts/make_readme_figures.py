"""Render README figures from SAVED research artifacts (never from synthetic data).

    python scripts/make_readme_figures.py            # defaults: artifacts/, data/processed/, docs/images/
Requires matplotlib (pip install -e ".[docs]"). Each figure records its source raw run.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = {"logistic": "Logistic regression", "random_forest": "Random forest",
          "hist_gradient_boosting": "Gradient boosting", "prevalence": "Prevalence baseline"}


def footer(fig, run):
    fig.text(0.01, 0.01, f"Source: ClinicalTrials.gov API v2, raw run {run}; computed by TrialSight Lite",
             fontsize=7, color="#666")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="docs/images")
    a = ap.parse_args()
    art, proc, out = Path(a.artifacts), Path(a.processed), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = json.loads((art / "models" / "experiment.json").read_text(encoding="utf-8"))
    if exp.get("development") or exp["config"].get("fast"):
        raise SystemExit("Refusing: artifacts are from a development or --fast run.")
    run = exp["source"]["raw_run_id"]
    metrics = pd.read_csv(art / "models" / "metrics_test.csv").set_index("model")
    curves = pd.read_csv(art / "models" / "curves.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for m, label in MODELS.items():
        c = curves[(curves.model == m) & (curves.curve == "roc")]
        ax[0].plot(c.x, c.y, label=f"{label} (AUC {metrics.loc[m, 'roc_auc']:.3f})", lw=1.4 if m != "prevalence" else 1,
                   ls="--" if m == "prevalence" else "-")
        c = curves[(curves.model == m) & (curves.curve == "pr")]
        if m == "prevalence":   # a constant score has no PR curve: its baseline is precision = prevalence
            ax[1].axhline(metrics.loc[m, "prevalence"], ls="--", lw=1, color="C3",
                          label=f"{label} (precision = prevalence {metrics.loc[m, 'prevalence']:.3f})")
        else:
            ax[1].plot(c.x, c.y, label=f"{label} (AP {metrics.loc[m, 'pr_auc']:.3f})", lw=1.4)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, temporal test set")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, temporal test set", ylim=(0, 1))
    for x in ax:
        x.legend(fontsize=8, loc="lower right" if x is ax[0] else "upper right")
        x.grid(alpha=0.3)
    footer(fig, run)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "model_discrimination.png", dpi=150)
    plt.close(fig)

    cal = pd.read_csv(art / "models" / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for m in ("logistic", "random_forest", "hist_gradient_boosting"):
        c = cal[cal.model == m]
        ax.plot(c.mean_predicted, c.observed_rate, marker="o", label=f"{MODELS[m]} (slope {metrics.loc[m, 'calibration_slope']:.2f})")
    ax.plot([0, 0.6], [0, 0.6], ls=":", color="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed termination rate", title="Calibration (quantile bins)",
           xlim=(0, 0.6), ylim=(0, 0.6))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    footer(fig, run)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)

    res = pd.read_csv(proc / "resolution_by_start_year.csv")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    tot = res.n_total
    ax.bar(res.start_year, res.n_resolved / tot, label="Resolved (completed/terminated/withdrawn)", color="#4C78A8")
    ax.bar(res.start_year, res.n_ambiguous / tot, bottom=res.n_resolved / tot, label="UNKNOWN status", color="#F58518")
    ax.bar(res.start_year, res.n_unresolved / tot, bottom=(res.n_resolved + res.n_ambiguous) / tot,
           label="Unresolved (recruiting/active/...)", color="#BAB0AC")
    ax.plot(res.start_year, res.resolved_share, "k.-", label="Resolved share (UNKNOWN excluded)")
    ax.axhline(0.9, color="red", ls="--", lw=1, label="90% rule")
    inw = res[res.in_analysis_window.astype(str).str.lower() == "true"].start_year
    ax.axvspan(inw.min() - 0.5, inw.max() + 0.5, color="green", alpha=0.08, label="Accepted window")
    ax.set(xlabel="Start year", ylabel="Share of eligible studies", title="Study resolution by start year")
    ax.set_xticks(res.start_year)
    ax.set_xticklabels(res.start_year.astype(int), rotation=45)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    footer(fig, run)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "resolution_by_year.png", dpi=150)
    plt.close(fig)

    lv = pd.read_csv(art / "eda" / "eda_levels.csv")
    lv = lv[lv.column == "phase_group"].sort_values("level")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(lv.level, lv.termination_rate, yerr=[lv.termination_rate - lv.ci_low, lv.ci_high - lv.termination_rate],
           capsize=4, color="#4C78A8")
    for x, (r, n) in enumerate(zip(lv.termination_rate, lv.n)):
        ax.text(x, 0.01, f"n={n}", ha="center", color="white", fontsize=8)
    ax.set(ylabel="Share terminated (Wilson 95% CI)", title="Termination rate by phase (cohort 2008-2014)")
    footer(fig, run)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "termination_by_phase.png", dpi=150)
    plt.close(fig)
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
