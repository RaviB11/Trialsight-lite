"""Exploratory statistical analysis of the classification cohort.

Every number written here is computed from the cohort table. Associations are
retrospective and non-causal. Multiple testing is controlled with
Benjamini-Hochberg FDR across ALL tests run in one EDA execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .cohort import wilson_interval
from .features import (PRIMARY_CATEGORICAL, PRIMARY_NUMERIC, SUSPECT_NUMERIC, TARGET,
                       add_derived_features, as_categorical_strings, availability)
from .report import md_table
from .storage import export_parquet, read_table, write_tables
from .study_data import load_study_data

log = logging.getLogger(__name__)

EDA_CATEGORICAL = PRIMARY_CATEGORICAL + ["start_year"]
EDA_NUMERIC = PRIMARY_NUMERIC + SUSPECT_NUMERIC


def cramers_v_corrected(table: np.ndarray, chi2: float) -> float:
    n = table.sum()
    r, k = table.shape
    if n <= 1 or min(r, k) < 2:
        return float("nan")
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def categorical_associations(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per column: chi-square (Fisher for sparse 2x2) + bias-corrected Cramer's V.
    Per level: n, terminated, rate with Wilson 95% CI."""
    tests, levels = [], []
    d = as_categorical_strings(df, columns)
    for col in columns:
        tab = pd.crosstab(d[col], df[TARGET])
        for c in (0, 1):
            if c not in tab.columns:
                tab[c] = 0
        tab = tab[[0, 1]]
        n_lvl = tab.sum(axis=1)
        lo, hi = wilson_interval(tab[1], n_lvl)
        for lvl in tab.index:
            levels.append({"column": col, "level": lvl, "n": int(n_lvl[lvl]), "n_terminated": int(tab.loc[lvl, 1]),
                           "termination_rate": round(tab.loc[lvl, 1] / n_lvl[lvl], 4) if n_lvl[lvl] else None,
                           "ci_low": lo[lvl], "ci_high": hi[lvl]})
        row: dict[str, Any] = {"family": "categorical", "column": col, "availability": availability(col),
                               "n": int(tab.values.sum()), "n_levels": int(len(tab))}
        if len(tab) < 2:
            row.update(test="not tested (single level)", statistic=None, p_value=None, effect_size=None,
                       effect_measure=None, note="")
        else:
            chi2, p, dof, expected = stats.chi2_contingency(tab.values, correction=False)
            sparse = float((expected < 5).mean())
            if tab.shape == (2, 2) and sparse > 0:
                _, p = stats.fisher_exact(tab.values)
                test = "Fisher exact"
            else:
                test = f"chi-square (dof={dof})"
            row.update(test=test, statistic=round(float(chi2), 4), p_value=float(p),
                       effect_size=round(cramers_v_corrected(tab.values, chi2), 4),
                       effect_measure="Cramer's V (bias-corrected)",
                       note=f"{sparse:.0%} of expected counts < 5" if sparse > 0 and test.startswith("chi") else "")
        tests.append(row)
    return pd.DataFrame(tests), pd.DataFrame(levels)


def numeric_associations(df: pd.DataFrame, columns: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    """Mann-Whitney U, rank-biserial correlation (positive = higher in terminated),
    median difference (terminated - completed) with percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    rows = []
    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce")
        t, c = x[df[TARGET] == 1].dropna().to_numpy(), x[df[TARGET] == 0].dropna().to_numpy()
        row: dict[str, Any] = {"family": "numeric", "column": col, "availability": availability(col),
                               "n": int(len(t) + len(c)), "missing_share": round(float(x.isna().mean()), 4),
                               "median_terminated": float(np.median(t)) if len(t) else None,
                               "median_completed": float(np.median(c)) if len(c) else None,
                               "iqr_terminated": f"{np.percentile(t, 25):g}-{np.percentile(t, 75):g}" if len(t) else "",
                               "iqr_completed": f"{np.percentile(c, 25):g}-{np.percentile(c, 75):g}" if len(c) else ""}
        if len(t) < 2 or len(c) < 2 or (np.ptp(np.concatenate([t, c])) == 0):
            row.update(test="not tested", statistic=None, p_value=None, effect_size=None,
                       effect_measure=None, median_diff=None, median_diff_ci_low=None, median_diff_ci_high=None)
        else:
            u, p = stats.mannwhitneyu(t, c, alternative="two-sided")
            boots = np.empty(n_boot)
            for b in range(n_boot):
                boots[b] = np.median(rng.choice(t, len(t))) - np.median(rng.choice(c, len(c)))
            row.update(test="Mann-Whitney U", statistic=float(u), p_value=float(p),
                       effect_size=round(2 * u / (len(t) * len(c)) - 1, 4), effect_measure="rank-biserial r",
                       median_diff=float(np.median(t) - np.median(c)),
                       median_diff_ci_low=float(np.percentile(boots, 2.5)),
                       median_diff_ci_high=float(np.percentile(boots, 97.5)))
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_associations(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        miss = df[col].isna()
        row = {"family": "missingness", "column": col, "availability": availability(col), "n": len(df),
               "missing_share": round(float(miss.mean()), 4),
               "missing_share_terminated": round(float(miss[df[TARGET] == 1].mean()), 4),
               "missing_share_completed": round(float(miss[df[TARGET] == 0].mean()), 4)}
        if 0 < miss.sum() < len(df):
            tab = pd.crosstab(miss, df[TARGET]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
            chi2, p, _, expected = stats.chi2_contingency(tab.values, correction=False)
            if (expected < 5).any():
                _, p = stats.fisher_exact(tab.values)
            row.update(test="missing vs outcome", p_value=float(p),
                       effect_size=round(cramers_v_corrected(tab.values, chi2), 4), effect_measure="Cramer's V")
        else:
            row.update(test="not tested (no variation)", p_value=None, effect_size=None, effect_measure=None)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_fdr(frames: list[pd.DataFrame], alpha: float = 0.05) -> list[pd.DataFrame]:
    """Benjamini-Hochberg across every p-value in all frames (one family per EDA run)."""
    pvals = [(i, j, p) for i, f in enumerate(frames) for j, p in zip(f.index, f["p_value"]) if pd.notna(p)]
    out = [f.assign(q_value=np.nan, significant_fdr=False) for f in frames]
    if pvals:
        reject, q, _, _ = multipletests([p for _, _, p in pvals], alpha=alpha, method="fdr_bh")
        for (i, j, _), qq, rj in zip(pvals, q, reject):
            out[i].loc[j, "q_value"] = float(qq)
            out[i].loc[j, "significant_fdr"] = bool(rj)
    return out


def therapeutic_areas(df: pd.DataFrame, mesh: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Descriptive termination rates by MeSH ancestor term (multi-label; not tested)."""
    m = mesh[(mesh["kind"] == "ancestor") & mesh["nct_id"].isin(df["nct_id"]) & (mesh["term"] != "Neoplasms")]
    m = m.drop_duplicates(["nct_id", "term"]).merge(df[["nct_id", TARGET]], on="nct_id")
    g = m.groupby("term")[TARGET].agg(n="size", n_terminated="sum").reset_index()
    g = g[g["n"] >= min_n].copy()
    g["termination_rate"] = (g["n_terminated"] / g["n"]).round(4)
    g["ci_low"], g["ci_high"] = wilson_interval(g["n_terminated"], g["n"])
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def _figures_html(df: pd.DataFrame, levels: pd.DataFrame, missing: pd.DataFrame, title: str) -> str:
    import plotly.express as px
    import plotly.graph_objects as go
    figs = []
    for col in ["phase_group", "lead_sponsor_class", "primary_purpose", "allocation", "start_year"]:
        lv = levels[levels["column"] == col].sort_values("level")
        if lv.empty:
            continue
        fig = go.Figure(go.Bar(
            x=lv["level"].astype(str), y=lv["termination_rate"],
            error_y=dict(type="data", symmetric=False, array=lv["ci_high"] - lv["termination_rate"],
                         arrayminus=lv["termination_rate"] - lv["ci_low"]),
            customdata=lv["n"], hovertemplate="%{x}<br>rate %{y:.3f}<br>n=%{customdata}<extra></extra>"))
        fig.update_layout(title=f"Termination rate by {col} (Wilson 95% CI)", yaxis_title="share terminated",
                          template="plotly_white", height=380)
        figs.append(fig)
    ms = missing.sort_values("missing_share", ascending=False)
    figs.append(px.bar(ms, x="column", y="missing_share", title="Missing share by field (cohort)",
                       template="plotly_white", height=380))
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body><h1>{title}</h1>"]
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    return "\n".join(parts)


def _summary_md(meta: dict, assoc: pd.DataFrame, levels: pd.DataFrame, areas: pd.DataFrame) -> str:
    banner = "" if meta["population_status"] in ("research_verified", "research_reconciled") else (
        "> **DEVELOPMENT DATA - NOT RESEARCH FINDINGS.** Computed on a non-validated build.\n\n")
    sig = assoc[assoc["significant_fdr"] == True].sort_values("effect_size", key=lambda s: s.abs(), ascending=False)  # noqa: E712
    lines = [
        "# Exploratory analysis summary (auto-generated)\n", banner,
        f"- Source build: `{meta['raw_run_id']}`, population status **{meta['population_status']}**\n",
        f"- Cohort: n={meta['n']}, terminated={meta['n_terminated']} ({meta['termination_rate']:.1%}), "
        f"start years {meta['year_min']}-{meta['year_max']}\n",
        f"- Tests run: {int(assoc['p_value'].notna().sum())}; Benjamini-Hochberg FDR at 5% across all tests; "
        f"{len(sig)} significant after correction.\n",
        "- These are retrospective associations in registry metadata. They are not causal, and fields marked "
        "time_varying_suspect may have been amended after study start.\n\n",
        "## Associations significant after FDR correction (sorted by |effect size|)\n\n",
        md_table(sig[["family", "column", "availability", "test", "effect_measure", "effect_size", "p_value", "q_value"]]),
        "\n## Termination rate by level (Wilson 95% CI)\n\n",
        md_table(levels[levels["column"].isin(["phase_group", "lead_sponsor_class", "primary_purpose", "allocation"])]),
        "\n## Therapeutic areas (MeSH ancestors; descriptive, overlapping, not tested)\n\n", md_table(areas.head(25)),
    ]
    return "".join(lines)


def run_eda(db_path: Path, out_dir: Path, *, development: bool = False, n_boot: int = 1000, seed: int = 20261012,
            min_area_n: int = 30) -> dict[str, Any]:
    data = load_study_data(db_path, development=development)
    df = add_derived_features(data.cohort)
    if df[TARGET].nunique() < 2:
        raise ValueError("Cohort must contain both outcome classes for association analysis.")
    cat_tests, levels = categorical_associations(df, EDA_CATEGORICAL)
    num_tests = numeric_associations(df, EDA_NUMERIC, n_boot=n_boot, seed=seed)
    miss_tests = missingness_associations(df, EDA_CATEGORICAL + EDA_NUMERIC)
    cat_tests, num_tests, miss_tests = apply_fdr([cat_tests, num_tests, miss_tests])
    assoc = pd.concat([cat_tests, num_tests, miss_tests], ignore_index=True, sort=False)
    areas = therapeutic_areas(df, data.mesh, min_area_n)
    conditions = (data.conditions[(data.conditions["kind"] == "condition") & data.conditions["nct_id"].isin(df["nct_id"])]
                  .groupby("term").size().rename("n").sort_values(ascending=False).head(50).reset_index())
    by_year = (df.groupby("start_year")[TARGET].agg(n="size", n_terminated="sum").reset_index())
    meta = {**data.metadata_summary, "n": int(len(df)), "n_terminated": int(df[TARGET].sum()),
            "termination_rate": float(df[TARGET].mean()), "year_min": int(df["start_year"].min()),
            "year_max": int(df["start_year"].max()), "n_tests": int(assoc["p_value"].notna().sum()),
            "fdr_alpha": 0.05, "bootstrap_resamples": n_boot, "seed": seed, "development": development}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"eda_associations": assoc, "eda_levels": levels, "eda_therapeutic_areas": areas,
              "eda_top_conditions": conditions, "eda_by_start_year": by_year}
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    write_tables(out_dir / "eda.duckdb", tables)
    export_parquet(out_dir / "eda.duckdb", list(tables), out_dir / "parquet")
    (out_dir / "eda_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "eda_summary.md").write_text(_summary_md(meta, assoc, levels, areas), encoding="utf-8")
    title = "TrialSight Lite EDA" + (" - DEVELOPMENT DATA" if development else "")
    (out_dir / "eda_figures.html").write_text(_figures_html(df, levels, miss_tests, title), encoding="utf-8")
    return meta
