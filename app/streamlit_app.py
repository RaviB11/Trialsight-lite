"""TrialSight Lite research application. Run: streamlit run app/streamlit_app.py
Every statistic shown is read from the processed DuckDB build or saved artifacts."""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trialsight.app_data import load_app_state
from trialsight.config import PROJECT_ROOT
from trialsight.report import MIN_EVENTS_STABLE, flag_stability, fmt_cell

st.set_page_config(page_title="TrialSight Lite", page_icon="🔬", layout="wide")


def show_df(df, **kwargs):
    """Display with explicit 'n/a' for undefined values (never 'nan'/'None')."""
    if df is None:
        df = pd.DataFrame()
    st.dataframe(df.astype(object).map(fmt_cell), **kwargs)
CONFIG = os.environ.get("TRIALSIGHT_CONFIG")


@st.cache_data(show_spinner="Loading processed data…")
def get_state(config_path, allow_dev):
    return load_app_state(config_path, allow_development=allow_dev)


st.sidebar.title("TrialSight Lite")
allow_dev = st.sidebar.checkbox("Show development build if no research build exists", value=False,
                                help="Development builds come from incomplete or unverified downloads.")
state = get_state(CONFIG, allow_dev)

if state.mode == "unavailable":
    st.title("TrialSight Lite")
    st.error(f"No accepted research build is available.\n\n{state.message}")
    st.markdown("Run the pipeline first: `trialsight fetch`, `trialsight build`, `trialsight eda`, `trialsight train`. "
                "See README.md.")
    st.stop()

if state.mode == "development":
    st.warning("**DEVELOPMENT BUILD - NOT A RESEARCH POPULATION.** Figures below describe an incomplete or "
               "unverified download and must not be reported as findings.", icon="⚠️")
for w in state.warnings:
    st.warning(w)

meta, t = state.metadata, state.tables
cohort = t.get("cohort", pd.DataFrame())
eligible = t.get("eligible_population", pd.DataFrame())
st.sidebar.caption(f"Raw run `{meta.get('raw_run_id')}`  \nBuilt {str(meta.get('built_at_utc'))[:19]}  \n"
                   f"Status: {meta.get('population_status')}")
with st.sidebar.expander("Provenance"):
    st.markdown("\n".join(f"- **{k}**: `{v}`" for k, v in state.provenance.items()))

tabs = st.tabs(["Overview", "Clinical Trial Explorer", "Statistical Research", "Machine Learning Evaluation",
                "Historical Study Analysis", "Research Findings", "Methodology & Limitations"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.title("Completion vs termination in Phase II/III oncology trials")
    st.markdown("A retrospective analysis of public **ClinicalTrials.gov (API v2)** registry metadata. "
                "Associations are not causal, and models describe historical patterns, not future-trial risk.")
    c = st.columns(4)
    c[0].metric("Classification cohort", f"{len(cohort):,}")
    if len(cohort):
        c[1].metric("Terminated share", f"{cohort['target_terminated'].mean():.1%}")
    c[2].metric("Analysis window", f"{meta.get('analysis_start_year')}–{meta.get('analysis_end_year')}")
    c[3].metric("Eligible studies (all statuses)", f"{len(eligible):,}")
    left, right = st.columns(2)
    with left:
        st.subheader("Cohort attrition")
        show_df(t.get("cohort_attrition", pd.DataFrame()), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Validation checks")
        show_df(t.get("cohort_validation", pd.DataFrame()), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Explorer
with tabs[1]:
    st.subheader("Eligible studies (all statuses, observed registry data)")
    if eligible.empty:
        st.info("No eligible-population table in this build.")
    else:
        f = st.columns(4)
        phases = f[0].multiselect("Phase", sorted(eligible["phase_group"].dropna().unique()))
        statuses = f[1].multiselect("Status", sorted(eligible["overall_status"].dropna().unique()))
        sponsors = f[2].multiselect("Sponsor class", sorted(eligible["lead_sponsor_class"].dropna().unique()))
        geo = f[3].selectbox("Geography", ["Any", "Has a US site", "No US site"])
        y0, y1 = int(eligible["start_year"].min()), int(eligible["start_year"].max())
        years = st.slider("Start year", y0, y1, (y0, y1)) if y1 > y0 else (y0, y1)
        d = eligible[eligible["start_year"].between(*years)]
        for col, sel in (("phase_group", phases), ("overall_status", statuses), ("lead_sponsor_class", sponsors)):
            if sel:
                d = d[d[col].isin(sel)]
        if geo != "Any" and "has_us_site" in d:
            d = d[d["has_us_site"] == (geo == "Has a US site")]
        st.caption(f"{len(d):,} studies match. Site fields are current-version and may have changed over time.")
        if len(d):
            counts = d.groupby(["start_year", "overall_status"]).size().rename("n").reset_index()
            st.plotly_chart(px.bar(counts, x="start_year", y="n", color="overall_status", template="plotly_white",
                                   title="Studies by start year and current status"), use_container_width=True)
            show = d[["nct_id", "brief_title", "overall_status", "phase_group", "lead_sponsor_class", "start_date"]].copy()
            show["link"] = "https://clinicaltrials.gov/study/" + show["nct_id"]
            show_df(show, hide_index=True, use_container_width=True,
                         column_config={"link": st.column_config.LinkColumn("Registry record")})

# ---------------------------------------------------------------- Statistics
with tabs[2]:
    st.subheader("Statistical research")
    if not state.eda:
        st.info("No EDA artifacts for this build. Run `trialsight eda`.")
    else:
        assoc, levels = state.eda["eda_associations"], state.eda["eda_levels"]
        st.caption(f"{state.eda_meta.get('n_tests')} tests; Benjamini–Hochberg FDR at 5% across all tests. "
                   "Effect sizes: Cramér's V (categorical), rank-biserial r (numeric).")
        only_sig = st.toggle("Only FDR-significant", value=True)
        a = assoc[assoc["significant_fdr"] == True] if only_sig else assoc  # noqa: E712
        show_df(a[["family", "column", "availability", "test", "effect_measure", "effect_size", "p_value",
                        "q_value"]], hide_index=True, use_container_width=True)
        col = st.selectbox("Termination rate by", sorted(levels["column"].unique()))
        lv = levels[levels["column"] == col].sort_values("level")
        small = lv[lv["n"] < 30]
        if len(small):
            st.caption("Small levels (n < 30), unstable estimates: " +
                       ", ".join(f"{x.level} (n={x.n})" for x in small.itertuples()))
        fig = go.Figure(go.Bar(x=lv["level"].astype(str), y=lv["termination_rate"],
                               error_y=dict(type="data", symmetric=False,
                                            array=lv["ci_high"] - lv["termination_rate"],
                                            arrayminus=lv["termination_rate"] - lv["ci_low"])))
        fig.update_layout(template="plotly_white", yaxis_title="share terminated (Wilson 95% CI)", title=col)
        st.plotly_chart(fig, use_container_width=True)
        if "threshold_sensitivity" in t:
            st.subheader("Window sensitivity to the resolution threshold")
            show_df(t["threshold_sensitivity"], hide_index=True, use_container_width=True)
        if "eda_therapeutic_areas" in state.eda:
            st.subheader("Therapeutic areas (MeSH ancestors; descriptive, overlapping)")
            show_df(state.eda["eda_therapeutic_areas"].head(30), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- ML
with tabs[3]:
    st.subheader("Model evaluation (retrospective, temporal hold-out)")
    if not state.models:
        st.info("No model artifacts for this build. Run `trialsight train`.")
    else:
        exp, m = state.experiment, state.models
        if exp.get("config", {}).get("fast"):
            st.warning("These models were trained in --fast smoke mode and are not reportable.")
        st.caption(f"Train n={exp.get('n_train')} ({exp.get('train_positive')} terminated) · test n={exp.get('n_test')} "
                   f"({exp.get('test_positive')} terminated) · test = starts on/after {exp.get('split')}. "
                   "Thresholds chosen on training out-of-fold predictions.")
        show_df(m["metrics_test"].drop(columns=["best_params"], errors="ignore"), hide_index=True,
                     use_container_width=True)
        cv = m["curves"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(cv[cv["curve"] == "roc"], x="x", y="y", color="model", template="plotly_white",
                                labels={"x": "False positive rate", "y": "True positive rate"}, title="ROC"),
                        use_container_width=True)
        c2.plotly_chart(px.line(cv[cv["curve"] == "pr"], x="x", y="y", color="model", template="plotly_white",
                                labels={"x": "Recall", "y": "Precision"}, title="Precision–recall"),
                        use_container_width=True)
        cal = px.line(m["calibration_bins"], x="mean_predicted", y="observed_rate", color="model", markers=True,
                      template="plotly_white", title="Calibration (quantile bins)")
        cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        c1.plotly_chart(cal, use_container_width=True)
        model = c2.selectbox("Confusion matrix for", m["metrics_test"]["model"].tolist(), index=2)
        r = m["metrics_test"].set_index("model").loc[model]
        cm = pd.DataFrame([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], index=["actual completed", "actual terminated"],
                          columns=["predicted completed", "predicted terminated"])
        c2.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"{model}"),
                        use_container_width=True)
        st.subheader("Uncertainty and comparison with the prevalence baseline")
        show_df(m["bootstrap"], hide_index=True, use_container_width=True)
        for name, ab in state.ablations.items():
            st.subheader(f"Feature ablation: without {name}")
            st.caption("Same temporal split, grids, seed and config as the primary run; deltas are ablated minus "
                       "full on identical bootstrap resamples (Brier: positive = worse).")
            show_df(ab["comparison"], hide_index=True, use_container_width=True)
            show_df(ab["paired_deltas"], hide_index=True, use_container_width=True)
        st.subheader("Interpretation")
        st.caption("Coefficients and importances are predictive associations, not causal effects.")
        c1, c2 = st.columns(2)
        c1.dataframe(m["logistic_coefficients"].head(25), hide_index=True, use_container_width=True)
        c2.dataframe(m["permutation_importance"].groupby("model").head(10), hide_index=True, use_container_width=True)
        st.subheader("Subgroups and sensitivity experiments")
        st.caption(f"Rows marked UNSTABLE have fewer than {MIN_EVENTS_STABLE} terminated or completed studies; "
                   "do not interpret their metrics. n/a = undefined (one class only).")
        show_df(flag_stability(m["subgroups"]), hide_index=True, use_container_width=True)
        show_df(m["sensitivity"], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Historical study
with tabs[4]:
    st.subheader("Inspect a real study record")
    if cohort.empty:
        st.info("No cohort in this build.")
    else:
        nct = st.selectbox("NCT ID", cohort["nct_id"].tolist())
        rec = cohort[cohort["nct_id"] == nct].iloc[0]
        st.markdown(f"**{rec['brief_title']}** · [{nct} on ClinicalTrials.gov](https://clinicaltrials.gov/study/{nct})")
        fields = ["overall_status", "phase_group", "lead_sponsor_class", "allocation", "primary_purpose", "masking",
                  "start_date", "n_arms", "n_interventions", "min_age_years", "max_age_years", "eligibility_criteria_chars"]
        show_df(rec[[f for f in fields if f in rec.index]].astype(str).rename("value"), use_container_width=True)
        nums = [f for f in ("n_arms", "n_interventions", "eligibility_criteria_chars") if f in cohort]
        pct = {f: float((cohort[f] <= rec[f]).mean()) for f in nums if pd.notna(rec[f])}
        if pct:
            st.caption("Percentile of this study within the cohort: " +
                       ", ".join(f"{k} {v:.0%}" for k, v in pct.items()))
        preds = state.models.get("predictions_test")
        if preds is not None and nct in set(preds["nct_id"]):
            row = preds[preds["nct_id"] == nct].iloc[0]
            st.info(f"Out-of-sample retrospective model score (logistic): {row['p_logistic']:.3f}. "
                    "This is a historical model output for a study in the temporal test set, not a validated "
                    "risk prediction.")
        else:
            st.caption("This study was in the training set (or no models exist); in-sample scores are not shown.")

# ---------------------------------------------------------------- Findings
with tabs[5]:
    st.subheader("Evidence-supported findings")
    if not state.eda:
        st.info("Findings appear after `trialsight eda` and `trialsight train` on the accepted research build.")
    else:
        assoc = state.eda["eda_associations"]
        sig = assoc[assoc["significant_fdr"] == True].copy()  # noqa: E712
        sig = sig.reindex(sig["effect_size"].abs().sort_values(ascending=False).index).head(10)
        if sig.empty:
            st.write("No association was significant after FDR correction.")
        for r in sig.itertuples():
            st.markdown(f"- **{r.column}** ({r.family}, {r.availability}): {r.effect_measure} = {r.effect_size:.3f}, "
                        f"q = {r.q_value:.3g}")
        if state.models:
            b = state.models["bootstrap"]
            pr = b[(b["metric"] == "pr_auc") & (b["delta_ci_low"] > 0)]
            st.markdown("**Models beating the prevalence baseline on PR-AUC (paired 95% CI above zero):** " +
                        (", ".join(pr["model"]) if len(pr) else "none"))

# ---------------------------------------------------------------- Methodology
with tabs[6]:
    path = PROJECT_ROOT / "docs" / "METHODOLOGY.md"
    st.markdown(path.read_text(encoding="utf-8") if path.exists() else "METHODOLOGY.md not found.")
