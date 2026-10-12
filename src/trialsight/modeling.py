"""Retrospective classification of COMPLETED (0) vs TERMINATED (1) studies.

Design (see docs/METHODOLOGY.md):
- Positive class = TERMINATED. Prevalence is reported, never assumed.
- Primary split is TEMPORAL: the latest `test_fraction` of the cohort by start
  date is held out. Hyperparameters are tuned by stratified K-fold CV on the
  training set only; decision thresholds come from out-of-fold training
  predictions (max F1), never from the test set.
- Imbalance: class_weight is a tuned option; no resampling anywhere.
- Primary features exclude time-varying-suspect fields and start_year
  (a follow-up/censoring proxy); both are tested in labelled sensitivity runs.
- Each NCT ID appears once in the cohort; train/test disjointness is asserted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import TARGET, add_derived_features, as_categorical_strings, availability, feature_set
from .report import md_table
from .study_data import load_study_data

log = logging.getLogger(__name__)
MODEL_ORDER = ["majority", "prevalence", "logistic", "lda", "random_forest", "hist_gradient_boosting"]


class InsufficientDataError(ValueError):
    """The cohort cannot support meaningful evaluation."""


@dataclass(frozen=True)
class ModelingConfig:
    seed: int = 20261012
    test_fraction: float = 0.25
    cv_folds: int = 5
    n_bootstrap: int = 1000
    permutation_repeats: int = 10
    min_events_train: int = 50
    min_events_test: int = 30
    fast: bool = False          # smaller grids; for smoke tests only, never for reported results


# ------------------------------------------------------------------ data
def prepare_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_features(cohort)
    if df["nct_id"].duplicated().any():
        raise ValueError("Duplicate nct_id in cohort; refuse to model.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be 0/1 for every cohort row.")
    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values(["start_date", "nct_id"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Latest studies by start date form the test set; ties at the cutoff date go to test."""
    cutoff = df["start_date"].iloc[int(np.floor(len(df) * (1 - test_fraction)))]
    train, test = df[df["start_date"] < cutoff], df[df["start_date"] >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), str(pd.Timestamp(cutoff).date())


def random_split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tr, te = train_test_split(df, test_size=test_fraction, stratify=df[TARGET], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True), "stratified random"


def check_split(train: pd.DataFrame, test: pd.DataFrame, cfg: ModelingConfig) -> None:
    overlap = set(train["nct_id"]) & set(test["nct_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} NCT IDs appear in both train and test.")
    for name, part, need in (("train", train, cfg.min_events_train), ("test", test, cfg.min_events_test)):
        counts = part[TARGET].value_counts()
        if min(counts.get(0, 0), counts.get(1, 0)) < need:
            raise InsufficientDataError(
                f"{name} split has {counts.get(1, 0)} terminated / {counts.get(0, 0)} completed; "
                f"at least {need} of each are required for meaningful evaluation.")


# ------------------------------------------------------------------ models
def build_preprocessor(cat: list[str], num: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="(missing)")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                   sparse_output=False))]), cat),
    ], verbose_feature_names_out=True)


def model_specs(seed: int, fast: bool) -> dict[str, tuple[Any, dict]]:
    rf_trees = 100 if fast else 400
    specs = {
        "majority": (DummyClassifier(strategy="most_frequent"), {}),
        "prevalence": (DummyClassifier(strategy="prior"), {}),
        "logistic": (LogisticRegression(max_iter=5000, random_state=seed),
                     {"model__C": [0.1, 1.0] if fast else [0.01, 0.1, 1.0, 10.0],
                      "model__class_weight": [None, "balanced"]}),
        "lda": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}),
        "random_forest": (RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=1),
                          {"model__min_samples_leaf": [10] if fast else [5, 20],
                           "model__max_features": ["sqrt"] if fast else ["sqrt", 0.3],
                           "model__class_weight": [None, "balanced_subsample"]}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=seed, early_stopping=False,
                                                                   max_iter=100 if fast else 300),
                                   {"model__learning_rate": [0.1] if fast else [0.03, 0.1],
                                    "model__max_leaf_nodes": [15] if fast else [15, 31],
                                    "model__l2_regularization": [0.0, 1.0]}),
    }
    return specs


def make_pipeline(estimator, cat, num) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(cat, num)), ("var", VarianceThreshold()), ("model", estimator)])


# ------------------------------------------------------------------ metrics
def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def calibration_fit(y, p) -> tuple[float, float]:
    """Calibration intercept and slope from logistic recalibration on logit(p)."""
    import statsmodels.api as sm
    lp = np.log(_clip(p) / (1 - _clip(p)))
    if np.ptp(lp) == 0:
        return float("nan"), float("nan")
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(np.asarray(y), sm.add_constant(lp)).fit(disp=0, maxiter=200)
        if not res.mle_retvals.get("converged", False) or not np.all(np.isfinite(res.params)):
            return float("nan"), float("nan")      # unreliable fit: report as unavailable, not a number
        return float(res.params[0]), float(res.params[1])
    except Exception:  # perfect separation etc.
        return float("nan"), float("nan")


def point_metrics(y, p, threshold: float) -> dict[str, Any]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    intercept, slope = calibration_fit(y, p) if both else (float("nan"), float("nan"))
    return {
        "n": int(len(y)), "n_positive": int(y.sum()), "prevalence": float(y.mean()), "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, _clip(p), labels=[0, 1])),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


METRIC_FNS = {
    "roc_auc": lambda y, p, t: roc_auc_score(y, p),
    "pr_auc": lambda y, p, t: average_precision_score(y, p),
    "brier": lambda y, p, t: brier_score_loss(y, p),
    "f1": lambda y, p, t: f1_score(y, (p >= t).astype(int), zero_division=0),
}


def bootstrap_cis(y, preds: dict[str, np.ndarray], thresholds: dict[str, float], n: int, seed: int,
                  reference: str = "prevalence") -> pd.DataFrame:
    """Percentile 95% CIs, and paired differences vs the reference model on the same resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    idx = [i for i in idx if len(np.unique(y[i])) == 2]
    rows = []
    for metric, fn in METRIC_FNS.items():
        ref = np.array([fn(y[i], preds[reference][i], thresholds[reference]) for i in idx])
        for name, p in preds.items():
            vals = np.array([fn(y[i], p[i], thresholds[name]) for i in idx])
            delta = vals - ref
            better = (delta < 0) if metric == "brier" else (delta > 0)
            rows.append({"model": name, "metric": metric, "ci_low": float(np.percentile(vals, 2.5)),
                         "ci_high": float(np.percentile(vals, 97.5)),
                         "delta_vs_" + reference: float(np.mean(delta)),
                         "delta_ci_low": float(np.percentile(delta, 2.5)),
                         "delta_ci_high": float(np.percentile(delta, 97.5)),
                         "share_resamples_better": float(better.mean()), "n_resamples": len(idx)})
    return pd.DataFrame(rows)


def f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


# ------------------------------------------------------------------ experiment
def fit_models(train: pd.DataFrame, cat, num, cfg: ModelingConfig, names: list[str]) -> dict[str, dict]:
    X, y = as_categorical_strings(train, cat)[cat + num], train[TARGET].to_numpy()
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    fitted = {}
    for name in names:
        est, grid = model_specs(cfg.seed, cfg.fast)[name]
        pipe = make_pipeline(est, cat, num)
        if grid:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="average_precision", n_jobs=1, refit=True)
            search.fit(X, y)
            best, best_params, cv_score = search.best_estimator_, search.best_params_, float(search.best_score_)
        else:
            best, best_params = pipe.fit(X, y), {}
            cv_score = float(np.mean([average_precision_score(y[te], clone(pipe).fit(X.iloc[tr], y[tr])
                                                              .predict_proba(X.iloc[te])[:, 1])
                                      for tr, te in cv.split(X, y)]))
        if name == "majority":
            threshold = 0.5
        else:
            oof = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba")[:, 1]
            threshold = f1_threshold(y, oof)
        fitted[name] = {"model": best, "params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                                  for k, v in best_params.items()},
                        "cv_pr_auc": cv_score, "threshold": threshold}
        log.info("fitted %s cv_pr_auc=%.4f", name, cv_score)
    return fitted


def predict(fitted: dict, df: pd.DataFrame, cat, num) -> dict[str, np.ndarray]:
    X = as_categorical_strings(df, cat)[cat + num]
    return {name: f["model"].predict_proba(X)[:, 1] for name, f in fitted.items()}


def subgroup_metrics(test: pd.DataFrame, preds: dict, thresholds: dict, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        for level, g in test.groupby(test[col].astype(str)):
            y = g[TARGET].to_numpy()
            for name, p in preds.items():
                m = point_metrics(y, p[g.index.to_numpy()], thresholds[name])
                rows.append({"subgroup": col, "level": level, "model": name, **{k: m[k] for k in
                             ("n", "n_positive", "prevalence", "roc_auc", "pr_auc", "brier", "recall", "precision")}})
    from .report import flag_stability
    return flag_stability(pd.DataFrame(rows))


def encoded_source(encoded: str, features: list[str]) -> str:
    """Map an encoded column (e.g. 'cat__phase_group_PHASE2', 'num__missingindicator_min_age_years')
    back to its source feature by longest matching feature name."""
    name = encoded.split("__", 1)[-1]
    name = name.removeprefix("missingindicator_")
    matches = [f for f in features if name == f or name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


def logistic_coefficients(fitted_logistic: Pipeline, features: list[str]) -> pd.DataFrame:
    """Coefficients on the standardised/one-hot scale. One-hot levels are not dropped (L2 keeps them
    identifiable), so a level's coefficient is relative to the other levels of the same feature."""
    names = fitted_logistic.named_steps["prep"].get_feature_names_out()
    names = names[fitted_logistic.named_steps["var"].get_support()]
    coef = fitted_logistic.named_steps["model"].coef_[0]
    out = pd.DataFrame({"encoded_feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    out["source_feature"] = [encoded_source(n, features) for n in names]
    out["availability"] = out["source_feature"].map(availability)
    return out.sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


def permutation_table(fitted: dict, test: pd.DataFrame, cat, num, cfg: ModelingConfig, names: list[str]) -> pd.DataFrame:
    X, y = as_categorical_strings(test, cat)[cat + num], test[TARGET].to_numpy()
    rows = []
    for name in names:
        r = permutation_importance(fitted[name]["model"], X, y, scoring="average_precision",
                                   n_repeats=cfg.permutation_repeats, random_state=cfg.seed, n_jobs=1)
        for col, mean, std in zip(X.columns, r.importances_mean, r.importances_std):
            rows.append({"model": name, "feature": col, "availability": availability(col),
                         "importance_mean_pr_auc_drop": float(mean), "importance_std": float(std)})
    return pd.DataFrame(rows).sort_values(["model", "importance_mean_pr_auc_drop"], ascending=[True, False])


def data_fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{a},{b}" for a, b in zip(df["nct_id"], df[TARGET]))
    return hashlib.sha256(payload.encode()).hexdigest()


def run_experiment(df, split_name, feature_set_name, cfg, names, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    cat, num = feature_set(feature_set_name)
    cat, num = [c for c in cat if c not in exclude], [c for c in num if c not in exclude]
    if split_name == "temporal":
        train, test, split_desc = temporal_split(df, cfg.test_fraction)
    else:
        train, test, split_desc = random_split(df, cfg.test_fraction, cfg.seed)
    check_split(train, test, cfg)
    fitted = fit_models(train, cat, num, cfg, names)
    preds = predict(fitted, test, cat, num)
    thresholds = {n: f["threshold"] for n, f in fitted.items()}
    metrics = pd.DataFrame([{"model": n, **point_metrics(test[TARGET], preds[n], thresholds[n]),
                             "cv_pr_auc_train": fitted[n]["cv_pr_auc"], "best_params": json.dumps(fitted[n]["params"])}
                            for n in names])
    return {"cat": cat, "num": num, "train": train, "test": test, "split": split_desc, "fitted": fitted,
            "preds": preds, "thresholds": thresholds, "metrics": metrics}


def run_modeling(db_path: Path, out_dir: Path, *, development: bool = False,
                 cfg: ModelingConfig = ModelingConfig()) -> dict[str, Any]:
    data = load_study_data(db_path, development=development)
    df = prepare_frame(data.cohort)
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    primary = run_experiment(df, "temporal", "primary", cfg, MODEL_ORDER)
    test, preds, thr = primary["test"], primary["preds"], primary["thresholds"]
    boot = bootstrap_cis(test[TARGET].to_numpy(), preds, thr, cfg.n_bootstrap, cfg.seed)
    subgroups = subgroup_metrics(test, preds, thr, ["phase_group", "lead_sponsor_class", "start_year"])
    calib = []
    for name, p in preds.items():
        frac, mean_p = calibration_curve(test[TARGET], p, n_bins=10, strategy="quantile")
        calib += [{"model": name, "bin": i, "mean_predicted": float(a), "observed_rate": float(b)}
                  for i, (a, b) in enumerate(zip(mean_p, frac))]
    curves = []
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(test[TARGET], p)
        prec, rec, _ = precision_recall_curve(test[TARGET], p)
        curves += [{"model": name, "curve": "roc", "x": float(a), "y": float(b)} for a, b in zip(fpr, tpr)]
        curves += [{"model": name, "curve": "pr", "x": float(a), "y": float(b)} for a, b in zip(rec, prec)]
    coefs = logistic_coefficients(primary["fitted"]["logistic"]["model"], primary["cat"] + primary["num"])
    perm = permutation_table(primary["fitted"], test, primary["cat"], primary["num"], cfg,
                             ["logistic", "random_forest", "hist_gradient_boosting"])

    sensitivity = []
    for split_name, fs in (("random", "primary"), ("temporal", "with_time_varying"), ("temporal", "with_start_year")):
        try:
            exp = run_experiment(df, split_name, fs, cfg, ["prevalence", "logistic", "random_forest"])
            sensitivity.append(exp["metrics"].assign(experiment=f"{split_name}_split/{fs}", split=exp["split"]))
        except InsufficientDataError as exc:
            sensitivity.append(pd.DataFrame([{"experiment": f"{split_name}_split/{fs}", "model": None,
                                              "note": f"not run: {exc}"}]))
    sensitivity = pd.concat(sensitivity, ignore_index=True, sort=False)

    pred_df = test[["nct_id", "start_date", "phase_group", "lead_sponsor_class", TARGET]].copy()
    for name, p in preds.items():
        pred_df[f"p_{name}"] = p
    for name, f in primary["fitted"].items():
        joblib.dump(f["model"], out_dir / "models" / f"{name}.joblib")

    experiment = {
        "design": "retrospective association; temporal hold-out; current-version registry records",
        "positive_class": "TERMINATED (target_terminated=1)", "negative_class": "COMPLETED",
        "split": primary["split"], "test_cutoff_start_date": primary["split"],
        "n_train": int(len(primary["train"])), "n_test": int(len(test)),
        "train_positive": int(primary["train"][TARGET].sum()), "test_positive": int(test[TARGET].sum()),
        "features_categorical": primary["cat"], "features_numeric": primary["num"],
        "thresholds": thr, "best_params": {n: f["params"] for n, f in primary["fitted"].items()},
        "data_fingerprint": data_fingerprint(df), "test_ids_sha256": hashlib.sha256(
            "\n".join(test["nct_id"]).encode()).hexdigest(),
        "source": data.metadata_summary, "development": development, "config": asdict(cfg),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    tables = {"metrics_test": primary["metrics"], "bootstrap": boot, "subgroups": subgroups,
              "calibration_bins": pd.DataFrame(calib), "curves": pd.DataFrame(curves),
              "logistic_coefficients": coefs, "permutation_importance": perm, "sensitivity": sensitivity,
              "predictions_test": pred_df}
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "experiment.json").write_text(json.dumps(experiment, indent=2, default=str), encoding="utf-8")
    (out_dir / "evaluation_summary.md").write_text(_evaluation_md(experiment, tables), encoding="utf-8")
    return experiment


def _evaluation_md(exp: dict, t: dict) -> str:
    dev = exp["development"] or exp["source"].get("population_status") not in ("research_verified", "research_reconciled")
    banner = ("> **DEVELOPMENT / SMOKE RUN - NOT REPORTABLE RESULTS.**\n\n" if dev or exp["config"]["fast"] else "")
    cols = ["model", "n", "n_positive", "prevalence", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1",
            "threshold", "calibration_intercept", "calibration_slope", "cv_pr_auc_train"]
    m = t["metrics_test"][cols].copy()
    for c in cols[3:]:
        m[c] = m[c].astype(float).round(4)
    b = t["bootstrap"].copy()
    for c in ("ci_low", "ci_high", "delta_vs_prevalence", "delta_ci_low", "delta_ci_high", "share_resamples_better"):
        b[c] = b[c].round(4)
    return "".join([
        "# Model evaluation summary (auto-generated)\n\n", banner,
        f"- Design: {exp['design']}\n- Positive class: {exp['positive_class']}\n",
        f"- Train n={exp['n_train']} ({exp['train_positive']} terminated); test n={exp['n_test']} "
        f"({exp['test_positive']} terminated); test = studies starting on/after {exp['split']}\n",
        f"- Data fingerprint `{exp['data_fingerprint'][:16]}`; source build `{exp['source'].get('raw_run_id')}`\n\n",
        "## Test-set metrics (thresholds chosen on training out-of-fold predictions)\n\n", md_table(m),
        "\n## Bootstrap 95% CIs and paired differences vs prevalence baseline\n\n", md_table(b, max_rows=40),
        "\n## Sensitivity experiments\n\n", md_table(t["sensitivity"]),
        "\nThese metrics describe retrospective association in current-version registry records. They are not "
        "evidence of prospective predictive validity.\n",
    ])


def reproduce(db_path: Path, out_dir: Path, *, development: bool = False, tol: float = 1e-9) -> pd.DataFrame:
    """Reload saved models, recompute test metrics, compare with saved metrics."""
    out_dir = Path(out_dir)
    exp = json.loads((out_dir / "experiment.json").read_text(encoding="utf-8"))
    df = prepare_frame(load_study_data(db_path, development=development).cohort)
    if data_fingerprint(df) != exp["data_fingerprint"]:
        raise ValueError("Cohort differs from the one the artifacts were trained on (fingerprint mismatch).")
    test = df[df["start_date"] >= pd.Timestamp(exp["split"])].reset_index(drop=True)
    if hashlib.sha256("\n".join(test["nct_id"]).encode()).hexdigest() != exp["test_ids_sha256"]:
        raise ValueError("Reconstructed test set differs from the saved one.")
    cat, num = exp["features_categorical"], exp["features_numeric"]
    saved = pd.read_csv(out_dir / "metrics_test.csv").set_index("model")
    X = as_categorical_strings(test, cat)[cat + num]
    rows = []
    for name in saved.index:
        model = joblib.load(out_dir / "models" / f"{name}.joblib")
        m = point_metrics(test[TARGET], model.predict_proba(X)[:, 1], exp["thresholds"][name])
        for k in ("roc_auc", "pr_auc", "brier", "f1", "precision", "recall"):
            a, b = float(saved.loc[name, k]), float(m[k])
            same = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol
            rows.append({"model": name, "metric": k, "saved": a, "recomputed": b, "match": bool(same)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ablation
def paired_deltas(y, full: dict, full_thr: dict, abl: dict, abl_thr: dict, n: int, seed: int) -> pd.DataFrame:
    """Ablated minus full, per model and metric, on identical bootstrap resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n)) if len(np.unique(y[i])) == 2]
    rows = []
    for name in abl:
        for metric, fn in METRIC_FNS.items():
            d = np.array([fn(y[i], abl[name][i], abl_thr[name]) - fn(y[i], full[name][i], full_thr[name]) for i in idx])
            worse = (d > 0) if metric == "brier" else (d < 0)
            rows.append({"model": name, "metric": metric, "delta_ablated_minus_full": float(d.mean()),
                         "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                         "share_resamples_ablated_worse": float(worse.mean()), "n_resamples": len(idx)})
    return pd.DataFrame(rows)


def registration_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: when were studies registered relative to start and (post-outcome) completion dates?
    Uses completion dates for DIAGNOSIS ONLY; they are never model features."""
    from .cohort import wilson_interval
    rows = []
    checks = {
        "registered > 30 days after start": df["registration_lag_days"] > 30,
        "registered > 1 year after start": df["registration_lag_days"] > 365,
        "registered after primary completion date": df["first_submit_date"] > df["primary_completion_date"],
        "registered after completion date": df["first_submit_date"] > df["completion_date"],
    }
    for label, mask in checks.items():
        for cls, name in ((1, "terminated"), (0, "completed")):
            g = df[TARGET] == cls
            k, n = int((mask & g).sum()), int(g.sum())
            lo, hi = wilson_interval(pd.Series([k]), pd.Series([n]))
            rows.append({"check": label, "group": name, "n": n, "count": k, "share": round(k / n, 4) if n else None,
                         "ci_low": lo.iloc[0], "ci_high": hi.iloc[0]})
    for cls, name in ((1, "terminated"), (0, "completed")):
        lag = df.loc[df[TARGET] == cls, "registration_lag_days"].dropna()
        rows.append({"check": "median registration lag (days)", "group": name, "n": int(len(lag)),
                     "count": None, "share": float(lag.median()) if len(lag) else None, "ci_low": None, "ci_high": None})
    return pd.DataFrame(rows)


def run_ablation(db_path: Path, artifacts_dir: Path, drop: list[str], *, development: bool = False) -> dict[str, Any]:
    """Re-run the SAVED primary protocol without `drop` features and compare against the saved primary
    predictions on the identical temporal test set (verified by data fingerprint and test-ID hash)."""
    models_dir = Path(artifacts_dir) / "models"
    exp = json.loads((models_dir / "experiment.json").read_text(encoding="utf-8"))
    cfg = ModelingConfig(**exp["config"])                     # same protocol as the primary run
    data = load_study_data(db_path, development=development)
    df = prepare_frame(data.cohort)
    if data_fingerprint(df) != exp["data_fingerprint"]:
        raise ValueError("Cohort differs from the primary experiment (fingerprint mismatch); retrain first.")
    features = exp["features_categorical"] + exp["features_numeric"]
    unknown = [d for d in drop if d not in features]
    if unknown:
        raise ValueError(f"Not primary features: {unknown}")
    cat = [c for c in exp["features_categorical"] if c not in drop]
    num = [c for c in exp["features_numeric"] if c not in drop]
    train, test, split = temporal_split(df, cfg.test_fraction)
    if split != exp["split"] or hashlib.sha256("\n".join(test["nct_id"]).encode()).hexdigest() != exp["test_ids_sha256"]:
        raise ValueError("Temporal split differs from the primary experiment.")
    check_split(train, test, cfg)
    names = [n for n in MODEL_ORDER if n != "majority"]
    fitted = fit_models(train, cat, num, cfg, names)
    abl = predict(fitted, test, cat, num)
    abl_thr = {n: f["threshold"] for n, f in fitted.items()}
    saved = pd.read_csv(models_dir / "predictions_test.csv")
    if saved["nct_id"].tolist() != test["nct_id"].tolist():
        raise ValueError("Saved primary predictions are not aligned with the reconstructed test set.")
    full = {n: saved[f"p_{n}"].to_numpy() for n in names}
    y = test[TARGET].to_numpy()
    rows = []
    for n in names:
        for variant, p, t in (("full (saved primary)", full[n], exp["thresholds"][n]), ("ablated", abl[n], abl_thr[n])):
            m = point_metrics(y, p, t)
            rows.append({"model": n, "variant": variant, **{k: m[k] for k in (
                "roc_auc", "pr_auc", "brier", "log_loss", "calibration_intercept", "calibration_slope",
                "precision", "recall", "f1", "threshold", "tn", "fp", "fn", "tp")}})
    comparison = pd.DataFrame(rows)
    deltas = paired_deltas(y, full, exp["thresholds"], abl, abl_thr, cfg.n_bootstrap, cfg.seed)
    timing = registration_timing(df)
    out = Path(artifacts_dir) / "ablation" / "_".join(sorted(drop))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"dropped_features": sorted(drop), "protocol": "saved primary config, temporal split, same grids/seed",
            "split": split, "n_train": int(len(train)), "n_test": int(len(test)),
            "data_fingerprint": exp["data_fingerprint"], "test_ids_sha256": exp["test_ids_sha256"],
            "source": data.metadata_summary, "primary_source": exp["source"], "development": development,
            "config": asdict(cfg), "best_params": {n: f["params"] for n, f in fitted.items()},
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}}
    comparison.to_csv(out / "comparison.csv", index=False)
    deltas.to_csv(out / "paired_deltas.csv", index=False)
    timing.to_csv(out / "registration_timing.csv", index=False)
    (out / "ablation.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out / "ablation_summary.md").write_text(_ablation_md(meta, comparison, deltas, timing), encoding="utf-8")
    return meta


def _ablation_md(meta, comparison, deltas, timing) -> str:
    c = comparison.copy()
    for col in c.columns[2:]:
        c[col] = pd.to_numeric(c[col], errors="coerce").round(4)
    d = deltas.copy()
    for col in ("delta_ablated_minus_full", "ci_low", "ci_high", "share_resamples_ablated_worse"):
        d[col] = d[col].round(4)
    return "".join([
        f"# Ablation: without {', '.join(meta['dropped_features'])} (auto-generated)\n\n",
        f"- Protocol: {meta['protocol']}; test = starts on/after {meta['split']} (n={meta['n_test']}).\n",
        f"- Source raw run `{meta['source'].get('raw_run_id')}`; data fingerprint `{meta['data_fingerprint'][:16]}`.\n",
        "- Deltas are ablated minus full on identical bootstrap resamples; for Brier, positive = worse.\n\n",
        "## Metrics\n\n", md_table(c), "\n## Paired differences (95% CI)\n\n", md_table(d, max_rows=40),
        "\n## Registration timing (descriptive; completion dates used for diagnosis only)\n\n", md_table(timing),
    ])
