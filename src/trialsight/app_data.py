"""Data loading for the Streamlit app (no Streamlit import, so it is unit-testable).
Research mode shows only accepted, validated research builds. Development mode
must be explicitly requested and is labelled on every page."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .config import load_config
from .pipeline import development_config
from .storage import ResearchPopulationError, require_research_population

BUILD_TABLES = ["cohort", "eligible_population", "cohort_attrition", "resolution_by_start_year",
                "cohort_validation", "build_metadata", "threshold_sensitivity", "oncology_capture_summary"]
EDA_FILES = ["eda_associations", "eda_levels", "eda_therapeutic_areas", "eda_by_start_year"]
MODEL_FILES = ["metrics_test", "bootstrap", "curves", "calibration_bins", "logistic_coefficients",
               "permutation_importance", "subgroups", "sensitivity", "predictions_test"]


@dataclass
class AppState:
    mode: str                                    # research | development | unavailable
    message: str = ""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    eda: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, pd.DataFrame] = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)
    eda_meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ablations: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def metadata(self) -> dict:
        md = self.tables.get("build_metadata")
        return md.iloc[0].to_dict() if md is not None and len(md) else {}


def _read_build(db: Path) -> dict[str, pd.DataFrame]:
    with duckdb.connect(str(db), read_only=True) as con:
        have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        return {t: con.execute(f'SELECT * FROM "{t}"').df() for t in BUILD_TABLES if t in have}


def _read_csvs(folder: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(folder / f"{n}.csv") for n in names if (folder / f"{n}.csv").exists()}


def load_app_state(config_path: str | None = None, allow_development: bool = False) -> AppState:
    cfg = load_config(config_path)
    try:
        require_research_population(cfg.paths.db_path)
        state = AppState(mode="research")
        db, art = Path(cfg.paths.db_path), Path(cfg.paths.artifacts_dir or "artifacts")
    except ResearchPopulationError as exc:
        dev_db = Path(development_config(cfg).paths.db_path)
        if allow_development and dev_db.exists():
            state = AppState(mode="development", message=f"No accepted research build: {exc}")
            db, art = dev_db, Path(cfg.paths.dev_artifacts_dir or "artifacts_dev")
        else:
            return AppState(mode="unavailable", message=str(exc))
    state.tables = _read_build(db)
    state.eda = _read_csvs(art / "eda", EDA_FILES)
    state.models = _read_csvs(art / "models", MODEL_FILES)
    if (art / "models" / "experiment.json").exists():
        state.experiment = json.loads((art / "models" / "experiment.json").read_text(encoding="utf-8"))
    if (art / "eda" / "eda_metadata.json").exists():
        state.eda_meta = json.loads((art / "eda" / "eda_metadata.json").read_text(encoding="utf-8"))
    run_id = state.metadata.get("raw_run_id")
    fp = state.experiment.get("data_fingerprint")
    for d in sorted((art / "ablation").glob("*/ablation.json")) if (art / "ablation").exists() else []:
        meta = json.loads(d.read_text(encoding="utf-8"))
        if meta.get("data_fingerprint") == fp and fp:
            state.ablations[d.parent.name] = {n: pd.read_csv(d.parent / f"{n}.csv")
                                             for n in ("comparison", "paired_deltas", "registration_timing")}
        else:
            state.warnings.append(f"Ablation {d.parent.name} does not match the loaded models; hidden.")
    state.provenance = {"build raw run": str(run_id), "build time": str(state.metadata.get("built_at_utc")),
                        "EDA raw run": str(state.eda_meta.get("raw_run_id")),
                        "models raw run": str(state.experiment.get("source", {}).get("raw_run_id")),
                        "models cohort fingerprint": str(fp)[:16]}
    for label, src in (("Model artifacts", state.experiment.get("source", {}).get("raw_run_id")),
                       ("EDA artifacts", state.eda_meta.get("raw_run_id"))):
        if src is not None and src != run_id:
            state.warnings.append(f"{label} come from raw run {src}, not the loaded build {run_id}; they are hidden.")
            if label.startswith("Model"):
                state.models, state.experiment = {}, {}
            else:
                state.eda, state.eda_meta = {}, {}
    return state
