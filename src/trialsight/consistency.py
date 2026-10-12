"""Cross-artifact consistency and publication checks.

Verifies that the processed build, EDA, model artifacts, ablations and generated documentation all
come from the same research run, that documented metrics equal the saved metrics, and that files
intended for publication contain no 'nan' displays or local filesystem paths.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

LOCAL_PATH = re.compile(r"([A-Za-z]:\\Users\\[^\\\s\"']+|/home/(?!claude)[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)")
NAN_TOKEN = re.compile(r"(?<![A-Za-z_])nan(?![A-Za-z_])", re.IGNORECASE)
DOCS = ("MODEL_CARD.md", "EVALUATION_REPORT.md", "RESEARCH_FINDINGS.md")


def _row(rows, cid, desc, ok, detail="", na=False):
    rows.append({"check_id": cid, "description": desc,
                 "status": "not_checked" if na else ("pass" if ok else "fail"), "detail": detail})


def check_consistency(processed_dir: Path, artifacts_dir: Path, docs_dir: Path, db_path: Path | None = None,
                      publish_files: list[Path] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    pdir, adir, ddir = Path(processed_dir), Path(artifacts_dir), Path(docs_dir)
    bi_path = pdir / "build_info.json"
    bi = json.loads(bi_path.read_text(encoding="utf-8")) if bi_path.exists() else {}
    _row(rows, "C01", "Processed build is an accepted, validated research population",
         bi.get("population_status") in ("research_verified", "research_reconciled") and bi.get("cohort_validated") is True
         and bi.get("output_state") == "accepted", f"{bi.get('population_status')}/{bi.get('output_state')}", na=not bi)
    run, built = bi.get("raw_run_id"), bi.get("built_at_utc")
    eda_p, exp_p = adir / "eda" / "eda_metadata.json", adir / "models" / "experiment.json"
    eda = json.loads(eda_p.read_text(encoding="utf-8")) if eda_p.exists() else {}
    exp = json.loads(exp_p.read_text(encoding="utf-8")) if exp_p.exists() else {}
    _row(rows, "C02", "EDA artifacts come from the processed build",
         eda.get("raw_run_id") == run and eda.get("built_at_utc") == built and not eda.get("development"),
         f"eda={eda.get('raw_run_id')}/{eda.get('built_at_utc')}", na=not eda)
    src = exp.get("source", {})
    _row(rows, "C03", "Model artifacts come from the processed build",
         src.get("raw_run_id") == run and src.get("built_at_utc") == built, f"models={src.get('raw_run_id')}", na=not exp)
    _row(rows, "C04", "Models are full (not --fast) research runs",
         bool(exp) and not exp.get("development") and not exp.get("config", {}).get("fast"), na=not exp)
    metrics_p = adir / "models" / "metrics_test.csv"
    metrics = pd.read_csv(metrics_p) if metrics_p.exists() else pd.DataFrame()
    missing = [m for m in metrics.get("model", []) if not (adir / "models" / "models" / f"{m}.joblib").exists()]
    _row(rows, "C05", "A saved model file exists for every evaluated model", not missing, f"missing={missing}",
         na=metrics.empty)
    for d in sorted((adir / "ablation").glob("*/ablation.json")) if (adir / "ablation").exists() else []:
        meta = json.loads(d.read_text(encoding="utf-8"))
        _row(rows, "C06", f"Ablation {d.parent.name} matches the primary experiment",
             meta.get("data_fingerprint") == exp.get("data_fingerprint") and meta.get("test_ids_sha256") ==
             exp.get("test_ids_sha256") and meta.get("source", {}).get("raw_run_id") == run, d.parent.name)
    for name in DOCS:
        path = ddir / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        generated = bool(text) and "Not yet generated" not in text
        _row(rows, "C07", f"{name} is generated from this run",
             generated and str(run) in text and str(built) in text and exp.get("data_fingerprint", "")[:16] in text,
             "" if (generated and str(run) in text) else ("placeholder, different run, or no fingerprint" if text else "missing"))
        _row(rows, "C08", f"{name} contains no 'nan' displays", not NAN_TOKEN.search(text),
             f"{len(NAN_TOKEN.findall(text))} occurrences", na=not text)
    ev = (ddir / "EVALUATION_REPORT.md").read_text(encoding="utf-8") if (ddir / "EVALUATION_REPORT.md").exists() else ""
    if ev and not metrics.empty:
        bad = []
        for r in metrics.itertuples():
            line = next((l for l in ev.splitlines() if l.startswith(f"| {r.model} | {r.n} |")), "")
            for col in ("roc_auc", "pr_auc", "brier"):
                if f"| {round(float(getattr(r, col)), 4):g} |" not in line:
                    bad.append(f"{r.model}.{col}")
        _row(rows, "C09", "EVALUATION_REPORT test metrics equal metrics_test.csv", not bad, f"mismatched={bad}")
    if db_path and Path(db_path).exists():
        from .modeling import data_fingerprint, prepare_frame
        from .study_data import load_study_data
        data = load_study_data(db_path)
        fp = data_fingerprint(prepare_frame(data.cohort))
        _row(rows, "C10", "Cohort in the research database matches the model fingerprint",
             fp == exp.get("data_fingerprint") and data.metadata.get("raw_run_id") == run, fp[:16])
    else:
        _row(rows, "C10", "Cohort in the research database matches the model fingerprint", False,
             "database not available here", na=True)
    leaks = []
    for f in publish_files or []:
        try:
            txt = Path(f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if LOCAL_PATH.search(txt):
            leaks.append(str(f))
    _row(rows, "C11", "Publishable text files contain no local user paths", not leaks, f"files={leaks}",
         na=publish_files is None)
    return pd.DataFrame(rows)


def publishable_files(root: Path) -> list[Path]:
    """Text files that would be committed (mirrors .gitignore exclusions for generated data)."""
    root = Path(root)
    skip = ("data/", "artifacts", ".venv", "__pycache__", ".git/", ".pytest_cache", "build/", ".egg-info")
    out = []
    for f in root.rglob("*"):
        rel = f.relative_to(root).as_posix()
        if f.is_file() and not any(s in rel for s in skip) and f.suffix in (".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".json", ".cfg"):
            out.append(f)
    return out
