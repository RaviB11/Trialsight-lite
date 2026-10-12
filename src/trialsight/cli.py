"""Command-line interface.

    trialsight inspect            # snapshot API metadata/search areas/enums + count for configured query
    trialsight fetch --dry-run    # report how many studies the configured query returns
    trialsight fetch              # download all pages into data/raw/<run_id>/
    trialsight build              # latest complete raw run -> DuckDB + Parquet + reports (research-grade runs only)
    trialsight build --development # any run -> separate *_dev outputs, labelled not-for-research
    trialsight verify             # recompute download integrity from disk (exit 3 if not research-grade)
    trialsight reconcile --run-dir D --reason "..."   # explicitly accept a complete-but-unverified run
    trialsight count --cond "..." # count an alternative query without downloading

    trialsight eda | train | reproduce [--development]   # analysis on the accepted research build
    trialsight publish-docs       # MODEL_CARD / EVALUATION_REPORT / RESEARCH_FINDINGS from research artifacts

Exit codes: 0 ok, 1 input/config error, 2 API error, 3 run not research-grade, 4 cohort not validated,
5 reproduction mismatch, 6 refused: not an accepted research population, 7 artifacts/docs inconsistent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .api_client import CTGovAPIError, CTGovClient
from .config import load_config
from .pipeline import build_database, development_config
from .storage import ResearchPopulationError
from .raw_store import fetch_to_raw, latest_complete_run, latest_run, reconcile_run, verify_run

log = logging.getLogger("trialsight")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def cmd_inspect(cfg, args) -> int:
    client = CTGovClient(cfg.api)
    out = Path(cfg.paths.api_snapshot_dir)
    for name, fn in [
        ("version", client.version),
        ("studies_metadata", client.studies_metadata),
        ("search_areas", client.search_areas),
        ("enums", client.enums),
    ]:
        _write_json(out / f"{name}.json", fn())
        print(f"Saved {out / (name + '.json')}")
    params = cfg.query.to_params()
    total = client.count(params)
    _write_json(out / "configured_query_count.json", {"query_params": params, "totalCount": total})
    print(f"Configured query returns totalCount={total}")
    return 0


def cmd_fetch(cfg, args) -> int:
    client = CTGovClient(cfg.api)
    params = cfg.query.to_params()
    if args.dry_run:
        total = client.count(params)
        print(json.dumps({"query_params": params, "totalCount": total}, indent=2))
        return 0
    run_dir = fetch_to_raw(client, params, cfg.paths.raw_dir, max_pages=args.max_pages, recount_at_end=True)
    print(f"Raw run written to {run_dir}")
    return _print_verification(verify_run(run_dir))


def _print_verification(v) -> int:
    keys = ("run_id", "status", "research_grade", "reported_count", "count_at_end", "retrieved_records",
            "unique_nct_ids", "conflicting_duplicate_ids", "exact_duplicate_records", "records_missing_nct_id",
            "reasons", "page_anomalies", "reconciliation_valid")
    d = v.to_dict()
    print(json.dumps({k: d[k] for k in keys}, indent=2))
    if not v.research_grade:
        print("NOT RESEARCH-GRADE: this run cannot be used for a research build without re-fetching "
              "or an explicit `trialsight reconcile`.", file=sys.stderr)
    return 0 if v.research_grade else 3


def cmd_verify(cfg, args) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(cfg.paths.raw_dir)
    return _print_verification(verify_run(run_dir))


def cmd_reconcile(cfg, args) -> int:
    rec = reconcile_run(Path(args.run_dir), args.reason, documentation_only=args.note_only)
    if not rec["grants_research_grade"]:
        print("Recorded a DOCUMENTATION-ONLY note. It does not grant research grade.", file=sys.stderr)
    print(json.dumps(rec, indent=2))
    return _print_verification(verify_run(Path(args.run_dir)))


def cmd_count(cfg, args) -> int:
    params = cfg.query.to_params()
    if args.cond is not None:
        params.pop("query.cond", None)
        if args.cond.strip():
            params["query.cond"] = args.cond
    if args.advanced is not None:
        params.pop("filter.advanced", None)
        if args.advanced.strip():
            params["filter.advanced"] = " ".join(args.advanced.split())
    total = CTGovClient(cfg.api).count(params)
    print(json.dumps({"query_params": params, "totalCount": total}, indent=2))
    return 0


def cmd_build(cfg, args) -> int:
    development = args.development or args.allow_incomplete
    if development:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run(cfg.paths.raw_dir)
        out_cfg = development_config(cfg)
        info = build_database(run_dir, out_cfg, allow_incomplete=True, development=True)
        print("DEVELOPMENT BUILD - NOT A RESEARCH POPULATION. Outputs written to development paths only.",
              file=sys.stderr)
    else:
        run_dir = Path(args.run_dir) if args.run_dir else latest_complete_run(cfg.paths.raw_dir)
        out_cfg = cfg
        info = build_database(run_dir, out_cfg, require_research_grade=True)
    keys = ("population_status", "cohort_validated", "raw_run_id", "n_studies_raw",
            "n_eligible_all_statuses", "analysis_window", "n_cohort")
    print(json.dumps({k: info[k] for k in keys}, indent=2))
    print(f"Output state: {info['output_state']}; outputs in {info['output_dir']}")
    print(f"Report: {Path(info['output_dir']) / 'cohort_validation_report.md'}")
    if not development and not info["cohort_validated"]:
        print("COHORT NOT VALIDATED: see cohort_validation in the report before any modelling.", file=sys.stderr)
        return 4
    return 0


def _analysis_paths(cfg, development: bool) -> tuple[Path, Path]:
    """(database, artifacts root) for downstream commands; development never touches research paths."""
    if development:
        dev = development_config(cfg)
        return Path(dev.paths.db_path), Path(cfg.paths.dev_artifacts_dir or "artifacts_dev")
    return Path(cfg.paths.db_path), Path(cfg.paths.artifacts_dir or "artifacts")


def cmd_eda(cfg, args) -> int:
    from .analysis import run_eda
    db, root = _analysis_paths(cfg, args.development)
    meta = run_eda(db, root / "eda", development=args.development,
                   n_boot=int(cfg.modeling.get("n_bootstrap", 1000)), seed=int(cfg.modeling.get("seed", 20261012)))
    print(json.dumps({k: meta[k] for k in ("population_status", "n", "n_terminated", "n_tests")}, indent=2, default=str))
    print(f"EDA outputs: {root / 'eda'}")
    return 0


def cmd_train(cfg, args) -> int:
    from .modeling import ModelingConfig, run_modeling
    db, root = _analysis_paths(cfg, args.development)
    mcfg = ModelingConfig(**{**cfg.modeling, **({"fast": True} if args.fast else {})})
    exp = run_modeling(db, root / "models", development=args.development, cfg=mcfg)
    print(json.dumps({k: exp[k] for k in ("n_train", "n_test", "train_positive", "test_positive", "split")}, indent=2))
    print((root / "models" / "evaluation_summary.md").read_text(encoding="utf-8").split("## Bootstrap")[0])
    return 0


def cmd_reproduce(cfg, args) -> int:
    from .modeling import reproduce
    db, root = _analysis_paths(cfg, args.development)
    rep = reproduce(db, root / "models", development=args.development)
    print(rep.to_string(index=False))
    ok = bool(rep["match"].all())
    print("REPRODUCED: all saved metrics match." if ok else "MISMATCH: see rows with match=False.")
    return 0 if ok else 5


def cmd_ablate(cfg, args) -> int:
    from .modeling import run_ablation
    db, root = _analysis_paths(cfg, args.development)
    meta = run_ablation(db, root, args.drop, development=args.development)
    out = root / "ablation" / "_".join(sorted(args.drop))
    print((out / "ablation_summary.md").read_text(encoding="utf-8").split("## Registration timing")[0])
    print(f"Ablation outputs: {out}")
    return 0


def cmd_check_consistency(cfg, args) -> int:
    from .config import PROJECT_ROOT
    from .consistency import check_consistency, publishable_files
    rep = check_consistency(cfg.paths.processed_dir, cfg.paths.artifacts_dir or "artifacts", PROJECT_ROOT / "docs",
                            db_path=cfg.paths.db_path, publish_files=publishable_files(PROJECT_ROOT))
    print(rep.to_string(index=False))
    failed = rep[rep["status"] == "fail"]
    print("CONSISTENT: all checks passed." if failed.empty else f"INCONSISTENT: {len(failed)} check(s) failed.")
    return 0 if failed.empty else 7


def cmd_publish_docs(cfg, args) -> int:
    from .config import PROJECT_ROOT
    from .docs_publish import publish
    for p in publish(Path(cfg.paths.artifacts_dir or "artifacts"), PROJECT_ROOT / "docs", Path(cfg.paths.processed_dir)):
        print(f"wrote {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trialsight")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Snapshot API metadata and count the configured query")
    f = sub.add_parser("fetch", help="Download raw study pages")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--max-pages", type=int, default=None, help="Development only; marks run incomplete")
    b = sub.add_parser("build", help="Build DuckDB from a raw run")
    b.add_argument("--run-dir", default=None)
    b.add_argument("--development", action="store_true",
                   help="Build any run (incl. incomplete samples) into separate *_dev paths, labelled not-for-research")
    b.add_argument("--allow-incomplete", action="store_true", help="Deprecated alias of --development")
    vf = sub.add_parser("verify", help="Recompute raw-run integrity from disk")
    vf.add_argument("--run-dir", default=None)
    rc = sub.add_parser("reconcile", help="Explicitly accept a complete but unverified run")
    rc.add_argument("--run-dir", required=True)
    rc.add_argument("--reason", required=True, help="Written justification (>= 20 characters)")
    rc.add_argument("--note-only", action="store_true",
                    help="Record a documentation note without granting research grade (allowed for critical failures)")
    ct = sub.add_parser("count", help="Count matches for an alternative query (no download)")
    ct.add_argument("--cond", default=None, help="Override query.cond ('' removes it)")
    ct.add_argument("--advanced", default=None, help="Override filter.advanced ('' removes it)")

    for name, help_text in (("eda", "Exploratory statistics on the accepted research cohort"),
                            ("train", "Train and evaluate models on the accepted research cohort"),
                            ("reproduce", "Recompute metrics from saved model artifacts")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--development", action="store_true",
                        help="Use the development build; outputs go to artifacts_dev and are labelled non-research")
        if name == "train":
            sp.add_argument("--fast", action="store_true", help="Small grids for smoke testing; never publishable")
    sub.add_parser("publish-docs", help="Write MODEL_CARD/EVALUATION_REPORT/RESEARCH_FINDINGS from research artifacts")
    ab = sub.add_parser("ablate", help="Re-run the saved primary protocol without given features")
    ab.add_argument("--drop", nargs="+", required=True, help="Primary feature(s) to remove, e.g. registration_lag_days")
    ab.add_argument("--development", action="store_true")
    sub.add_parser("check-consistency", help="Verify build, artifacts and docs refer to the same research run")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    handlers = {"inspect": cmd_inspect, "fetch": cmd_fetch, "build": cmd_build, "verify": cmd_verify,
                "reconcile": cmd_reconcile, "count": cmd_count, "eda": cmd_eda, "train": cmd_train,
                "reproduce": cmd_reproduce, "publish-docs": cmd_publish_docs, "ablate": cmd_ablate,
                "check-consistency": cmd_check_consistency}
    try:
        return handlers[args.command](cfg, args)
    except CTGovAPIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 2
    except ResearchPopulationError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 6
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
