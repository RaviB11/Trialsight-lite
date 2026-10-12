"""CLI exit codes for the analysis commands (SYNTHETIC data; machinery only)."""
import yaml

from trialsight.cli import main
from trialsight.config import DEFAULT_CONFIG_PATH
from v3_synthetic import write_synthetic_research_db


def _cfg(tmp_path):
    base = yaml.safe_load(open(DEFAULT_CONFIG_PATH, encoding="utf-8"))
    base["paths"] = {k: str(tmp_path / v) for k, v in {
        "raw_dir": "raw", "db_path": "processed/t.duckdb", "processed_dir": "processed", "api_snapshot_dir": "snap",
        "dev_db_path": "processed_dev/t_dev.duckdb", "dev_processed_dir": "processed_dev",
        "artifacts_dir": "artifacts", "dev_artifacts_dir": "artifacts_dev"}.items()}
    base["modeling"].update({"n_bootstrap": 20, "cv_folds": 3, "permutation_repeats": 1,
                             "min_events_train": 10, "min_events_test": 10})
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return ["--config", str(p)]


def test_analysis_commands_refuse_without_accepted_build(tmp_path):
    c = _cfg(tmp_path)
    for cmd in ("eda", "train", "reproduce"):
        assert main(c + [cmd]) in (1, 6), cmd          # 6 = refused by the research guard
    assert main(c + ["eda"]) == 6
    assert main(c + ["publish-docs"]) == 1            # no research artifacts


def test_development_analysis_flow_and_publish_refusal(tmp_path, capsys):
    c = _cfg(tmp_path)
    dev = tmp_path / "processed_dev" / "t_dev.duckdb"
    dev.parent.mkdir(parents=True)
    write_synthetic_research_db(dev, n=400, output_state="development", status="development_not_for_research")
    assert main(c + ["eda", "--development"]) == 0
    assert main(c + ["train", "--development", "--fast"]) == 0
    assert main(c + ["reproduce", "--development"]) == 0
    assert "REPRODUCED" in capsys.readouterr().out
    assert not (tmp_path / "artifacts").exists()        # development never writes research artifacts
    assert (tmp_path / "artifacts_dev" / "models" / "experiment.json").exists()
    assert main(c + ["publish-docs"]) == 1              # nothing publishable


def test_reproduce_mismatch_exit_code(tmp_path):
    c = _cfg(tmp_path)
    dev = tmp_path / "processed_dev" / "t_dev.duckdb"
    dev.parent.mkdir(parents=True)
    write_synthetic_research_db(dev, n=400, output_state="development", status="development_not_for_research")
    assert main(c + ["train", "--development", "--fast"]) == 0
    m = tmp_path / "artifacts_dev" / "models" / "metrics_test.csv"
    text = m.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    i = header.index("brier")
    row = text[3].split(",")
    row[i] = str(float(row[i]) + 0.01)                  # tamper with one saved metric
    text[3] = ",".join(row)
    m.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert main(c + ["reproduce", "--development"]) == 5
