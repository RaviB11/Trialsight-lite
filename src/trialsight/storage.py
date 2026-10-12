"""DuckDB persistence and Parquet export."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_tables(db_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for name, df in tables.items():
            if not name.isidentifier():
                raise ValueError(f"Unsafe table name: {name!r}")
            con.register("_tmp_df", df)
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _tmp_df')
            con.unregister("_tmp_df")


def read_table(db_path: Path, name: str) -> pd.DataFrame:
    if not name.isidentifier():
        raise ValueError(f"Unsafe table name: {name!r}")
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f'SELECT * FROM "{name}"').df()


def list_tables(db_path: Path) -> list[str]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def export_parquet(db_path: Path, table_names: list[str], out_dir: Path) -> list[Path]:
    """Write DuckDB tables to Parquet with DuckDB's native writer (no PyArrow).

    Exports from the database itself, so each file mirrors the stored table
    exactly. COPY targets cannot be bound as query parameters, so the path is
    passed as an escaped SQL string literal; table names are validated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        for name in table_names:
            if not name.isidentifier():
                raise ValueError(f"Unsafe table name: {name!r}")
            if name not in existing:
                raise ValueError(f"Table {name!r} not found in {db_path}")
            p = out_dir / f"{name}.parquet"
            target = _sql_string_literal(p.resolve().as_posix())
            con.execute(f'COPY "{name}" TO {target} (FORMAT PARQUET, COMPRESSION ZSTD)')
            paths.append(p)
    return paths


class ResearchPopulationError(RuntimeError):
    """Raised when a build is not a validated research population."""


RESEARCH_STATUSES = ("research_verified", "research_reconciled")


def require_research_population(db_path: Path) -> dict:
    """Gate for any analysis that could produce reportable results (e.g. modelling).

    Fails closed: a missing database, a database built before build_metadata
    existed, a development build, or an unvalidated cohort all raise.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise ResearchPopulationError(f"No database at {db_path}.")
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "build_metadata" not in tables:
            raise ResearchPopulationError(
                f"{db_path} has no build_metadata table (built by an older version or a development step); "
                "rebuild from a verified run."
            )
        meta = con.execute("SELECT * FROM build_metadata").df()
    if len(meta) != 1:
        raise ResearchPopulationError("build_metadata must contain exactly one row.")
    row = meta.iloc[0].to_dict()
    if row.get("population_status") not in RESEARCH_STATUSES:
        raise ResearchPopulationError(
            f"Population status is {row.get('population_status')!r}: this is a development build and must not be "
            "used for reported analyses or validated modelling results."
        )
    if not bool(row.get("cohort_validated")):
        raise ResearchPopulationError("Cohort has not passed all critical validation checks.")
    if row.get("output_state") != "accepted":
        raise ResearchPopulationError(
            f"Output state is {row.get('output_state')!r}; only staged-and-accepted research builds "
            "(`trialsight build`) can be analysed."
        )
    return row
