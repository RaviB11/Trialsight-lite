"""Single entry point for downstream analysis data. Enforces the research guard:
research mode accepts only staged-and-accepted, validated research builds;
development mode loads any build but results are labelled DEVELOPMENT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .storage import ResearchPopulationError, require_research_population


@dataclass
class StudyData:
    cohort: pd.DataFrame
    mesh: pd.DataFrame
    conditions: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    development: bool = False

    @property
    def metadata_summary(self) -> dict:
        keys = ("raw_run_id", "population_status", "output_state", "cohort_validated", "built_at_utc",
                "config_sha256", "analysis_start_year", "analysis_end_year")
        return {k: self.metadata.get(k) for k in keys}


def load_study_data(db_path: Path, development: bool = False) -> StudyData:
    db_path = Path(db_path)
    if development:
        if not db_path.exists():
            raise ResearchPopulationError(f"No database at {db_path}.")
    else:
        require_research_population(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "build_metadata" not in tables or "cohort" not in tables:
            raise ResearchPopulationError(f"{db_path} lacks build_metadata/cohort tables; rebuild it.")
        meta = con.execute("SELECT * FROM build_metadata").df().iloc[0].to_dict()
        cohort = con.execute("SELECT * FROM cohort ORDER BY nct_id").df()
        mesh = con.execute("SELECT m.* FROM mesh m JOIN cohort c USING (nct_id)").df()
        conditions = con.execute("SELECT x.* FROM conditions x JOIN cohort c USING (nct_id)").df()
    if development:
        meta = {**meta, "population_status": f"{meta.get('population_status')} (loaded in development mode)"}
    return StudyData(cohort=cohort, mesh=mesh, conditions=conditions, metadata=meta, development=development)
