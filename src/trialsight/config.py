"""Typed configuration loaded from config/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class ApiConfig:
    base_url: str = "https://clinicaltrials.gov/api/v2"
    page_size: int = 1000
    requests_per_minute: int = 40
    timeout_seconds: int = 60
    max_retries: int = 5


@dataclass(frozen=True)
class QueryConfig:
    cond: str | None = None
    advanced_filter: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        """Translate to ClinicalTrials.gov v2 /studies query parameters."""
        params: dict[str, Any] = {"format": "json"}
        if self.cond:
            params["query.cond"] = self.cond
        if self.advanced_filter:
            params["filter.advanced"] = " ".join(self.advanced_filter.split())
        params.update(self.extra_params or {})
        return params


@dataclass(frozen=True)
class ResolutionRule:
    """Selects the primary analysis window from observed outcome resolution.

    For each start year: resolved_share = resolved / (resolved + unresolved).
    Ambiguous statuses (default UNKNOWN) are excluded from that denominator and
    reported separately. The window runs from the earliest start year up to the
    last year of the initial consecutive run of years meeting the threshold.
    """

    min_resolved_share: float = 0.90
    resolved_statuses: list[str] = field(default_factory=lambda: ["COMPLETED", "TERMINATED", "WITHDRAWN"])
    unresolved_statuses: list[str] = field(default_factory=lambda: [
        "NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING", "SUSPENDED",
    ])
    ambiguous_statuses: list[str] = field(default_factory=lambda: ["UNKNOWN"])
    # How ambiguous statuses enter resolved_share:
    #   "exclude"             -> resolved / (resolved + unresolved)              (original behaviour)
    #   "count_as_unresolved" -> resolved / (resolved + unresolved + ambiguous)  (stricter)
    ambiguous_handling: str = "exclude"
    # Calendar years inside the eligibility range that contain no studies:
    #   "skip"  -> ignored when testing contiguity (original behaviour)
    #   "break" -> treated as failing, so the window cannot span an empty year
    missing_year_policy: str = "skip"
    # Years whose share denominator is below this are treated as failing
    # ("insufficient_n"). 1 reproduces the original behaviour.
    min_studies_per_year: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.min_resolved_share <= 1:
            raise ValueError(f"min_resolved_share must be in (0, 1], got {self.min_resolved_share}")
        if self.ambiguous_handling not in ("exclude", "count_as_unresolved"):
            raise ValueError(f"ambiguous_handling must be 'exclude' or 'count_as_unresolved', got {self.ambiguous_handling!r}")
        if self.missing_year_policy not in ("skip", "break"):
            raise ValueError(f"missing_year_policy must be 'skip' or 'break', got {self.missing_year_policy!r}")
        if int(self.min_studies_per_year) < 1:
            raise ValueError("min_studies_per_year must be >= 1")
        groups = [set(self.resolved_statuses), set(self.unresolved_statuses), set(self.ambiguous_statuses)]
        overlap = (groups[0] & groups[1]) | (groups[0] & groups[2]) | (groups[1] & groups[2])
        if overlap:
            raise ValueError(f"Statuses appear in more than one resolution group: {sorted(overlap)}")


@dataclass(frozen=True)
class CohortConfig:
    start_date_min: str
    start_date_max: str
    phase_groups: list[str]
    target_statuses: dict[str, int]
    require_oncology_mesh: bool = True
    oncology_mesh_terms: list[str] = field(default_factory=lambda: ["Neoplasms"])
    resolution_rule: ResolutionRule | None = None


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    db_path: Path
    processed_dir: Path
    api_snapshot_dir: Path
    # Development (non-research) builds are written here so a sample can never
    # overwrite or be mistaken for the research outputs.
    dev_db_path: Path | None = None
    dev_processed_dir: Path | None = None
    artifacts_dir: Path | None = None
    dev_artifacts_dir: Path | None = None


DEFAULT_CANCER_TERMS = [
    "cancer", "cancers", "carcinoma", "carcinomas", "neoplasm", "neoplasms", "neoplasia",
    "tumor", "tumors", "tumour", "tumours", "malignancy", "malignancies", "malignant",
    "lymphoma", "lymphomas", "leukemia", "leukemias", "leukaemia", "myeloma", "sarcoma",
    "sarcomas", "melanoma", "glioma", "gliomas", "glioblastoma", "blastoma", "mesothelioma",
    "metastatic", "metastases", "metastasis", "oncology", "myelodysplastic",
]


@dataclass(frozen=True)
class DiagnosticsConfig:
    thresholds: list[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95])
    cancer_text_terms: list[str] = field(default_factory=lambda: list(DEFAULT_CANCER_TERMS))
    review_sample_size: int = 200
    profile_columns: list[str] = field(
        default_factory=lambda: ["phase_group", "lead_sponsor_class", "allocation", "primary_purpose"]
    )


@dataclass(frozen=True)
class Config:
    api: ApiConfig
    query: QueryConfig
    cohort: CohortConfig
    paths: PathsConfig
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    modeling: dict = field(default_factory=dict)  # kwargs for trialsight.modeling.ModelingConfig


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    root = root or PROJECT_ROOT
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [k for k in ("api", "query", "cohort", "paths") if k not in raw]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")

    paths = raw["paths"]
    cohort_raw = dict(raw["cohort"])
    rule_raw = cohort_raw.pop("resolution_rule", None)
    rule = ResolutionRule(**rule_raw) if rule_raw is not None else None
    return Config(
        api=ApiConfig(**raw["api"]),
        query=QueryConfig(**raw["query"]),
        cohort=CohortConfig(**cohort_raw, resolution_rule=rule),
        paths=PathsConfig(
            raw_dir=_resolve(root, paths["raw_dir"]),
            db_path=_resolve(root, paths["db_path"]),
            processed_dir=_resolve(root, paths["processed_dir"]),
            api_snapshot_dir=_resolve(root, paths["api_snapshot_dir"]),
            dev_db_path=_resolve(root, paths["dev_db_path"]) if paths.get("dev_db_path") else None,
            dev_processed_dir=_resolve(root, paths["dev_processed_dir"]) if paths.get("dev_processed_dir") else None,
            artifacts_dir=_resolve(root, paths.get("artifacts_dir", "artifacts")),
            dev_artifacts_dir=_resolve(root, paths.get("dev_artifacts_dir", "artifacts_dev")),
        ),
        modeling=dict(raw.get("modeling") or {}),
        diagnostics=DiagnosticsConfig(**(raw.get("diagnostics") or {})),
    )
