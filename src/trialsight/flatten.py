"""Flatten nested API v2 study records into normalised tables.

Field paths follow the API v2 JSON structure (protocolSection.*Module.*,
derivedSection.*, hasResults). Every derived column is documented in
docs/DATA_DICTIONARY.md.

Privacy: contact persons (centralContacts, overallOfficials, per-site contacts,
responsible-party investigator names) are intentionally NOT extracted.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterable

import pandas as pd

DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$", re.IGNORECASE)
UNIT_TO_YEARS = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 7 / 365.25,
    "day": 1 / 365.25,
    "hour": 1 / 8766,
    "minute": 1 / 525960,
}
DATE_STRUCTS = [
    ("start", "startDateStruct"),
    ("primary_completion", "primaryCompletionDateStruct"),
    ("completion", "completionDateStruct"),
    ("first_post", "studyFirstPostDateStruct"),
    ("last_update_post", "lastUpdatePostDateStruct"),
]
CHILD_COLUMNS = {
    "conditions": ["nct_id", "kind", "term"],
    "mesh": ["nct_id", "kind", "mesh_id", "term"],
    "interventions": ["nct_id", "type", "name"],
    "sponsors": ["nct_id", "role", "name", "class"],
    "locations": ["nct_id", "facility", "city", "state", "country", "site_status"],
}
CHILD_TABLES = tuple(CHILD_COLUMNS)


def _d(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else {}


def _l(obj: Any) -> list:
    return obj if isinstance(obj, list) else []


def parse_ct_date(value: Any) -> tuple[dt.date | None, str | None]:
    """Parse 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'.

    Returns (date, precision). Partial dates are anchored to the first day of
    the period and flagged via precision ('year'/'month'/'day'). Unparseable
    input returns (None, 'invalid'); missing input returns (None, None).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    m = DATE_RE.match(str(value).strip())
    if not m:
        return None, "invalid"
    y, mo, d = m.groups()
    try:
        if d:
            return dt.date(int(y), int(mo), int(d)), "day"
        if mo:
            return dt.date(int(y), int(mo), 1), "month"
        return dt.date(int(y), 1, 1), "year"
    except ValueError:
        return None, "invalid"


def parse_age_years(value: Any) -> float | None:
    if not value:
        return None
    m = AGE_RE.match(str(value))
    if not m:
        return None
    return round(float(m.group(1)) * UNIT_TO_YEARS[m.group(2).lower()], 4)


def phase_group(phases: list[str]) -> str:
    s = sorted(set(phases))
    if s == ["PHASE2"]:
        return "PHASE2"
    if s == ["PHASE3"]:
        return "PHASE3"
    if s == ["PHASE2", "PHASE3"]:
        return "PHASE2_3"
    return "|".join(s) if s else "NONE"


def flatten_study(study: dict[str, Any]) -> dict[str, Any]:
    ps = _d(study.get("protocolSection"))
    ident = _d(ps.get("identificationModule"))
    status = _d(ps.get("statusModule"))
    sponsor_mod = _d(ps.get("sponsorCollaboratorsModule"))
    oversight = _d(ps.get("oversightModule"))
    cond_mod = _d(ps.get("conditionsModule"))
    design = _d(ps.get("designModule"))
    design_info = _d(design.get("designInfo"))
    masking = _d(design_info.get("maskingInfo"))
    enrollment = _d(design.get("enrollmentInfo"))
    arms_mod = _d(ps.get("armsInterventionsModule"))
    outcomes = _d(ps.get("outcomesModule"))
    elig = _d(ps.get("eligibilityModule"))
    locs = _l(_d(ps.get("contactsLocationsModule")).get("locations"))
    derived = _d(study.get("derivedSection"))
    cond_browse = _d(derived.get("conditionBrowseModule"))

    nct = ident.get("nctId")
    phases = [p for p in _l(design.get("phases")) if isinstance(p, str)]
    lead = _d(sponsor_mod.get("leadSponsor"))
    collaborators = [_d(c) for c in _l(sponsor_mod.get("collaborators"))]
    interventions = [_d(i) for i in _l(arms_mod.get("interventions"))]
    countries = sorted({_d(l).get("country") for l in locs if _d(l).get("country")})
    criteria = elig.get("eligibilityCriteria")

    row: dict[str, Any] = {
        "nct_id": nct,
        "brief_title": ident.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "last_known_status": status.get("lastKnownStatus"),
        "why_stopped": status.get("whyStopped"),
        "status_verified_date_raw": status.get("statusVerifiedDate"),
        "study_type": design.get("studyType"),
        "phases": "|".join(sorted(phases)) or None,
        "phase_group": phase_group(phases),
        "allocation": design_info.get("allocation"),
        "intervention_model": design_info.get("interventionModel"),
        "primary_purpose": design_info.get("primaryPurpose"),
        "masking": masking.get("masking"),
        "n_masked_roles": len(_l(masking.get("whoMasked"))),
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "lead_sponsor_name": lead.get("name"),
        "lead_sponsor_class": lead.get("class"),
        "n_collaborators": len(collaborators),
        "has_dmc": oversight.get("oversightHasDmc"),
        "is_fda_regulated_drug": oversight.get("isFdaRegulatedDrug"),
        "is_fda_regulated_device": oversight.get("isFdaRegulatedDevice"),
        "n_conditions": len(_l(cond_mod.get("conditions"))),
        "n_arms": len(_l(arms_mod.get("armGroups"))),
        "n_interventions": len(interventions),
        "intervention_types": "|".join(sorted({i.get("type") for i in interventions if i.get("type")})) or None,
        "n_primary_outcomes": len(_l(outcomes.get("primaryOutcomes"))),
        "n_secondary_outcomes": len(_l(outcomes.get("secondaryOutcomes"))),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "sex": elig.get("sex"),
        "min_age_years": parse_age_years(elig.get("minimumAge")),
        "max_age_years": parse_age_years(elig.get("maximumAge")),
        "std_ages": "|".join(sorted(_l(elig.get("stdAges")))) or None,
        "eligibility_criteria_chars": len(criteria) if isinstance(criteria, str) else None,
        "n_locations": len(locs),
        "n_countries": len(countries),
        "has_us_site": "United States" in countries,
        "has_results": study.get("hasResults"),
    }

    for prefix, key in DATE_STRUCTS:
        s = _d(status.get(key))
        parsed, precision = parse_ct_date(s.get("date"))
        row[f"{prefix}_date"] = parsed
        row[f"{prefix}_date_precision"] = precision
        row[f"{prefix}_date_raw"] = s.get("date")
        row[f"{prefix}_date_type"] = s.get("type")

    first_submit, fs_prec = parse_ct_date(status.get("studyFirstSubmitDate"))
    row["first_submit_date"] = first_submit
    row["first_submit_date_precision"] = fs_prec
    row["first_submit_date_raw"] = status.get("studyFirstSubmitDate")
    row["start_year"] = row["start_date"].year if row["start_date"] else None
    row["registration_lag_days"] = (
        (first_submit - row["start_date"]).days if first_submit and row["start_date"] else None
    )

    children: dict[str, list[dict[str, Any]]] = {t: [] for t in CHILD_TABLES}
    for c in _l(cond_mod.get("conditions")):
        children["conditions"].append({"nct_id": nct, "kind": "condition", "term": c})
    for k in _l(cond_mod.get("keywords")):
        children["conditions"].append({"nct_id": nct, "kind": "keyword", "term": k})
    for kind, key in (("mesh", "meshes"), ("ancestor", "ancestors")):
        for m in _l(cond_browse.get(key)):
            m = _d(m)
            children["mesh"].append({"nct_id": nct, "kind": kind, "mesh_id": m.get("id"), "term": m.get("term")})
    for b in _l(cond_browse.get("browseBranches")):
        b = _d(b)
        children["mesh"].append({"nct_id": nct, "kind": "branch", "mesh_id": b.get("abbrev"), "term": b.get("name")})
    for i in interventions:
        children["interventions"].append({"nct_id": nct, "type": i.get("type"), "name": i.get("name")})
    children["sponsors"].append({"nct_id": nct, "role": "lead", "name": lead.get("name"), "class": lead.get("class")})
    for c in collaborators:
        children["sponsors"].append({"nct_id": nct, "role": "collaborator", "name": c.get("name"), "class": c.get("class")})
    for l in locs:
        l = _d(l)
        children["locations"].append(
            {
                "nct_id": nct,
                "facility": l.get("facility"),
                "city": l.get("city"),
                "state": l.get("state"),
                "country": l.get("country"),
                "site_status": l.get("status"),
            }
        )
    return {"studies": row, **children}


def flatten_studies(studies: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    children: dict[str, list[dict[str, Any]]] = {t: [] for t in CHILD_TABLES}
    for study in studies:
        flat = flatten_study(study)
        rows.append(flat["studies"])
        for t in CHILD_TABLES:
            children[t].extend(flat[t])

    studies_df = pd.DataFrame(rows)
    for col in [c for c in studies_df.columns if c.endswith("_date") and not c.endswith("_raw")]:
        studies_df[col] = pd.to_datetime(studies_df[col])
    out = {"studies": studies_df}
    for t in CHILD_TABLES:
        out[t] = pd.DataFrame(children[t], columns=CHILD_COLUMNS[t])
    return out
