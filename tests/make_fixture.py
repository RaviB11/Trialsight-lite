"""Generates tests/fixtures/synthetic_studies.json.

SYNTHETIC, STRUCTURE-ONLY TEST DATA. These are NOT real trials. NCT IDs in the
NCT9999xxxx range are placeholders. The fixture mimics the documented API v2
JSON shape to exercise parsing, data-quality rules and cohort logic. It must
never be used for analysis or reported results.
"""
import json
from pathlib import Path

ONC = {"meshes": [{"id": "D000000", "term": "Example Carcinoma"}],
       "ancestors": [{"id": "D009369", "term": "Neoplasms"}],
       "browseBranches": [{"abbrev": "BC04", "name": "Neoplasms"}]}
NON_ONC = {"meshes": [{"id": "D000001", "term": "Example Metabolic Disease"}],
           "ancestors": [{"id": "D000002", "term": "Metabolic Diseases"}]}


def study(nct, status, phases, start, completion, comp_type="ACTUAL", enroll=(100, "ACTUAL"),
          why=None, browse=ONC, countries=("United States",), first_submit=None):
    st = {"overallStatus": status,
          "startDateStruct": {"date": start, "type": "ACTUAL"},
          "primaryCompletionDateStruct": {"date": completion, "type": comp_type},
          "completionDateStruct": {"date": completion, "type": comp_type},
          "studyFirstSubmitDate": first_submit or start,
          "studyFirstPostDateStruct": {"date": first_submit or start, "type": "ACTUAL"}}
    if why:
        st["whyStopped"] = why
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Synthetic study {nct}"},
            "statusModule": st,
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Synthetic Sponsor", "class": "INDUSTRY"},
                                           "collaborators": [{"name": "Synthetic Collaborator", "class": "OTHER"}]},
            "oversightModule": {"oversightHasDmc": True},
            "conditionsModule": {"conditions": ["Example condition"], "keywords": ["example"]},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": list(phases),
                             "designInfo": {"allocation": "RANDOMIZED", "interventionModel": "PARALLEL",
                                            "primaryPurpose": "TREATMENT",
                                            "maskingInfo": {"masking": "DOUBLE",
                                                            "whoMasked": ["PARTICIPANT", "INVESTIGATOR"]}},
                             "enrollmentInfo": {"count": enroll[0], "type": enroll[1]}},
            "armsInterventionsModule": {"armGroups": [{"label": "A"}, {"label": "B"}],
                                        "interventions": [{"type": "DRUG", "name": "Drug X"},
                                                          {"type": "OTHER", "name": "Placebo"}]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "m1"}], "secondaryOutcomes": [{"measure": "m2"}]},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion: adults.", "healthyVolunteers": False,
                                  "sex": "ALL", "minimumAge": "18 Years", "maximumAge": "6 Months",
                                  "stdAges": ["ADULT", "OLDER_ADULT"]},
            "contactsLocationsModule": {"locations": [{"facility": f"Site {i}", "city": "City", "country": c}
                                                      for i, c in enumerate(countries)]},
        },
        "derivedSection": {"conditionBrowseModule": browse},
        "hasResults": status == "COMPLETED",
    }


studies = [
    study("NCT99990001", "COMPLETED", ["PHASE3"], "2011-03-15", "2015-06-30", countries=("United States", "Canada")),
    study("NCT99990002", "TERMINATED", ["PHASE2"], "2012-05", "2013-01", enroll=(12, "ACTUAL"),
          why="Slow accrual", first_submit="2012-08-01"),
    study("NCT99990003", "WITHDRAWN", ["PHASE2", "PHASE3"], "2014-01-01", "2014-02-01", enroll=(0, "ACTUAL")),
    study("NCT99990004", "RECRUITING", ["PHASE2"], "2016-09-01", "2027-01-01", comp_type="ESTIMATED",
          enroll=(80, "ESTIMATED")),
    study("NCT99990005", "COMPLETED", ["PHASE3"], "2010-01-01", "2012-01-01", browse=NON_ONC),
    study("NCT99990006", "COMPLETED", ["PHASE2"], "2005-01-01", "2007-01-01"),
    study("NCT99990001", "COMPLETED", ["PHASE3"], "2011-03-15", "2015-06-30"),
    study("NCT99990007", "TERMINATED", ["PHASE2"], "2013-13", "2012-01-01"),
    study("NCT99990008", "COMPLETED", ["PHASE1"], "2012-01-01", "2013-01-01"),
    study("NCT123", "COMPLETED", ["PHASE2"], "2012-01-01", "2013-01-01"),
    study("NCT99990009", "COMPLETED", ["PHASE2"], "2014-04-01", "2016-04-01", countries=()),
]

payload = {"_note": "SYNTHETIC STRUCTURE-ONLY TEST FIXTURE - NOT REAL TRIALS. See make_fixture.py.",
           "studies": studies}
out = Path(__file__).parent / "fixtures" / "synthetic_studies.json"
out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(studies)} records)")
