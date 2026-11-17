
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT / "data" / "external"
    / "europe_historical_stress_events.csv"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_financial_assistance_population_report.txt"
)

SOURCE_URL = (
    "https://www.esm.europa.eu/system/files/document/"
    "2022-03/ESM_DP_19.pdf"
)

SOURCE_TITLE = (
    "ESM Appendix 1 – Euro area financial assistance programmes"
)

SOURCE_ORG = "European Stability Mechanism"

# Formal programme agreement years and completion years.
# These are programme dates, not necessarily the first year
# in which the underlying economic stress began.

events = [
    {
        "event_id": "EU_FA_GRC_2010_01",
        "iso": "GRC",
        "country": "Greece",
        "event_type": "financial_assistance",
        "event_name": "Greek Loan Facility programme",
        "onset_year": 2010,
        "end_year": 2012,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 2 May 2010",
        "evidence_note": (
            "Formal Greek Loan Facility financial assistance programme. "
            "The programme was subsequently replaced."
        ),
    },
    {
        "event_id": "EU_FA_GRC_2012_02",
        "iso": "GRC",
        "country": "Greece",
        "event_type": "financial_assistance",
        "event_name": "Second Greek financial assistance programme",
        "onset_year": 2012,
        "end_year": 2015,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 15 March 2012",
        "evidence_note": (
            "Second programme provided through EFSF and IMF. "
            "Programme expired in June 2015."
        ),
    },
    {
        "event_id": "EU_FA_GRC_2015_03",
        "iso": "GRC",
        "country": "Greece",
        "event_type": "financial_assistance",
        "event_name": "Third Greek financial assistance programme",
        "onset_year": 2015,
        "end_year": 2018,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 19 August 2015",
        "evidence_note": (
            "Third programme provided through ESM and IMF. "
            "Programme completed in August 2018."
        ),
    },
    {
        "event_id": "EU_FA_IRL_2010_01",
        "iso": "IRL",
        "country": "Ireland",
        "event_type": "financial_assistance",
        "event_name": "Irish financial assistance programme",
        "onset_year": 2010,
        "end_year": 2013,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 22 December 2010",
        "evidence_note": (
            "Financial assistance involving EFSF, EFSM, IMF and bilateral "
            "lenders. Programme completed in December 2013."
        ),
    },
    {
        "event_id": "EU_FA_PRT_2011_01",
        "iso": "PRT",
        "country": "Portugal",
        "event_type": "financial_assistance",
        "event_name": "Portuguese financial assistance programme",
        "onset_year": 2011,
        "end_year": 2014,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 27 May 2011",
        "evidence_note": (
            "Financial assistance involving EFSF, EFSM and IMF. "
            "Programme completed in May 2014."
        ),
    },
    {
        "event_id": "EU_FA_ESP_2012_01",
        "iso": "ESP",
        "country": "Spain",
        "event_type": "financial_assistance",
        "event_name": "Spanish banking-sector recapitalisation programme",
        "onset_year": 2012,
        "end_year": 2013,
        "severity": "official_financial_assistance",
        "source_reference": "Appendix 1; agreement entered 24 July 2012",
        "evidence_note": (
            "ESM assistance for banking-sector recapitalisation. "
            "This was not a general sovereign fiscal adjustment programme."
        ),
    },
]

df_new = pd.DataFrame(events)

df_new["source_organisation"] = SOURCE_ORG
df_new["source_title"] = SOURCE_TITLE
df_new["source_url"] = SOURCE_URL
df_new["verification_status"] = "verified"
df_new["inclusion_decision"] = "include"
df_new["exclusion_reason"] = ""

if not REGISTRY.exists():
    raise FileNotFoundError(
        f"Registry not found: {REGISTRY}"
    )

existing = pd.read_csv(REGISTRY, dtype={"iso": str})

# Ensure the expected schema is present.
missing = set(df_new.columns) - set(existing.columns)

if missing:
    raise ValueError(
        f"Registry is missing columns: {sorted(missing)}"
    )

# Do not silently overwrite previously entered events.
existing_ids = set(existing["event_id"].dropna().astype(str))

duplicates = existing_ids.intersection(
    set(df_new["event_id"])
)

if duplicates:
    raise ValueError(
        "These event IDs already exist. Review them before rerunning: "
        f"{sorted(duplicates)}"
    )

combined = pd.concat(
    [existing, df_new],
    ignore_index=True
)

combined.to_csv(
    REGISTRY,
    index=False,
    encoding="utf-8"
)

# Re-read the saved file for validation.
saved = pd.read_csv(REGISTRY, dtype={"iso": str})

checks = {
    "six_new_records": len(df_new) == 6,
    "six_unique_event_ids": df_new["event_id"].nunique() == 6,
    "all_verified": (
        df_new["verification_status"] == "verified"
    ).all(),
    "all_included": (
        df_new["inclusion_decision"] == "include"
    ).all(),
    "all_assistance_events": (
        df_new["event_type"] == "financial_assistance"
    ).all(),
    "no_duplicate_registry_ids": (
        not saved["event_id"].duplicated().any()
    ),
    "four_countries": df_new["iso"].nunique() == 4,
}

errors = [
    name for name, passed in checks.items()
    if not passed
]

report = [
    "=" * 72,
    "EUROPEAN FINANCIAL ASSISTANCE EVENT POPULATION",
    "=" * 72,
    "",
    f"Registry: {REGISTRY}",
    f"New records: {len(df_new)}",
    f"Total registry records: {len(saved)}",
    f"Countries represented in new records: {df_new['iso'].nunique()}",
    "",
    "EVENT DISTRIBUTION",
    "-" * 72,
]

for iso, group in df_new.groupby("iso"):
    report.append(
        f"{iso} | {group['country'].iloc[0]} | "
        f"{len(group)} programmes"
    )

report.extend([
    "",
    "VALIDATION",
    "-" * 72,
])

for name, passed in checks.items():
    report.append(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

report.extend([
    "",
    f"Errors: {len(errors)}",
    "",
    (
        "STATUS: POPULATION VALIDATED"
        if not errors else "STATUS: VALIDATION FAILED"
    ),
    "",
    "QUALIFICATION",
    "-" * 72,
    "These records identify formal financial assistance programmes.",
    "They do not establish sovereign default or the exact onset of crisis.",
    "The formal agreement year is used as the programme event year.",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))

if errors:
    raise SystemExit(1)
