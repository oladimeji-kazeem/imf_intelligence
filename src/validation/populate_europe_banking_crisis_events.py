
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT / "data" / "external"
    / "europe_historical_stress_events.csv"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_banking_crisis_population_report.txt"
)

SOURCE_ORG = "International Monetary Fund"
SOURCE_TITLE = (
    "Laeven and Valencia (2026), "
    "Systemic Banking Crises Database: 1970-2025"
)
SOURCE_URL = (
    "https://www.imf.org/en/publications/wp/issues/"
    "2026/05/14/systemic-banking-crises-database-1970-2025-576036"
)

episodes = [
    ("AUT", "Austria", 2008, 2012, "AUT_2008",
     "IMF Appendix I, Table A1."),
    ("FIN", "Finland", 1991, 1995, "FIN_1991",
     "IMF Appendix I, Table A1."),
    ("FRA", "France", 2008, 2009, "FRA_2008",
     "IMF Appendix I, Table A1."),
    ("DEU", "Germany", 2008, 2009, "DEU_2008",
     "IMF Appendix I, Table A1."),
    ("GRC", "Greece", 2008, 2012, "GRC_2008",
     "IMF Appendix I, Table A1; duration capped at five years."),
    ("HUN", "Hungary", 1991, 1995, "HUN_1991",
     "IMF Appendix I, Table A1."),
    ("HUN", "Hungary", 2008, 2012, "HUN_2008",
     "IMF Appendix I, Table A1; duration capped at five years."),
    ("IRL", "Ireland", 2008, 2012, "IRL_2008",
     "IMF Appendix I, Table A1; duration capped at five years."),
    ("ITA", "Italy", 2008, 2009, "ITA_2008",
     "IMF Appendix I, Table A1."),
    ("NOR", "Norway", 1991, 1993, "NOR_1991",
     "IMF Appendix I, Table A1."),
    ("POL", "Poland", 1992, 1994, "POL_1992",
     "IMF Appendix I, Table A1."),
    ("PRT", "Portugal", 2008, 2012, "PRT_2008",
     "IMF Appendix I, Table A1; duration capped at five years."),
    ("ROU", "Romania", 1998, 1999, "ROU_1998",
     "IMF Appendix I, Table A1."),
]

if not REGISTRY.exists():
    raise FileNotFoundError(REGISTRY)

existing = pd.read_csv(REGISTRY, dtype={"iso": str})

required_columns = [
    "event_id", "iso", "country", "event_type",
    "event_name", "onset_year", "end_year", "severity",
    "source_organisation", "source_title", "source_url",
    "source_reference", "evidence_note", "verification_status",
    "inclusion_decision", "exclusion_reason",
]

missing = set(required_columns) - set(existing.columns)
if missing:
    raise ValueError(f"Missing registry columns: {sorted(missing)}")

rows = []

for iso, country, start, end, suffix, reference in episodes:
    rows.append({
        "event_id": f"EU_BANK_{suffix}",
        "iso": iso,
        "country": country,
        "event_type": "systemic_banking_crisis",
        "event_name": f"{country} systemic banking crisis",
        "onset_year": start,
        "end_year": end,
        "severity": "IMF_systemic_banking_crisis",
        "source_organisation": SOURCE_ORG,
        "source_title": SOURCE_TITLE,
        "source_url": SOURCE_URL,
        "source_reference": reference,
        "evidence_note": (
            "Episode classified in the IMF Laeven-Valencia 2026 "
            "systemic banking crisis database. "
            "Reported end year follows the IMF table; where stated, "
            "duration is capped at five years."
        ),
        "verification_status": "verified",
        "inclusion_decision": "include",
        "exclusion_reason": "",
    })

new = pd.DataFrame(rows)

existing_ids = set(existing["event_id"].dropna().astype(str))
new_ids = set(new["event_id"])

duplicates = existing_ids.intersection(new_ids)
if duplicates:
    raise ValueError(f"Event IDs already exist: {sorted(duplicates)}")

combined = pd.concat([existing, new], ignore_index=True)

# Validate the complete registry before writing.
errors = []

if combined["event_id"].duplicated().any():
    errors.append("Duplicate event IDs.")

if len(new) != 13:
    errors.append("Expected 13 new banking-crisis episodes.")

if new["iso"].nunique() != 12:
    errors.append("Expected 12 distinct countries.")

if not (
    new["event_type"] == "systemic_banking_crisis"
).all():
    errors.append("Unexpected event type.")

if not (
    new["verification_status"] == "verified"
).all():
    errors.append("Unexpected verification status.")

if not (
    new["inclusion_decision"] == "include"
).all():
    errors.append("Unexpected inclusion decision.")

if not (
    pd.to_numeric(new["onset_year"]) >= 1980
).all():
    errors.append("An event begins before the project data window.")

if not (
    pd.to_numeric(new["end_year"]) >=
    pd.to_numeric(new["onset_year"])
).all():
    errors.append("Invalid event date range.")

if errors:
    raise ValueError("\n".join(errors))

combined.to_csv(REGISTRY, index=False, encoding="utf-8")

# Re-read the saved registry.
saved = pd.read_csv(REGISTRY, dtype={"iso": str})

if saved["event_id"].duplicated().any():
    raise ValueError("Saved registry contains duplicate IDs.")

report = [
    "=" * 72,
    "EUROPEAN SYSTEMIC BANKING CRISIS POPULATION",
    "=" * 72,
    "",
    f"New records: {len(new)}",
    f"New countries: {new['iso'].nunique()}",
    f"Total registry records: {len(saved)}",
    "",
    "COUNTRY DISTRIBUTION",
    "-" * 72,
]

for iso, group in new.groupby("iso"):
    report.append(
        f"{iso} | {group['country'].iloc[0]} | "
        f"{len(group)} episode(s)"
    )

report.extend([
    "",
    "VALIDATION",
    "-" * 72,
    f"Unique event IDs: {'PASS' if not saved['event_id'].duplicated().any() else 'FAIL'}",
    f"13 new episodes: {'PASS' if len(new) == 13 else 'FAIL'}",
    f"12 countries: {'PASS' if new['iso'].nunique() == 12 else 'FAIL'}",
    f"All systemic banking crises: PASS",
    f"All sourced to IMF 2026: PASS",
    "",
    "QUALIFICATIONS",
    "-" * 72,
    "IMF end years may be capped at five years.",
    "End years are not necessarily exact resolution dates.",
    "These events are banking crises, not sovereign defaults.",
    "Spain's 1977 onset is excluded from this batch.",
    "",
    "STATUS: POPULATION VALIDATED",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
