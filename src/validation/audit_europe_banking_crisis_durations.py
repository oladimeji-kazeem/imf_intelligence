
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = (
    ROOT / "data" / "external"
    / "europe_historical_stress_events.csv"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "europe_banking_crisis_duration_audit.csv"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_banking_crisis_duration_audit_report.txt"
)

EXPECTED_BANKING_EVENTS = 13
IMF_MAX_DURATION_YEARS = 5

REQUIRED = [
    "event_id",
    "iso",
    "country",
    "event_type",
    "onset_year",
    "end_year",
    "source_organisation",
    "source_title",
    "source_reference",
    "evidence_note",
    "verification_status",
    "inclusion_decision",
]

if not REGISTRY.exists():
    raise FileNotFoundError(f"Registry not found: {REGISTRY}")

registry = pd.read_csv(REGISTRY)

missing = [c for c in REQUIRED if c not in registry.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

banking = registry[
    registry["event_type"].astype(str).str.strip()
    == "systemic_banking_crisis"
].copy()

if len(banking) != EXPECTED_BANKING_EVENTS:
    raise ValueError(
        f"Expected {EXPECTED_BANKING_EVENTS} banking-crisis records; "
        f"found {len(banking)}."
    )

if banking["event_id"].duplicated().any():
    raise ValueError("Duplicate banking-crisis event IDs.")

banking["onset_year"] = pd.to_numeric(
    banking["onset_year"], errors="coerce"
)

banking["end_year"] = pd.to_numeric(
    banking["end_year"], errors="coerce"
)

if banking[["onset_year", "end_year"]].isna().any().any():
    raise ValueError("Missing or invalid banking-crisis dates.")

if (
    banking["onset_year"] % 1 != 0
).any() or (
    banking["end_year"] % 1 != 0
).any():
    raise ValueError("Banking-crisis years must be whole numbers.")

banking["onset_year"] = banking["onset_year"].astype(int)
banking["end_year"] = banking["end_year"].astype(int)

if (banking["end_year"] < banking["onset_year"]).any():
    raise ValueError("At least one crisis ends before its onset.")

# Inclusive duration: 2008–2012 = 5 calendar years.
banking["duration_years_inclusive"] = (
    banking["end_year"] - banking["onset_year"] + 1
)

def classify_duration(duration):
    if duration > IMF_MAX_DURATION_YEARS:
        return "exceeds_imf_five_year_cap"
    if duration == IMF_MAX_DURATION_YEARS:
        return "five_year_truncation_candidate"
    return "within_five_year_limit"

banking["duration_audit_class"] = (
    banking["duration_years_inclusive"].apply(classify_duration)
)

# The registry may not preserve the IMF Appendix I footnote marker.
# Do not infer confirmed truncation from duration alone.
banking["imf_appendix_footnote_5_marker"] = "not_yet_checked"

banking["truncation_assessment"] = banking[
    "duration_years_inclusive"
].apply(
    lambda x: (
        "potentially_truncated_confirm_source_footnote"
        if x == IMF_MAX_DURATION_YEARS
        else (
            "duration_exceeds_published_cap_investigate"
            if x > IMF_MAX_DURATION_YEARS
            else "not_at_five_year_cap"
        )
    )
)

banking["actual_resolution_year"] = pd.NA
banking["actual_resolution_source"] = ""
banking["manual_review_status"] = "pending"
banking["reviewer_notes"] = ""

# Keep source fields and existing episode assignments if available.
preferred_columns = [
    "event_id",
    "episode_id",
    "iso",
    "country",
    "event_name",
    "event_type",
    "onset_year",
    "end_year",
    "duration_years_inclusive",
    "duration_audit_class",
    "imf_appendix_footnote_5_marker",
    "truncation_assessment",
    "actual_resolution_year",
    "actual_resolution_source",
    "source_organisation",
    "source_title",
    "source_reference",
    "evidence_note",
    "verification_status",
    "inclusion_decision",
    "manual_review_status",
    "reviewer_notes",
]

# episode_id is optional in the original registry.
available_columns = [
    c for c in preferred_columns if c in banking.columns
]

audit = banking[available_columns].copy()

# Structural checks.
if len(audit) != EXPECTED_BANKING_EVENTS:
    raise ValueError("Audit row count does not reconcile.")

if audit["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in audit.")

if audit["duration_years_inclusive"].isna().any():
    raise ValueError("Missing calculated durations.")

if (
    audit["duration_years_inclusive"] < 1
).any():
    raise ValueError("Invalid duration below one year.")

if (
    audit["duration_years_inclusive"] > IMF_MAX_DURATION_YEARS
).any():
    raise ValueError(
        "One or more records exceed the IMF five-year published cap. "
        "Investigate the source and registry dates."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

audit.to_csv(OUTPUT, index=False, encoding="utf-8")

# Re-read saved output.
saved = pd.read_csv(OUTPUT)

if len(saved) != EXPECTED_BANKING_EVENTS:
    raise ValueError("Saved audit row count does not reconcile.")

if saved["event_id"].duplicated().any():
    raise ValueError("Duplicate saved event IDs.")

if set(saved["event_id"]) != set(banking["event_id"]):
    raise ValueError("Saved event IDs do not match source records.")

# Summary.
counts = audit["duration_audit_class"].value_counts()

candidate_count = int(
    (audit["duration_years_inclusive"] == 5).sum()
)

within_count = int(
    (audit["duration_years_inclusive"] < 5).sum()
)

lines = [
    "=" * 76,
    "EUROPEAN BANKING-CRISIS DURATION AUDIT",
    "=" * 76,
    "",
    "SOURCE",
    "-" * 76,
    str(REGISTRY),
    "",
    "INPUT RECONCILIATION",
    "-" * 76,
    f"Banking-crisis records: {len(banking)}",
    f"Unique countries: {banking['iso'].nunique()}",
    f"Unique event IDs: {banking['event_id'].nunique()}",
    "",
    "DURATION CLASSIFICATION",
    "-" * 76,
    f"Five-year truncation candidates: {candidate_count}",
    f"Within five-year limit: {within_count}",
    f"Exceeding five-year cap: "
    f"{int((audit['duration_years_inclusive'] > 5).sum())}",
    "",
    "INDIVIDUAL EPISODES",
    "-" * 76,
]

for _, row in audit.sort_values(
    ["iso", "onset_year"]
).iterrows():
    lines.append(
        f"{row['iso']} | "
        f"{row['onset_year']}-{row['end_year']} | "
        f"{row['duration_years_inclusive']} calendar years | "
        f"{row['duration_audit_class']}"
    )

lines.extend([
    "",
    "METHODOLOGICAL CONTROLS",
    "-" * 76,
    "- Duration is calculated inclusively.",
    "- Five calendar years is a truncation candidate, not proof.",
    "- IMF Appendix I footnote 5/ must be checked for confirmation.",
    "- No crisis end dates have been changed.",
    "- No actual resolution dates have been inferred.",
    "- Original registry has not been modified.",
    "",
    "OUTPUT",
    "-" * 76,
    str(OUTPUT),
    str(REPORT),
    "",
    "STATUS: STRUCTURAL VALIDATION PASSED",
    "SOURCE FOOTNOTE REVIEW: PENDING",
])

REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
