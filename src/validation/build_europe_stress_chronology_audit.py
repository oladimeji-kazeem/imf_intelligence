
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGISTRY = ROOT / "data" / "external" / "europe_historical_stress_events.csv"
LINKAGE = ROOT / "data" / "processed" / "europe_stress_event_linkage.csv"
RELATIONSHIPS = (
    ROOT / "data" / "processed"
    / "europe_stress_event_relationship_candidates.csv"
)

OUT = ROOT / "data" / "processed"
REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_stress_chronology_audit_report.txt"
)

CHRONOLOGY_OUT = OUT / "europe_stress_event_chronology_audit.csv"
REVIEW_OUT = OUT / "europe_stress_relationship_adjudication.csv"

REQUIRED_REGISTRY = [
    "event_id", "iso", "country", "event_type",
    "event_name", "onset_year", "end_year",
    "source_organisation", "source_title",
    "source_url", "source_reference",
    "evidence_note", "verification_status",
    "inclusion_decision",
]

for path in [REGISTRY, LINKAGE, RELATIONSHIPS]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

registry = pd.read_csv(REGISTRY)
linkage = pd.read_csv(LINKAGE)
relationships = pd.read_csv(RELATIONSHIPS)

missing = [c for c in REQUIRED_REGISTRY if c not in registry.columns]
if missing:
    raise ValueError(f"Registry missing required columns: {missing}")

for name, frame in [
    ("registry", registry),
    ("linkage", linkage),
]:
    if "event_id" not in frame.columns:
        raise ValueError(f"{name} has no event_id column")
    if frame["event_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate event IDs")

if "episode_id" not in linkage.columns:
    raise ValueError("Linkage file has no episode_id column")

if set(registry["event_id"]) != set(linkage["event_id"]):
    raise ValueError("Registry and linkage event IDs do not match")

if len(registry) != len(linkage):
    raise ValueError("Registry and linkage row counts do not match")

# Join episode assignments to the source registry.
if "episode_id" in registry.columns:
    registry = registry.drop(columns=["episode_id"])

data = registry.merge(
    linkage[["event_id", "episode_id", "episode_assignment_method"]],
    on="event_id",
    how="left",
    validate="one_to_one",
)

def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def parse_year(value):
    if pd.isna(value) or clean(value) == "":
        return None
    return int(float(value))

def parse_date(value):
    if pd.isna(value) or clean(value) == "":
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")

def source_completeness(row):
    fields = [
        "source_organisation",
        "source_title",
        "source_url",
        "source_reference",
        "evidence_note",
    ]
    return sum(bool(clean(row.get(c, ""))) for c in fields)

audit_rows = []

for _, row in data.iterrows():
    event_type = clean(row["event_type"])
    onset_year = parse_year(row["onset_year"])
    end_year = parse_year(row["end_year"])

    start_date = parse_date(row.get("event_start_date", None))
    finish_date = parse_date(row.get("event_end_date", None))

    date_warnings = []

    if onset_year is None:
        date_warnings.append("missing_onset_year")

    if end_year is None:
        date_warnings.append("missing_end_year")

    if (
        onset_year is not None
        and end_year is not None
        and end_year < onset_year
    ):
        date_warnings.append("end_year_precedes_onset_year")

    if event_type == "financial_assistance":
        date_semantics = "programme_or_intervention_date"
        onset_interpretation = (
            "Agreement or programme date; not established as crisis onset"
        )
        date_warnings.append("do_not_use_as_crisis_onset_without_evidence")

    elif event_type == "systemic_banking_crisis":
        date_semantics = "source_defined_crisis_period"
        onset_interpretation = (
            "Source-defined banking-crisis onset; verify source methodology"
        )
        date_warnings.append(
            "check_whether_end_year_is_capped_or_source_defined"
        )

    elif event_type == "sovereign_restructuring":
        date_semantics = "restructuring_transaction_period"
        onset_interpretation = (
            "Restructuring transaction dates; not necessarily crisis onset"
        )

    else:
        date_semantics = "requires_manual_interpretation"
        onset_interpretation = "Not yet adjudicated"

    if pd.isna(start_date) and onset_year is not None:
        start_date_display = str(onset_year)
    else:
        start_date_display = (
            start_date.strftime("%Y-%m-%d")
            if not pd.isna(start_date) else ""
        )

    if pd.isna(finish_date) and end_year is not None:
        end_date_display = str(end_year)
    else:
        end_date_display = (
            finish_date.strftime("%Y-%m-%d")
            if not pd.isna(finish_date) else ""
        )

    verification = clean(row["verification_status"])
    inclusion = clean(row["inclusion_decision"])

    if verification != "verified":
        date_warnings.append("source_verification_not_verified")

    if inclusion != "include":
        date_warnings.append("event_not_currently_included")

    completeness = source_completeness(row)

    audit_rows.append({
        "event_id": row["event_id"],
        "episode_id": row["episode_id"],
        "iso": row["iso"],
        "country": row["country"],
        "event_type": event_type,
        "event_name": row["event_name"],
        "onset_year": onset_year,
        "end_year": end_year,
        "event_start_date": start_date_display,
        "event_end_date": end_date_display,
        "date_semantics": date_semantics,
        "onset_interpretation": onset_interpretation,
        "verification_status": verification,
        "inclusion_decision": inclusion,
        "source_completeness_fields": completeness,
        "source_completeness_out_of_5": 5,
        "date_warnings": ";".join(date_warnings),
        "manual_chronology_review": "required",
    })

chronology = pd.DataFrame(audit_rows)

# Create a human-review template for episode relationships.
review = relationships.copy()

if not review.empty:
    review["adjudication_decision"] = ""
    review["evidence_for_relationship"] = ""
    review["evidence_against_relationship"] = ""
    review["reviewer_notes"] = ""
    review["reviewed_source"] = ""
    review["reviewed_date"] = ""
    review["confirmed_episode_merge"] = "no"
else:
    review = pd.DataFrame(columns=[
        "relationship_id",
        "iso",
        "country",
        "episode_id_1",
        "episode_id_2",
        "relationship_type",
        "review_status",
        "adjudication_decision",
        "evidence_for_relationship",
        "evidence_against_relationship",
        "reviewer_notes",
        "reviewed_source",
        "reviewed_date",
        "confirmed_episode_merge",
    ])

# Validate review choices without pre-populating decisions.
allowed_decisions = {
    "",
    "confirmed_related",
    "plausibly_related",
    "not_established",
}

if "adjudication_decision" in review.columns:
    invalid = set(
        review["adjudication_decision"].fillna("").astype(str)
    ) - allowed_decisions

    if invalid:
        raise ValueError(f"Invalid adjudication values: {invalid}")

# Save outputs.
OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

chronology.to_csv(CHRONOLOGY_OUT, index=False, encoding="utf-8")
review.to_csv(REVIEW_OUT, index=False, encoding="utf-8")

# Re-read saved files.
saved_chronology = pd.read_csv(CHRONOLOGY_OUT)
saved_review = pd.read_csv(REVIEW_OUT)

if len(saved_chronology) != len(registry):
    raise ValueError("Saved chronology row count does not reconcile")

if len(saved_review) != len(relationships):
    raise ValueError("Saved relationship review count does not reconcile")

if saved_chronology["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in chronology output")

if saved_chronology["episode_id"].isna().any():
    raise ValueError("Missing episode IDs in chronology output")

# Report.
lines = [
    "=" * 76,
    "EUROPEAN HISTORICAL STRESS CHRONOLOGY AUDIT",
    "=" * 76,
    "",
    "INPUT RECONCILIATION",
    "-" * 76,
    f"Registry event records: {len(registry)}",
    f"Linked event records: {len(linkage)}",
    f"Unique assigned episodes: {linkage['episode_id'].nunique()}",
    f"Relationship candidates: {len(relationships)}",
    "",
    "SOURCE VERIFICATION",
    "-" * 76,
]

for label, count in (
    registry["verification_status"].fillna("missing").value_counts().items()
):
    lines.append(f"{label}: {count}")

lines.extend([
    "",
    "INCLUSION DECISIONS",
    "-" * 76,
])

for label, count in (
    registry["inclusion_decision"].fillna("missing").value_counts().items()
):
    lines.append(f"{label}: {count}")

lines.extend([
    "",
    "DATE SEMANTICS",
    "-" * 76,
])

for label, count in chronology["date_semantics"].value_counts().items():
    lines.append(f"{label}: {count}")

warning_count = sum(
    bool(value)
    for value in chronology["date_warnings"]
)

lines.extend([
    "",
    "AUDIT FLAGS",
    "-" * 76,
    f"Event records with one or more date/source flags: {warning_count}",
    f"Events requiring manual chronology review: {len(chronology)}",
    f"Relationship decisions awaiting review: {len(review)}",
    "",
    "OUTPUT FILES",
    "-" * 76,
    str(CHRONOLOGY_OUT),
    str(REVIEW_OUT),
    "",
    "METHODOLOGICAL CONTROLS",
    "-" * 76,
    "- Original event registry was not modified.",
    "- Existing episode assignments were preserved.",
    "- Assistance agreement dates are not assumed to be crisis onset.",
    "- Banking-crisis end dates require source-methodology review.",
    "- Restructuring transaction dates are not assumed to be crisis onset.",
    "- Relationship decisions are left blank for human adjudication.",
    "- No event relationships were automatically merged.",
    "",
    "STATUS: VALIDATION PASSED",
])

REPORT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
