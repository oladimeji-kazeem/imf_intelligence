from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "external"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "europe_historical_stress_events.csv"
REPORT_FILE = REPORT_DIR / "europe_stress_event_registry_readiness.txt"

COLUMNS = [
    "event_id",
    "iso",
    "country",
    "event_type",
    "event_name",
    "onset_year",
    "end_year",
    "severity",
    "source_organisation",
    "source_title",
    "source_url",
    "source_reference",
    "evidence_note",
    "verification_status",
    "inclusion_decision",
    "exclusion_reason",
]

ALLOWED_EVENT_TYPES = {
    "sovereign_default",
    "sovereign_restructuring",
    "financial_assistance",
    "systemic_banking_crisis",
    "sovereign_market_stress",
    "macroeconomic_recession",
    "inflation_crisis",
    "labour_market_stress",
}

ALLOWED_VERIFICATION = {
    "verified",
    "provisional",
    "unverified",
}

ALLOWED_DECISIONS = {
    "include",
    "exclude",
    "review",
}

# Create a blank registry if one does not already exist.
if not OUTPUT_FILE.exists():
    pd.DataFrame(columns=COLUMNS).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

df = pd.read_csv(OUTPUT_FILE, dtype={"iso": str})

errors = []
warnings = []

# Schema validation
missing_columns = [c for c in COLUMNS if c not in df.columns]

if missing_columns:
    errors.append(f"Missing columns: {missing_columns}")

if not missing_columns:
    # Duplicate event IDs
    if df["event_id"].duplicated().any():
        errors.append("Duplicate event_id values found.")

    # Required fields for populated records
    populated = df.dropna(how="all")

    required = [
        "event_id",
        "iso",
        "country",
        "event_type",
        "onset_year",
        "source_organisation",
        "source_title",
        "source_url",
        "evidence_note",
        "verification_status",
        "inclusion_decision",
    ]

    for column in required:
        missing = populated[column].isna() | (
            populated[column].astype(str).str.strip() == ""
        )

        if missing.any():
            errors.append(
                f"{column}: {int(missing.sum())} populated records "
                "have missing required values."
            )

    # ISO code validation
    valid_iso = populated["iso"].astype(str).str.fullmatch(
        r"[A-Z]{3}"
    )

    if not valid_iso.all():
        errors.append("Invalid ISO alpha-3 country codes found.")

    # Event type validation
    invalid_types = set(
        populated["event_type"].dropna().unique()
    ) - ALLOWED_EVENT_TYPES

    if invalid_types:
        errors.append(f"Invalid event types: {sorted(invalid_types)}")

    # Year validation
    for column in ["onset_year", "end_year"]:
        values = pd.to_numeric(
            populated[column], errors="coerce"
        )

        invalid = values.notna() & (
            (values < 1980) | (values > 2026)
        )

        if invalid.any():
            errors.append(
                f"{column}: years outside the 1980–2026 project window."
            )

    onset = pd.to_numeric(
        populated["onset_year"], errors="coerce"
    )
    end = pd.to_numeric(
        populated["end_year"], errors="coerce"
    )

    invalid_periods = end.notna() & onset.notna() & (end < onset)

    if invalid_periods.any():
        errors.append("Some event end years precede their onset years.")

    # Verification status
    invalid_verification = set(
        populated["verification_status"].dropna().unique()
    ) - ALLOWED_VERIFICATION

    if invalid_verification:
        errors.append(
            f"Invalid verification statuses: "
            f"{sorted(invalid_verification)}"
        )

    # Inclusion decision
    invalid_decisions = set(
        populated["inclusion_decision"].dropna().unique()
    ) - ALLOWED_DECISIONS

    if invalid_decisions:
        errors.append(
            f"Invalid inclusion decisions: {sorted(invalid_decisions)}"
        )

    # Evidence quality warnings
    if not populated.empty:
        if populated["source_reference"].isna().any():
            warnings.append(
                "Some records lack a source reference or page/table identifier."
            )

        if populated["severity"].isna().any():
            warnings.append(
                "Some records have no severity assessment."
            )

        if populated["end_year"].isna().any():
            warnings.append(
                "Some events have no recorded end year. This may be valid "
                "for ongoing or date-limited events."
            )

        unverified_included = populated[
            (populated["verification_status"] != "verified")
            & (populated["inclusion_decision"] == "include")
        ]

        if not unverified_included.empty:
            errors.append(
                "Unverified or provisional events cannot be included."
            )

report = [
    "=" * 72,
    "EUROPEAN HISTORICAL STRESS-EVENT REGISTRY READINESS",
    "=" * 72,
    "",
    f"Registry: {OUTPUT_FILE}",
    f"Records: {len(df)}",
    f"Countries: {df['iso'].nunique() if 'iso' in df.columns else 0}",
    "",
    "VALIDATION",
    "-" * 72,
    f"Errors: {len(errors)}",
    f"Warnings: {len(warnings)}",
    "",
]

if errors:
    report.append("ERRORS")
    report.extend(f"- {item}" for item in errors)
else:
    report.append("No schema or validation errors.")

if warnings:
    report.extend(["", "WARNINGS"])
    report.extend(f"- {item}" for item in warnings)

if len(df) == 0:
    report.extend([
        "",
        "STATUS: TEMPLATE CREATED — NO EVENTS ENTERED",
        "This is not evidence that no historical stress events occurred.",
    ])
elif not errors:
    report.extend([
        "",
        "STATUS: STRUCTURAL VALIDATION PASSED",
        "Event evidence still requires substantive review.",
    ])
else:
    report.extend(["", "STATUS: VALIDATION FAILED"])

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))

if errors:
    raise SystemExit(1)