
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data" / "external"
    / "esm_domestic_sovereign_defaults.xlsx"
)

REGISTRY = (
    ROOT / "data" / "external"
    / "europe_historical_stress_events.csv"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "europe_greece_psi_population_report.txt"
)

SHEET = "2022 version"
EPISODE_ID = "EU_SOV_GRC_PSI_2011_2012"

SOURCE_ORG = "European Stability Mechanism"
SOURCE_TITLE = (
    "A Journey in the History of Sovereign Defaults "
    "on Domestic-Law Public Debt"
)

REQUIRED_SOURCE_COLUMNS = [
    "Country",
    "Start date",
    "End date",
    "Instrument",
    "Amount (mln US$)",
    "Face value reduction",
    "Maturity change",
    "Coupon change",
    "NPV loss (%)",
    "Post-default restructuring",
]

ADDITIONAL_COLUMNS = [
    "episode_id",
    "instrument",
    "amount_usd_mn",
    "face_value_reduction",
    "maturity_change",
    "coupon_change",
    "npv_loss_pct",
    "post_default_restructuring",
    "event_start_date",
    "event_end_date",
    "event_linkage_note",
]

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not REGISTRY.exists():
    raise FileNotFoundError(REGISTRY)

# Read source workbook.
source_df = pd.read_excel(
    SOURCE,
    sheet_name=SHEET
)

missing_source = (
    set(REQUIRED_SOURCE_COLUMNS) - set(source_df.columns)
)

if missing_source:
    raise ValueError(
        f"Unexpected ESM workbook schema. Missing: "
        f"{sorted(missing_source)}"
    )

# Identify the Greek domestic-law restructuring records.
greek = source_df[
    source_df["Country"].astype(str).str.strip().str.casefold()
    == "greece"
].copy()

if len(greek) != 2:
    raise ValueError(
        f"Expected exactly two Greek records; found {len(greek)}. "
        "Inspect the source before proceeding."
    )

# Convert and validate dates.
greek["Start date"] = pd.to_datetime(
    greek["Start date"], errors="coerce"
)

greek["End date"] = pd.to_datetime(
    greek["End date"], errors="coerce"
)

if greek[["Start date", "End date"]].isna().any().any():
    raise ValueError("Missing or invalid restructuring dates.")

if not (
    (greek["Start date"].dt.year == 2011)
    & (greek["Start date"].dt.month == 10)
    & (greek["Start date"].dt.day == 1)
).all():
    raise ValueError("Unexpected Greek restructuring start dates.")

if not (
    (greek["End date"].dt.year == 2012)
    & (greek["End date"].dt.month == 3)
    & (greek["End date"].dt.day == 1)
).all():
    raise ValueError("Unexpected Greek restructuring end dates.")

# Convert financial amounts and NPV losses.
for column in [
    "Amount (mln US$)",
    "NPV loss (%)",
    "Face value reduction",
    "Maturity change",
    "Coupon change",
    "Post-default restructuring",
]:
    greek[column] = pd.to_numeric(
        greek[column], errors="coerce"
    )

if greek["Amount (mln US$)"].isna().any():
    raise ValueError("Invalid restructuring amount.")

if greek["NPV loss (%)"].isna().any():
    raise ValueError("Invalid NPV loss values.")

if (greek["Amount (mln US$)"] <= 0).any():
    raise ValueError("Non-positive restructuring amount found.")

# Read existing registry.
existing = pd.read_csv(
    REGISTRY,
    dtype={"iso": str}
)

required_registry_columns = [
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

missing_registry = (
    set(required_registry_columns) - set(existing.columns)
)

if missing_registry:
    raise ValueError(
        f"Registry schema missing: {sorted(missing_registry)}"
    )

# Extend the schema while preserving all existing records.
for column in ADDITIONAL_COLUMNS:
    if column not in existing.columns:
        existing[column] = pd.NA

# Construct two tranche records linked to one episode.
new_records = []

for source_row_number, (_, row) in enumerate(
    greek.iterrows(), start=1
):
    instrument = str(row["Instrument"]).strip()

    slug = re.sub(
        r"[^A-Z0-9]+",
        "_",
        instrument.upper()
    ).strip("_")

    event_id = f"{EPISODE_ID}_{slug}"

    new_records.append({
        "event_id": event_id,
        "iso": "GRC",
        "country": "Greece",
        "event_type": "sovereign_restructuring",
        "event_name": (
            "Greek Private Sector Involvement "
            "(PSI) domestic-law restructuring"
        ),
        "onset_year": 2011,
        "end_year": 2012,
        "severity": "documented_domestic_law_restructuring",
        "source_organisation": SOURCE_ORG,
        "source_title": SOURCE_TITLE,
        "source_url": (
            "https://www.esm.europa.eu/publications/"
            "journey-history-sovereign-defaults-domestic-law-public-debt"
        ),
        "source_reference": (
            f"{SHEET} worksheet; Greek source record "
            f"{source_row_number}"
        ),
        "evidence_note": (
            "Instrument-level tranche of the Greek 2011–2012 "
            "PSI restructuring. Both tranche records share one "
            "episode identifier and must count as one episode "
            "in crisis-event validation."
        ),
        "verification_status": "verified",
        "inclusion_decision": "include",
        "exclusion_reason": "",
        "episode_id": EPISODE_ID,
        "instrument": instrument,
        "amount_usd_mn": row["Amount (mln US$)"],
        "face_value_reduction": row["Face value reduction"],
        "maturity_change": row["Maturity change"],
        "coupon_change": row["Coupon change"],
        "npv_loss_pct": row["NPV loss (%)"],
        "post_default_restructuring": (
            row["Post-default restructuring"]
        ),
        "event_start_date": row["Start date"].date().isoformat(),
        "event_end_date": row["End date"].date().isoformat(),
        "event_linkage_note": (
            "Same underlying Greek PSI episode; do not count "
            "as a separate sovereign crisis."
        ),
    })

new = pd.DataFrame(new_records)

# Prevent accidental duplicate imports.
existing_ids = set(
    existing["event_id"].dropna().astype(str)
)

duplicates = existing_ids.intersection(
    set(new["event_id"])
)

if duplicates:
    raise ValueError(
        f"These tranche records already exist: {sorted(duplicates)}"
    )

# Validate source totals against inspected figures.
source_total = greek["Amount (mln US$)"].sum()
expected_total = 250510.0

if abs(source_total - expected_total) > 0.01:
    raise ValueError(
        f"Source amount total differs from inspected value: "
        f"{source_total}"
    )

# Append records.
combined = pd.concat(
    [existing, new],
    ignore_index=True
)

# Validate final dataset.
checks = {
    "two_tranche_records": len(new) == 2,
    "one_shared_episode": new["episode_id"].nunique() == 1,
    "unique_tranche_ids": new["event_id"].nunique() == 2,
    "correct_country": (new["iso"] == "GRC").all(),
    "correct_event_type": (
        new["event_type"] == "sovereign_restructuring"
    ).all(),
    "source_amount_reconciliation": (
        abs(new["amount_usd_mn"].sum() - expected_total) < 0.01
    ),
    "no_duplicate_registry_ids": (
        not combined["event_id"].duplicated().any()
    ),
    "source_dates_valid": (
        (pd.to_datetime(new["event_start_date"]).dt.year == 2011)
        & (pd.to_datetime(new["event_end_date"]).dt.year == 2012)
    ).all(),
}

errors = [
    name for name, passed in checks.items()
    if not passed
]

if errors:
    raise ValueError(
        "Validation failed: " + ", ".join(errors)
    )

# Write only after all checks pass.
combined.to_csv(
    REGISTRY,
    index=False,
    encoding="utf-8"
)

# Re-read the saved file.
saved = pd.read_csv(REGISTRY)

if saved["event_id"].duplicated().any():
    raise ValueError("Saved registry contains duplicate event IDs.")

report = [
    "=" * 72,
    "GREEK PSI SOVEREIGN RESTRUCTURING POPULATION",
    "=" * 72,
    "",
    f"New tranche records: {len(new)}",
    f"New episode IDs: {new['episode_id'].nunique()}",
    f"Total registry records: {len(saved)}",
    f"Source amount total: ${source_total:,.2f} million",
    "",
    "TRANCHE DETAILS",
    "-" * 72,
]

for _, row in new.iterrows():
    report.extend([
        f"Instrument: {row['instrument']}",
        f"Amount: ${row['amount_usd_mn']:,.2f} million",
        f"Face value reduction: {row['face_value_reduction']}",
        f"NPV loss: {row['npv_loss_pct']:.2f}%",
        f"Start: {row['event_start_date']}",
        f"End: {row['event_end_date']}",
        f"Episode ID: {row['episode_id']}",
        "",
    ])

report.extend([
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
    "QUALIFICATION",
    "-" * 72,
    "Two tranche records represent one underlying PSI episode.",
    "Amounts are preserved in million US dollars as reported.",
    "The source dataset concerns domestic-law debt.",
    "Do not infer that this is Greece's complete sovereign debt loss.",
    "",
    "STATUS: POPULATION VALIDATED",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
