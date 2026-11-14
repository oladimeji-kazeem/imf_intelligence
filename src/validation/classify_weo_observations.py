from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
INPUT_FILE = ROOT / "data" / "processed" / "europe_weo_long.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

VINTAGE_YEAR = 2026

# --------------------------------------------------
# 1. Locate the original WEO workbook
# --------------------------------------------------

workbooks = list(RAW_DIR.glob("*.xlsx"))

if len(workbooks) != 1:
    sys.exit(
        f"Expected exactly one WEO Excel workbook in {RAW_DIR}; "
        f"found {len(workbooks)}."
    )

workbook = workbooks[0]

if not INPUT_FILE.exists():
    sys.exit(f"Extracted dataset not found: {INPUT_FILE}")

# --------------------------------------------------
# 2. Load source data
# --------------------------------------------------

print("Loading WEO workbook and European dataset...")

weo = pd.read_excel(workbook, sheet_name="Countries")
europe = pd.read_csv(INPUT_FILE)

# Standardise column names for matching.
weo.columns = weo.columns.astype(str).str.strip()
europe.columns = europe.columns.astype(str).str.strip()

required_weo = [
    "ISO",
    "WEO Subject Code",
    "LATEST_ACTUAL_ANNUAL_DATA",
]

required_europe = ["iso", "weo_code", "year"]

for column in required_weo:
    if column not in weo.columns:
        sys.exit(f"Missing workbook column: {column}")

for column in required_europe:
    if column not in europe.columns:
        sys.exit(f"Missing extracted dataset column: {column}")

# --------------------------------------------------
# 3. Build country-indicator threshold registry
# --------------------------------------------------

thresholds = weo[
    [
        "ISO",
        "WEO Subject Code",
        "LATEST_ACTUAL_ANNUAL_DATA",
    ]
].copy()

thresholds = thresholds.rename(
    columns={
        "ISO": "iso",
        "WEO Subject Code": "weo_code",
        "LATEST_ACTUAL_ANNUAL_DATA": "latest_actual_raw",
    }
)

thresholds["iso"] = thresholds["iso"].astype(str).str.strip()
thresholds["weo_code"] = (
    thresholds["weo_code"].astype(str).str.strip()
)

thresholds["latest_actual_raw"] = (
    thresholds["latest_actual_raw"]
    .astype("string")
    .str.strip()
)

# Keep only records needed by the extracted European dataset.
europe_keys = europe[["iso", "weo_code"]].drop_duplicates()

thresholds = thresholds.merge(
    europe_keys,
    on=["iso", "weo_code"],
    how="inner",
)

# A country-indicator pair must have exactly one threshold.
duplicates = thresholds.duplicated(
    subset=["iso", "weo_code"], keep=False
)

if duplicates.any():
    print(thresholds.loc[duplicates].to_string(index=False))
    sys.exit("Duplicate country-indicator threshold records detected.")

# --------------------------------------------------
# 4. Identify threshold formats
# --------------------------------------------------

def identify_threshold(value):
    if pd.isna(value) or str(value).strip() in {"", "nan", "<NA>"}:
        return "MISSING"

    value = str(value).strip()

    if re.fullmatch(r"\d{4}", value):
        return "NUMERIC_YEAR"

    if re.fullmatch(r"FY\d{4}/\d{2,4}", value, flags=re.IGNORECASE):
        return "FISCAL_YEAR"

    return "OTHER"


thresholds["threshold_type"] = (
    thresholds["latest_actual_raw"].apply(identify_threshold)
)

thresholds["latest_actual_year"] = pd.to_numeric(
    thresholds["latest_actual_raw"].where(
        thresholds["threshold_type"] == "NUMERIC_YEAR"
    ),
    errors="coerce",
)

# --------------------------------------------------
# 5. Join threshold metadata to observations
# --------------------------------------------------

data = europe.merge(
    thresholds,
    on=["iso", "weo_code"],
    how="left",
    validate="many_to_one",
)

if data["threshold_type"].isna().any():
    missing = data.loc[
        data["threshold_type"].isna(),
        ["iso", "weo_code"],
    ].drop_duplicates()

    print(missing.to_string(index=False))
    sys.exit("Some European observations have no matching threshold record.")

data["year"] = pd.to_numeric(data["year"], errors="coerce")

if data["year"].isna().any():
    sys.exit("Invalid observation years detected.")

# --------------------------------------------------
# 6. Classify each observation
# --------------------------------------------------

def classify_observation(row):
    threshold_type = row["threshold_type"]
    year = int(row["year"])

    if threshold_type != "NUMERIC_YEAR":
        return "unclassified"

    cutoff = int(row["latest_actual_year"])

    if year <= cutoff:
        return "historical_actual"

    if year <= VINTAGE_YEAR:
        return "imf_estimate"

    return "imf_projection"


def classification_basis(row):
    if row["threshold_type"] == "NUMERIC_YEAR":
        return (
            "Numeric latest-actual cutoff; "
            "years after cutoff through 2026 treated as estimates; "
            "later years treated as projections."
        )

    if row["threshold_type"] == "FISCAL_YEAR":
        return (
            "Fiscal-year cutoff retained; "
            "calendar-year alignment not established."
        )

    if row["threshold_type"] == "MISSING":
        return "Latest-actual threshold missing in source workbook."

    return "Unrecognised threshold format; requires investigation."


data["observation_status"] = data.apply(
    classify_observation, axis=1
)

data["classification_basis"] = data.apply(
    classification_basis, axis=1
)

# --------------------------------------------------
# 7. Save classified dataset
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "europe_weo_classified.csv"

data.to_csv(output_file, index=False)

# --------------------------------------------------
# 8. Generate classification reports
# --------------------------------------------------

status_summary = (
    data.groupby(
        ["weo_code", "observation_status"],
        dropna=False,
    )
    .size()
    .reset_index(name="records")
)

status_summary.to_csv(
    REPORT_DIR / "europe_observation_status_summary.csv",
    index=False,
)

exception_summary = (
    data.loc[
        data["observation_status"] == "unclassified",
        [
            "iso",
            "country",
            "weo_code",
            "year",
            "latest_actual_raw",
            "threshold_type",
        ],
    ]
    .drop_duplicates()
)

exception_summary.to_csv(
    REPORT_DIR / "europe_observation_status_exceptions.csv",
    index=False,
)

# --------------------------------------------------
# 9. Print validation results
# --------------------------------------------------

print("\nOBSERVATION CLASSIFICATION RESULTS")
print("----------------------------------")

print(f"Input records: {len(europe):,}")
print(f"Output records: {len(data):,}")
print(f"Countries: {data['iso'].nunique()}")
print(f"Indicators: {data['weo_code'].nunique()}")

print("\nClassification counts:")
print(data["observation_status"].value_counts(dropna=False))

print("\nThreshold types:")
print(data["threshold_type"].value_counts(dropna=False))

print("\nClassification by indicator:")
print(
    pd.crosstab(
        data["weo_code"],
        data["observation_status"],
    ).to_string()
)

print("\nUnclassified records:", len(exception_summary))

print("\nGenerated files:")
print(output_file)
print(REPORT_DIR / "europe_observation_status_summary.csv")
print(REPORT_DIR / "europe_observation_status_exceptions.csv")

print("\nCLASSIFICATION COMPLETED.")
