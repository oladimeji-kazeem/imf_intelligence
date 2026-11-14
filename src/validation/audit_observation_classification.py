from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_weo_classified.csv"
REPORTS = ROOT / "outputs/reports"

VINTAGE_YEAR = 2026

df = pd.read_csv(INPUT)

# 1. Basic structural checks
print("\n1. STRUCTURAL VALIDATION")
print("------------------------")

print("Total records:", len(df))
print("Countries:", df["iso"].nunique())
print("Indicators:", df["weo_code"].nunique())
print("Unique country-indicator pairs:",
      df[["iso", "weo_code"]].drop_duplicates().shape[0])

expected_pairs = df["iso"].nunique() * df["weo_code"].nunique()
actual_pairs = df[["iso", "weo_code"]].drop_duplicates().shape[0]

print("Expected country-indicator pairs:", expected_pairs)
print("Pair coverage complete:", actual_pairs == expected_pairs)

# 2. Threshold integrity
print("\n2. THRESHOLD INTEGRITY")
print("----------------------")

thresholds = df[
    ["iso", "weo_code", "latest_actual_raw", "threshold_type"]
].drop_duplicates()

print("Unique threshold records:", len(thresholds))
print("\nThreshold types:")
print(thresholds["threshold_type"].value_counts(dropna=False))

print("\nThreshold distribution by indicator:")
print(
    thresholds.groupby(
        ["weo_code", "latest_actual_raw", "threshold_type"],
        dropna=False
    ).size().to_string()
)

# 3. Classification consistency
print("\n3. CLASSIFICATION RULE AUDIT")
print("----------------------------")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["latest_actual_year"] = pd.to_numeric(
    df["latest_actual_year"], errors="coerce"
)

def expected_status(row):
    if row["threshold_type"] != "NUMERIC_YEAR":
        return "unclassified"

    if pd.isna(row["latest_actual_year"]):
        return "unclassified"

    if row["year"] <= row["latest_actual_year"]:
        return "historical_actual"

    if row["year"] <= VINTAGE_YEAR:
        return "imf_estimate"

    return "imf_projection"

df["expected_status"] = df.apply(expected_status, axis=1)

mismatches = df[
    df["observation_status"] != df["expected_status"]
]

print("Classification mismatches:", len(mismatches))

# 4. Data availability by classification
print("\n4. DATA AVAILABILITY")
print("--------------------")

df["value"] = pd.to_numeric(df["value"], errors="coerce")

availability = (
    df.groupby("observation_status")
    .agg(
        total_records=("value", "size"),
        available_values=("value", "count"),
        missing_values=("value", lambda x: x.isna().sum())
    )
)

availability["missing_percentage"] = (
    availability["missing_values"]
    / availability["total_records"] * 100
).round(2)

print(availability.to_string())

# 5. Missing values by indicator and status
print("\n5. MISSING VALUES BY INDICATOR AND STATUS")
print("-----------------------------------------")

missing_by_indicator = (
    df.assign(is_missing=df["value"].isna())
    .groupby(["weo_code", "observation_status"])["is_missing"]
    .sum()
    .unstack(fill_value=0)
)

print(missing_by_indicator.to_string())

# 6. Duplicate and key validation
print("\n6. DUPLICATE VALIDATION")
print("-----------------------")

duplicates = df.duplicated(
    subset=["iso", "weo_code", "year"]
).sum()

print("Duplicate country-indicator-year records:", duplicates)

# 7. Save audit reports
REPORTS.mkdir(parents=True, exist_ok=True)

thresholds.to_csv(
    REPORTS / "europe_threshold_integrity.csv",
    index=False
)

availability.to_csv(
    REPORTS / "europe_data_availability_by_status.csv"
)

missing_by_indicator.to_csv(
    REPORTS / "europe_missing_values_by_status.csv"
)

if len(mismatches) > 0:
    mismatches.to_csv(
        REPORTS / "europe_classification_mismatches.csv",
        index=False
    )

print("\nAUDIT COMPLETED.")