from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_weo_classified.csv"
OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

if not INPUT.exists():
    sys.exit(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

required = [
    "iso",
    "country",
    "weo_code",
    "indicator",
    "year",
    "value",
    "observation_status",
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    sys.exit(f"Missing required columns: {missing_columns}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# 1. Split by observation status.
historical = df[
    df["observation_status"] == "historical_actual"
].copy()

estimates = df[
    df["observation_status"] == "imf_estimate"
].copy()

projections = df[
    df["observation_status"] == "imf_projection"
].copy()

# 2. Create a historical dataset containing all records.
# Missing values are deliberately retained.
historical_panel = historical.copy()

# 3. Create an observed-only dataset for modelling.
# This excludes records with missing economic values.
historical_observed = historical[
    historical["value"].notna()
].copy()

# 4. Generate missing-data diagnostics.
missing_by_indicator = (
    historical.assign(is_missing=historical["value"].isna())
    .groupby("weo_code")
    .agg(
        total_records=("value", "size"),
        available_values=("value", "count"),
        missing_values=("is_missing", "sum"),
    )
)

missing_by_indicator["missing_percentage"] = (
    missing_by_indicator["missing_values"]
    / missing_by_indicator["total_records"] * 100
).round(2)

missing_by_country = (
    historical.assign(is_missing=historical["value"].isna())
    .groupby("iso")
    .agg(
        total_records=("value", "size"),
        available_values=("value", "count"),
        missing_values=("is_missing", "sum"),
    )
)

missing_by_country["missing_percentage"] = (
    missing_by_country["missing_values"]
    / missing_by_country["total_records"] * 100
).round(2)

# 5. Check dataset integrity.
key = ["iso", "weo_code", "year"]

if historical_panel.duplicated(key).any():
    sys.exit("Duplicate records detected in historical panel.")

if historical_observed["value"].isna().any():
    sys.exit("Missing values found in observed-only dataset.")

# 6. Save separate datasets.
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

historical_panel.to_csv(
    OUTPUT / "europe_historical_panel.csv",
    index=False
)

historical_observed.to_csv(
    OUTPUT / "europe_historical_observed.csv",
    index=False
)

estimates.to_csv(
    OUTPUT / "europe_imf_estimates.csv",
    index=False
)

projections.to_csv(
    OUTPUT / "europe_imf_projections.csv",
    index=False
)

missing_by_indicator.to_csv(
    REPORTS / "historical_missingness_by_indicator.csv"
)

missing_by_country.to_csv(
    REPORTS / "historical_missingness_by_country.csv"
)

# 7. Print reconciliation.
print("\nMODELLING DATASET BUILD")
print("-----------------------")

print("Original records:", len(df))
print("Historical panel:", len(historical_panel))
print("Historical observed:", len(historical_observed))
print("Historical missing:", historical["value"].isna().sum())
print("IMF estimates:", len(estimates))
print("IMF projections:", len(projections))

print("\nHistorical year range:")
print(
    int(historical["year"].min()),
    "to",
    int(historical["year"].max())
)

print("\nMissingness by indicator:")
print(missing_by_indicator.to_string())

print("\nOutput files created:")
for filename in [
    "europe_historical_panel.csv",
    "europe_historical_observed.csv",
    "europe_imf_estimates.csv",
    "europe_imf_projections.csv",
]:
    print(OUTPUT / filename)

print("\nMODELLING DATASETS CREATED.")
