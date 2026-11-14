from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_historical_observed.csv"
OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

if not INPUT.exists():
    sys.exit(f"Dataset not found: {INPUT}")

df = pd.read_csv(INPUT)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

required = [
    "iso", "country", "indicator", "weo_code",
    "year", "value", "unit", "scale"
]

missing = [c for c in required if c not in df.columns]

if missing:
    sys.exit(f"Missing required columns: {missing}")

# --------------------------------------------------
# 1. Historical country-indicator profiles
# --------------------------------------------------

profiles = (
    df.groupby(
        ["iso", "country", "indicator", "weo_code",
         "unit", "scale"],
        dropna=False
    )
    .agg(
        observations=("value", "count"),
        first_observation_year=("year", "min"),
        latest_observation_year=("year", "max"),
        mean=("value", "mean"),
        median=("value", "median"),
        standard_deviation=("value", "std"),
        minimum=("value", "min"),
        maximum=("value", "max"),
    )
    .reset_index()
)

numeric_columns = [
    "mean", "median", "standard_deviation",
    "minimum", "maximum"
]

profiles[numeric_columns] = profiles[numeric_columns].round(4)

# --------------------------------------------------
# 2. Latest available historical snapshot
# --------------------------------------------------

latest = (
    df.sort_values("year")
    .groupby(
        ["iso", "country", "indicator", "weo_code"],
        as_index=False
    )
    .tail(1)
    .copy()
)

latest = latest.rename(
    columns={
        "year": "latest_observation_year",
        "value": "latest_value"
    }
)

latest = latest[
    [
        "iso", "country", "indicator", "weo_code",
        "latest_observation_year", "latest_value",
        "unit", "scale"
    ]
].sort_values(["iso", "indicator"])

# --------------------------------------------------
# 3. Coverage diagnostics
# --------------------------------------------------

coverage = (
    df.groupby(["iso", "country", "indicator"])
    .agg(
        available_observations=("value", "count"),
        first_year=("year", "min"),
        latest_year=("year", "max")
    )
    .reset_index()
)

# --------------------------------------------------
# 4. Save outputs
# --------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

profiles.to_csv(
    OUTPUT / "europe_country_indicator_profiles.csv",
    index=False
)

latest.to_csv(
    OUTPUT / "europe_latest_economic_snapshot.csv",
    index=False
)

coverage.to_csv(
    REPORTS / "europe_historical_coverage.csv",
    index=False
)

# --------------------------------------------------
# 5. Validation and summary
# --------------------------------------------------

print("\nEUROPEAN COUNTRY ECONOMIC PROFILES")
print("----------------------------------")

print("Countries:", df["iso"].nunique())
print("Indicators:", df["weo_code"].nunique())
print("Input observations:", len(df))
print("Country-indicator profiles:", len(profiles))
print("Latest snapshot records:", len(latest))

print("\nLatest observation years by indicator:")
print(
    latest.groupby("indicator")["latest_observation_year"]
    .agg(["min", "max"])
    .to_string()
)

print("\nCountries represented:")
print(", ".join(sorted(df["iso"].unique())))

print("\nGenerated files:")
print(OUTPUT / "europe_country_indicator_profiles.csv")
print(OUTPUT / "europe_latest_economic_snapshot.csv")
print(REPORTS / "europe_historical_coverage.csv")

print("\nCOUNTRY PROFILE GENERATION COMPLETED.")
