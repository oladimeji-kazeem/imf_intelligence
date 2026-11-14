from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "Country Indicator Profiles":
        ROOT / "data/processed/europe_country_indicator_profiles.csv",

    "Latest Economic Snapshot":
        ROOT / "data/processed/europe_latest_economic_snapshot.csv",

    "Historical Coverage":
        ROOT / "outputs/reports/europe_historical_coverage.csv",
}

print("\nEUROPEAN ECONOMIC PROFILE SCHEMA INSPECTION")
print("=" * 65)

for name, path in FILES.items():

    print(f"\n\nDATASET: {name}")
    print("-" * 65)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    print(f"File: {path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names and data types:")
    for column, dtype in df.dtypes.items():
        print(f"  {column}: {dtype}")

    print("\nMissing values by column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() or "No missing values")

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nNumeric summary:")
    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        print(numeric.describe().round(3).to_string())
    else:
        print("No numeric columns detected.")

print("\n\nSCHEMA INSPECTION COMPLETED.")
