from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/europe_weo_long.csv"
OUTPUT_DIR = ROOT / "outputs/reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "europe_data_quality_report.txt"

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("EUROPEAN WEO DATA QUALITY VALIDATION")
print("=" * 65)

# 1. Basic structure
print("\n1. DATASET STRUCTURE")
print("-" * 65)

print(f"Rows: {len(df):,}")
print(f"Countries: {df['ISO'].nunique()}")
print(f"Indicators: {df['indicator'].nunique()}")
print(f"Years: {df['year'].min()}–{df['year'].max()}")

# 2. Duplicate records
print("\n2. DUPLICATE CHECK")
print("-" * 65)

keys = ["ISO", "WEO Subject Code", "year"]
duplicates = df.duplicated(subset=keys).sum()

print(f"Duplicate country-indicator-year records: {duplicates}")

# 3. Missing values by indicator
print("\n3. MISSING VALUES BY INDICATOR")
print("-" * 65)

indicator_report = (
    df.groupby("indicator")
    .agg(
        total_records=("value", "size"),
        missing_values=("value", lambda x: x.isna().sum()),
        available_values=("value", lambda x: x.notna().sum())
    )
)

indicator_report["missing_pct"] = (
    indicator_report["missing_values"]
    / indicator_report["total_records"] * 100
).round(2)

print(indicator_report.to_string())

# 4. Missing values by country
print("\n4. MISSING VALUES BY COUNTRY")
print("-" * 65)

country_report = (
    df.groupby(["ISO", "country"])
    .agg(
        total_records=("value", "size"),
        missing_values=("value", lambda x: x.isna().sum())
    )
)

country_report["missing_pct"] = (
    country_report["missing_values"]
    / country_report["total_records"] * 100
).round(2)

print(country_report.to_string())

# 5. Missing values by year
print("\n5. MISSING VALUES BY YEAR")
print("-" * 65)

year_report = (
    df.groupby("year")
    .agg(
        total_records=("value", "size"),
        missing_values=("value", lambda x: x.isna().sum())
    )
)

year_report["missing_pct"] = (
    year_report["missing_values"]
    / year_report["total_records"] * 100
).round(2)

print(year_report.to_string())

# 6. Missing values by country and indicator
print("\n6. COUNTRY-INDICATOR MISSINGNESS")
print("-" * 65)

country_indicator = (
    df.groupby(["ISO", "indicator"])
    .agg(
        total_records=("value", "size"),
        missing_values=("value", lambda x: x.isna().sum())
    )
)

country_indicator["missing_pct"] = (
    country_indicator["missing_values"]
    / country_indicator["total_records"] * 100
).round(2)

missing_combinations = country_indicator[
    country_indicator["missing_values"] > 0
].sort_values("missing_pct", ascending=False)

print(missing_combinations.to_string())

# 7. Save diagnostic tables
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("EUROPEAN WEO DATA QUALITY REPORT\n")
    f.write("=" * 65 + "\n\n")

    f.write("INDICATOR MISSINGNESS\n")
    f.write(indicator_report.to_string())
    f.write("\n\nCOUNTRY MISSINGNESS\n")
    f.write(country_report.to_string())
    f.write("\n\nYEAR MISSINGNESS\n")
    f.write(year_report.to_string())
    f.write("\n\nCOUNTRY-INDICATOR MISSINGNESS\n")
    f.write(missing_combinations.to_string())
    f.write(f"\n\nDuplicate records: {duplicates}\n")

print("\n7. VALIDATION SUMMARY")
print("-" * 65)

print(f"Total missing values: {df['value'].isna().sum():,}")
print(f"Missing percentage: {df['value'].isna().mean() * 100:.2f}%")
print(f"Duplicate records: {duplicates}")

print(f"\nReport saved: {REPORT_FILE}")
print("=" * 65)
