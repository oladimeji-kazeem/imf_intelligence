from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data/raw/WEOOct2026all.xlsx"
OUTPUT = ROOT / "outputs/reports"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    WORKBOOK,
    sheet_name="Countries",
    engine="openpyxl"
)

df.columns = [str(col).strip() for col in df.columns]

column_mapping = {
    "INDICATOR.ID": "WEO Subject Code",
    "INDICATOR": "Subject Descriptor",
    "UNIT": "Units",
    "SCALE": "Scale",
    "COUNTRY.ID": "ISO"
}
for modern_col, legacy_col in column_mapping.items():
    if modern_col in df.columns and legacy_col not in df.columns:
        df[legacy_col] = df[modern_col]

threshold_col = "LATEST_ACTUAL_ANNUAL_DATA"

if threshold_col not in df.columns:
    raise ValueError(
        f"Metadata column not found: {threshold_col}"
    )

# Inspect threshold values by indicator
print("=" * 70)
print("LATEST ACTUAL ANNUAL DATA — THRESHOLD ANALYSIS")
print("=" * 70)

print("\n1. THRESHOLD DISTRIBUTION BY INDICATOR")
print("-" * 70)

threshold_by_indicator = (
    df.groupby("WEO Subject Code")[threshold_col]
    .value_counts(dropna=False)
    .rename("series_count")
    .reset_index()
)

print(threshold_by_indicator.to_string(index=False))

# Inspect fiscal-year and non-standard thresholds
print("\n2. NON-STANDARD THRESHOLD VALUES")
print("-" * 70)

threshold_text = df[threshold_col].astype(str).str.strip()

standard_year = threshold_text.str.fullmatch(r"\d{4}")

non_standard = df.loc[
    ~standard_year,
    ["ISO", "COUNTRY", "WEO Subject Code", threshold_col]
]

print(f"Non-standard or missing records: {len(non_standard)}")

print(non_standard.head(40).to_string(index=False))

# Save detailed results
threshold_by_indicator.to_csv(
    OUTPUT / "weo_thresholds_by_indicator.csv",
    index=False
)

non_standard.to_csv(
    OUTPUT / "weo_nonstandard_thresholds.csv",
    index=False
)

print("\n3. OUTPUT FILES")
print("-" * 70)

print(OUTPUT / "weo_thresholds_by_indicator.csv")
print(OUTPUT / "weo_nonstandard_thresholds.csv")

print("\nINSPECTION COMPLETED.")
print("=" * 70)
