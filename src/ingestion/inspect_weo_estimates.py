from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "data/raw/WEOOct2026all.xlsx"

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

print("=" * 65)
print("WEO ESTIMATES AND PROJECTIONS INSPECTION")
print("=" * 65)

# Identify year columns
year_columns = [
    col for col in df.columns
    if str(col).strip().isdigit()
    and len(str(col).strip()) == 4
]

# Identify potentially relevant metadata columns
keywords = [
    "estimate",
    "projection",
    "forecast",
    "status",
    "note"
]

metadata_columns = [
    col for col in df.columns
    if any(word in str(col).lower() for word in keywords)
]

print("\n1. POTENTIAL ESTIMATION METADATA")
print("-" * 65)

if metadata_columns:
    for col in metadata_columns:
        print(f"\nColumn: {col}")
        print("Unique values:")
        print(df[col].value_counts(dropna=False).head(20).to_string())
else:
    print("No obvious estimation metadata columns found.")

print("\n2. ALL NON-YEAR COLUMN NAMES")
print("-" * 65)

non_year_columns = [
    col for col in df.columns
    if col not in year_columns
]

for col in non_year_columns:
    print(f"- {col}")

# Inspect the most likely estimation threshold field
estimate_columns = [
    col for col in df.columns
    if "estimates start after" in str(col).lower()
    or "latest_actual_annual_data" in str(col).lower()
]

print("\n3. ESTIMATION THRESHOLD INSPECTION")
print("-" * 65)

if estimate_columns:

    estimate_col = estimate_columns[0]

    country_col = next(
        (c for c in ["COUNTRY", "Country"] if c in df.columns),
        None
    )

    display_columns = [
        c for c in [
            "ISO",
            country_col,
            "WEO Subject Code",
            "Subject Descriptor",
            estimate_col
        ]
        if c is not None and c in df.columns
    ]

    print(df[display_columns].head(25).to_string(index=False))

    print("\nThreshold distribution:")
    print(
        df[estimate_col]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

else:
    print("No 'Estimates Start After' column found.")

print("\n4. YEAR COLUMNS")
print("-" * 65)

print(f"First year: {min(map(int, year_columns))}")
print(f"Last year: {max(map(int, year_columns))}")
print(f"Number of years: {len(year_columns)}")

print("\nINSPECTION COMPLETED.")
print("=" * 65)
