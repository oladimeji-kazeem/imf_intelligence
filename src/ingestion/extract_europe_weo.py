from pathlib import Path
import re
import pandas as pd
import yaml

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data/raw/WEOOct2026all.xlsx"
MAPPING_FILE = ROOT / "config/weo_indicator_mapping.yaml"
COUNTRIES_FILE = ROOT / "config/countries.yaml"

OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "europe_weo_long.csv"

# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

with open(MAPPING_FILE, encoding="utf-8") as f:
    mapping = yaml.safe_load(f)

with open(COUNTRIES_FILE, encoding="utf-8") as f:
    country_config = yaml.safe_load(f)

indicator_map = mapping["indicators"]

country_codes = country_config["regions"]["europe"]["countries"]

code_to_indicator = {
    details["weo_code"]: {
        "internal_name": name,
        "description": details["description"],
        "unit": details["unit"],
        "scale": details["scale"]
    }
    for name, details in indicator_map.items()
}

indicator_codes = list(code_to_indicator.keys())

# --------------------------------------------------
# LOAD WEO WORKBOOK
# --------------------------------------------------

print("=" * 65)
print("IMF ECONOMIC INTELLIGENCE LAB")
print("EUROPEAN WEO DATA EXTRACTION")
print("=" * 65)

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

# Identify country name column
country_column = next(
    (col for col in ["COUNTRY", "Country"] if col in df.columns),
    None
)

if country_column is None:
    raise ValueError("Country name column not found.")

required = [
    "ISO",
    "WEO Subject Code",
    "Subject Descriptor",
    "Units",
    "Scale"
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# --------------------------------------------------
# IDENTIFY YEAR COLUMNS
# --------------------------------------------------

year_columns = [
    col for col in df.columns
    if re.fullmatch(r"\d{4}", str(col).strip())
]

if not year_columns:
    raise ValueError("No annual year columns found.")

year_columns = sorted(year_columns, key=lambda x: int(str(x)))

# --------------------------------------------------
# FILTER COUNTRIES AND INDICATORS
# --------------------------------------------------

df["ISO"] = df["ISO"].astype(str).str.strip()
df["WEO Subject Code"] = (
    df["WEO Subject Code"].astype(str).str.strip()
)

available_countries = set(df["ISO"].unique())
available_indicators = set(df["WEO Subject Code"].unique())

missing_countries = sorted(
    set(country_codes) - available_countries
)

missing_indicators = sorted(
    set(indicator_codes) - available_indicators
)

if missing_countries:
    raise ValueError(f"Countries not found: {missing_countries}")

if missing_indicators:
    raise ValueError(f"Indicators not found: {missing_indicators}")

filtered = df[
    df["ISO"].isin(country_codes)
    & df["WEO Subject Code"].isin(indicator_codes)
].copy()

# --------------------------------------------------
# CONVERT FROM WIDE TO LONG FORMAT
# --------------------------------------------------

id_columns = [
    "ISO",
    country_column,
    "WEO Subject Code",
    "Subject Descriptor",
    "Units",
    "Scale"
]

long_df = filtered.melt(
    id_vars=id_columns,
    value_vars=year_columns,
    var_name="year",
    value_name="value_raw"
)

long_df["year"] = long_df["year"].astype(int)

# Preserve original values and create numeric analytical values
long_df["value"] = pd.to_numeric(
    long_df["value_raw"],
    errors="coerce"
)

# --------------------------------------------------
# ADD INTERNAL INDICATOR METADATA
# --------------------------------------------------

long_df["indicator"] = long_df["WEO Subject Code"].map(
    lambda code: code_to_indicator[code]["internal_name"]
)

long_df["indicator_description"] = long_df["WEO Subject Code"].map(
    lambda code: code_to_indicator[code]["description"]
)

long_df["unit"] = long_df["WEO Subject Code"].map(
    lambda code: code_to_indicator[code]["unit"]
)

long_df["scale"] = long_df["WEO Subject Code"].map(
    lambda code: code_to_indicator[code]["scale"]
)

long_df["source"] = "IMF World Economic Outlook"
long_df["source_vintage"] = mapping["source"]["vintage"]

# --------------------------------------------------
# ORDER OUTPUT COLUMNS
# --------------------------------------------------

output_columns = [
    "ISO",
    country_column,
    "indicator",
    "WEO Subject Code",
    "indicator_description",
    "year",
    "value",
    "value_raw",
    "unit",
    "scale",
    "source",
    "source_vintage"
]

long_df = long_df[output_columns].rename(
    columns={country_column: "country"}
)

long_df = long_df.sort_values(
    ["ISO", "indicator", "year"]
).reset_index(drop=True)

# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

long_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# EXTRACTION SUMMARY
# --------------------------------------------------

print("\nEXTRACTION SUMMARY")
print("-" * 65)

print(f"Countries requested: {len(country_codes)}")
print(f"Countries extracted: {long_df['ISO'].nunique()}")
print(f"Indicators requested: {len(indicator_codes)}")
print(f"Indicators extracted: {long_df['indicator'].nunique()}")
print(f"Year range: {long_df['year'].min()}–{long_df['year'].max()}")
print(f"Total rows: {len(long_df):,}")
print(f"Missing numeric values: {long_df['value'].isna().sum():,}")

print("\nRecords per indicator:")
print(long_df.groupby("indicator").size().to_string())

print("\nSaved dataset:")
print(OUTPUT_FILE)

print("\nEXTRACTION COMPLETED.")
print("=" * 65)
