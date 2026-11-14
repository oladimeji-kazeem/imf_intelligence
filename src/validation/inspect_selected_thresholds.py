from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data/raw/WEOOct2026all.xlsx"
OUTPUT = ROOT / "outputs/reports"
OUTPUT.mkdir(parents=True, exist_ok=True)

SELECTED_CODES = [
    "NGDP_RPCH",
    "NGDP",
    "NGDPD",
    "PCPIPCH",
    "GGXWDG_NGDP",
    "GGXCNL_NGDP",
    "GGXONLB_NGDP",
    "LUR",
    "BCA_NGDPD",
    "TX_RPCH",
    "TM_RPCH"
]

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

selected = df[
    df["WEO Subject Code"].isin(SELECTED_CODES)
].copy()

if threshold_col not in selected.columns:
    raise ValueError(f"Missing column: {threshold_col}")

# Preserve the original threshold text
selected["threshold_raw"] = (
    selected[threshold_col].astype("string").str.strip()
)

# Classify threshold format, without interpreting it yet
selected["threshold_format"] = "OTHER"

selected.loc[
    selected["threshold_raw"].str.fullmatch(r"\d{4}", na=False),
    "threshold_format"
] = "NUMERIC_YEAR"

selected.loc[
    selected["threshold_raw"].str.fullmatch(
        r"FY\d{4}/\d{2,4}", case=False, na=False
    ),
    "threshold_format"
] = "FISCAL_YEAR"

selected.loc[
    selected[threshold_col].isna(),
    "threshold_format"
] = "MISSING"

summary = (
    selected.groupby(
        ["WEO Subject Code", "threshold_format", "threshold_raw"],
        dropna=False
    )
    .size()
    .reset_index(name="country_series_count")
    .sort_values(["WEO Subject Code", "threshold_format", "threshold_raw"])
)

print("=" * 70)
print("SELECTED INDICATOR THRESHOLD ANALYSIS")
print("=" * 70)

print("\n1. THRESHOLD DISTRIBUTION")
print(summary.to_string(index=False))

print("\n2. COUNTS BY INDICATOR AND FORMAT")
format_summary = pd.crosstab(
    selected["WEO Subject Code"],
    selected["threshold_format"]
)

print(format_summary.to_string())

print("\n3. COUNTRIES WITH NON-NUMERIC OR MISSING THRESHOLDS")

exceptions = selected[
    selected["threshold_format"] != "NUMERIC_YEAR"
][[
    "ISO",
    "COUNTRY",
    "WEO Subject Code",
    threshold_col,
    "threshold_format"
]]

print(exceptions.to_string(index=False))

# Save audit outputs
summary.to_csv(
    OUTPUT / "selected_indicator_threshold_summary.csv",
    index=False
)

exceptions.to_csv(
    OUTPUT / "selected_indicator_threshold_exceptions.csv",
    index=False
)

print("\n4. SAVED FILES")
print(OUTPUT / "selected_indicator_threshold_summary.csv")
print(OUTPUT / "selected_indicator_threshold_exceptions.csv")

print("\nINSPECTION COMPLETED.")
print("=" * 70)
