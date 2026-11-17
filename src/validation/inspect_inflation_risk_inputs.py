
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/europe_weo_classified.csv"
REPORT_FILE = ROOT / "outputs/reports/europe_inflation_input_inspection.txt"

WEO_CODE = "PCPIPCH"

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = [
    "ISO",
    "country",
    "year",
    "indicator",
    "weo_code",
    "value",
    "observation_status"
]

missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

code_mask = (
    df["weo_code"].astype(str).str.strip().eq(WEO_CODE)
    | df["indicator"].astype(str).str.strip().eq(WEO_CODE)
)

inflation = df.loc[code_mask].copy()

if inflation.empty:
    raise ValueError(
        f"No records found for WEO code {WEO_CODE}. "
        "Inspect the available WEO code fields."
    )

inflation["year"] = pd.to_numeric(
    inflation["year"], errors="coerce"
)
inflation["value"] = pd.to_numeric(
    inflation["value"], errors="coerce"
)

inflation = inflation.loc[
    inflation["observation_status"].eq("historical_actual")
].copy()

if inflation.duplicated(["ISO", "year"]).any():
    raise ValueError("Duplicate country-year inflation records detected.")

countries = sorted(inflation["ISO"].dropna().unique())

coverage = (
    inflation.groupby("ISO")
    .agg(
        country=("country", "first"),
        records=("year", "count"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        numeric_observations=("value", "count"),
        missing_values=("value", lambda x: x.isna().sum()),
        mean_inflation=("value", "mean"),
        median_inflation=("value", "median"),
        minimum_inflation=("value", "min"),
        maximum_inflation=("value", "max")
    )
    .reset_index()
)

common_period = inflation.loc[
    inflation["year"].between(2020, 2024)
].copy()

common_coverage = (
    common_period.groupby("ISO")
    .agg(
        observations=("value", "count"),
        missing_values=("value", lambda x: x.isna().sum()),
        mean_inflation=("value", "mean"),
        median_inflation=("value", "median"),
        maximum_inflation=("value", "max")
    )
    .reset_index()
)

expected_common = len(countries) * 5
actual_common = len(common_period)

lines = [
    "=" * 75,
    "EUROPEAN INFLATION RISK INPUT INSPECTION",
    "=" * 75,
    "",
    f"Input: {INPUT_FILE}",
    f"WEO indicator: {WEO_CODE}",
    f"Historical records: {len(inflation)}",
    f"Countries: {len(countries)}",
    f"Country-year duplicates: {inflation.duplicated(['ISO', 'year']).sum()}",
    f"Numeric observations: {inflation['value'].notna().sum()}",
    f"Missing numeric observations: {inflation['value'].isna().sum()}",
    "",
    "COUNTRY COVERAGE",
    "-" * 75,
    coverage.to_string(index=False),
    "",
    "COMMON PERIOD: 2020-2024",
    "-" * 75,
    f"Expected country-year observations: {expected_common}",
    f"Available country-year records: {actual_common}",
    common_coverage.to_string(index=False),
    "",
    "SOURCE METADATA",
    "-" * 75,
]

for column in [
    "WEO Subject Code",
    "weo_code",
    "indicator",
    "subject_descriptor",
    "subject",
    "unit",
    "scale",
    "source",
    "source_vintage"
]:
    if column in df.columns:
        values = df.loc[code_mask, column].dropna().unique()
        lines.append(f"{column}: {list(values[:10])}")

lines.extend([
    "",
    "VALIDATION",
    "-" * 75,
    f"Code records found: {'PASS' if len(inflation) > 0 else 'FAIL'}",
    f"20-country coverage: {'PASS' if len(countries) == 20 else 'CHECK'}",
    f"No duplicate country-year records: "
    f"{'PASS' if not inflation.duplicated(['ISO', 'year']).any() else 'FAIL'}",
    f"Common-period record count: "
    f"{'PASS' if actual_common == expected_common else 'CHECK'}",
    "",
    "Note: A complete record count does not guarantee complete numeric values.",
    "No missing observations have been imputed.",
])

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("\nREPORT CREATED")
print(REPORT_FILE)
