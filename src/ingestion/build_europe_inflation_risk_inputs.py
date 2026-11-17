
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/europe_weo_classified.csv"

OUTPUT_FILE = (
    ROOT / "data/processed/europe_inflation_risk_input_panel.csv"
)

COVERAGE_FILE = (
    ROOT / "outputs/reports/europe_inflation_risk_country_coverage.csv"
)

MISSINGNESS_FILE = (
    ROOT / "outputs/reports/europe_inflation_risk_missingness.csv"
)

REPORT_FILE = (
    ROOT / "outputs/reports/europe_inflation_risk_input_report.txt"
)

WEO_CODE = "PCPIPCH"
EXPECTED_COUNTRIES = 20
START_YEAR = 1980
END_YEAR = 2025
COMMON_START_YEAR = 2020
COMMON_END_YEAR = 2024


# ============================================================
# LOAD AND VALIDATE SOURCE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

source = pd.read_csv(INPUT_FILE)

required_columns = [
    "ISO",
    "country",
    "year",
    "indicator",
    "weo_code",
    "value",
    "unit",
    "scale",
    "source",
    "source_vintage",
    "observation_status"
]

missing_columns = sorted(
    set(required_columns) - set(source.columns)
)

if missing_columns:
    raise ValueError(
        f"Source file is missing required columns: {missing_columns}"
    )


# ============================================================
# EXTRACT PCPIPCH HISTORICAL ACTUALS
# ============================================================

code_mask = (
    source["weo_code"].astype(str).str.strip().eq(WEO_CODE)
    | source["indicator"].astype(str).str.strip().eq(WEO_CODE)
)

inflation = source.loc[code_mask].copy()

if inflation.empty:
    raise ValueError(f"No records found for WEO code {WEO_CODE}")

inflation["year"] = pd.to_numeric(
    inflation["year"], errors="coerce"
)

inflation["value"] = pd.to_numeric(
    inflation["value"], errors="coerce"
)

inflation = inflation.loc[
    inflation["observation_status"].eq("historical_actual")
    & inflation["year"].between(START_YEAR, END_YEAR)
].copy()

if "ISO" in inflation.columns and "iso" in inflation.columns:
    inflation = inflation.drop(columns=["ISO"])
elif "ISO" in inflation.columns:
    inflation = inflation.rename(columns={"ISO": "iso"})

inflation = inflation.rename(
    columns={
        "value": "inflation_pct"
    }
)

inflation["iso"] = inflation["iso"].astype(str).str.upper()

inflation = inflation[
    [
        "iso",
        "country",
        "year",
        "inflation_pct",
        "unit",
        "scale",
        "source",
        "source_vintage",
        "observation_status"
    ]
].copy()

inflation = inflation.sort_values(
    ["iso", "year"]
).reset_index(drop=True)


# ============================================================
# DUPLICATE AND COVERAGE CHECKS
# ============================================================

duplicates = inflation.duplicated(
    ["iso", "year"],
    keep=False
)

if duplicates.any():
    raise ValueError(
        "Duplicate country-year records found:\n"
        + inflation.loc[duplicates].to_string(index=False)
    )

countries = sorted(inflation["iso"].dropna().unique())

if len(countries) != EXPECTED_COUNTRIES:
    raise ValueError(
        f"Expected {EXPECTED_COUNTRIES} countries; "
        f"found {len(countries)}"
    )

expected_years = END_YEAR - START_YEAR + 1
expected_records = EXPECTED_COUNTRIES * expected_years

if len(inflation) != expected_records:
    raise ValueError(
        f"Expected {expected_records} country-year records; "
        f"found {len(inflation)}"
    )


# ============================================================
# ADD DERIVED DIAGNOSTIC FIELDS
# ============================================================

inflation["is_missing_inflation"] = (
    inflation["inflation_pct"].isna()
)

inflation["is_deflation"] = (
    inflation["inflation_pct"].notna()
    & (inflation["inflation_pct"] < 0)
)

inflation["is_high_inflation_10pct"] = (
    inflation["inflation_pct"].notna()
    & (inflation["inflation_pct"] >= 10)
)

inflation["is_high_inflation_20pct"] = (
    inflation["inflation_pct"].notna()
    & (inflation["inflation_pct"] >= 20)
)


# ============================================================
# COUNTRY COVERAGE
# ============================================================

coverage = (
    inflation.groupby("iso")
    .agg(
        country=("country", "first"),
        records=("year", "count"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        numeric_observations=("inflation_pct", "count"),
        missing_values=("inflation_pct", lambda x: x.isna().sum()),
        mean_inflation_pct=("inflation_pct", "mean"),
        median_inflation_pct=("inflation_pct", "median"),
        standard_deviation_pct=("inflation_pct", "std"),
        minimum_inflation_pct=("inflation_pct", "min"),
        maximum_inflation_pct=("inflation_pct", "max"),
        deflation_years=("is_deflation", "sum"),
        high_inflation_years_10pct=("is_high_inflation_10pct", "sum"),
        high_inflation_years_20pct=("is_high_inflation_20pct", "sum")
    )
    .reset_index()
)

coverage["numeric_coverage_pct"] = (
    coverage["numeric_observations"]
    / coverage["records"]
    * 100
)


# ============================================================
# MISSINGNESS BY COUNTRY AND YEAR
# ============================================================

missingness = (
    inflation.loc[inflation["is_missing_inflation"]]
    .groupby(["iso", "country"], as_index=False)
    .agg(
        missing_years=("year", lambda x: ", ".join(
            str(int(y)) for y in sorted(x.dropna())
        )),
        missing_count=("year", "count")
    )
)

if missingness.empty:
    missingness = pd.DataFrame(
        columns=["iso", "country", "missing_years", "missing_count"]
    )


# ============================================================
# COMMON PERIOD VALIDATION
# ============================================================

common_period = inflation.loc[
    inflation["year"].between(
        COMMON_START_YEAR,
        COMMON_END_YEAR
    )
].copy()

expected_common_records = (
    EXPECTED_COUNTRIES
    * (COMMON_END_YEAR - COMMON_START_YEAR + 1)
)

common_numeric = int(
    common_period["inflation_pct"].notna().sum()
)

common_coverage = (
    common_period.groupby("iso")
    .agg(
        observations=("inflation_pct", "count"),
        missing_values=("inflation_pct", lambda x: x.isna().sum())
    )
)

common_complete_countries = int(
    (
        common_coverage["observations"]
        == COMMON_END_YEAR - COMMON_START_YEAR + 1
    ).sum()
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

checks = []

checks.append((
    "Expected historical record count",
    len(inflation) == expected_records
))

checks.append((
    "Expected country coverage",
    len(countries) == EXPECTED_COUNTRIES
))

checks.append((
    "No duplicate country-year records",
    not inflation.duplicated(["iso", "year"]).any()
))

checks.append((
    "Historical observations only",
    inflation["observation_status"].eq("historical_actual").all()
))

checks.append((
    "All records use PCPIPCH",
    (
        source.loc[code_mask, "weo_code"]
        .astype(str)
        .str.strip()
        .eq(WEO_CODE)
    ).all()
))

checks.append((
    "No missing values imputed",
    inflation["is_missing_inflation"].equals(
        inflation["inflation_pct"].isna()
    )
))

checks.append((
    "Common period has expected record count",
    len(common_period) == expected_common_records
))

checks.append((
    "Common period has complete numeric coverage",
    common_numeric == expected_common_records
))

checks.append((
    "All 20 countries complete in common period",
    common_complete_countries == EXPECTED_COUNTRIES
))

checks.append((
    "No impossible inflation flags",
    not (
        inflation["is_deflation"]
        & inflation["is_high_inflation_10pct"]
    ).any()
))


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

inflation.to_csv(OUTPUT_FILE, index=False)
coverage.to_csv(COVERAGE_FILE, index=False)
missingness.to_csv(MISSINGNESS_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

lines = [
    "=" * 75,
    "EUROPEAN INFLATION RISK INPUT PANEL",
    "=" * 75,
    "",
    f"Source: {INPUT_FILE}",
    f"WEO indicator: {WEO_CODE}",
    f"Historical period: {START_YEAR}-{END_YEAR}",
    f"Source vintage: October 2026",
    "",
    f"Countries: {len(countries)}",
    f"Historical country-year records: {len(inflation)}",
    f"Numeric observations: {inflation['inflation_pct'].notna().sum()}",
    f"Missing observations: {inflation['inflation_pct'].isna().sum()}",
    f"Duplicate country-year records: "
    f"{inflation.duplicated(['iso', 'year']).sum()}",
    "",
    "COUNTRY COVERAGE",
    "-" * 75,
    coverage.to_string(index=False),
    "",
    "MISSINGNESS",
    "-" * 75,
    missingness.to_string(index=False),
    "",
    "COMMON PERIOD",
    "-" * 75,
    f"Period: {COMMON_START_YEAR}-{COMMON_END_YEAR}",
    f"Expected records: {expected_common_records}",
    f"Available records: {len(common_period)}",
    f"Numeric observations: {common_numeric}",
    f"Countries with complete numeric coverage: "
    f"{common_complete_countries}/{EXPECTED_COUNTRIES}",
    "",
    "VALIDATION",
    "-" * 75
]

for label, passed in checks:
    lines.append(
        f"{label}: {'PASS' if passed else 'FAIL'}"
    )

all_passed = all(passed for _, passed in checks)

lines.append("")
lines.append(
    "VALIDATION PASSED" if all_passed else "VALIDATION FAILED"
)

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n".join(lines))

print("")
print("OUTPUTS CREATED")
print(OUTPUT_FILE)
print(COVERAGE_FILE)
print(MISSINGNESS_FILE)
print(REPORT_FILE)

if not all_passed:
    raise SystemExit(1)
