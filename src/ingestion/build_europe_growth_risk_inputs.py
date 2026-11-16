
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_weo_classified.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_growth_risk_input_panel.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"

COVERAGE_FILE = (
    REPORT_DIR / "europe_growth_risk_country_coverage.csv"
)

MISSING_FILE = (
    REPORT_DIR / "europe_growth_risk_missingness.csv"
)

REPORT_FILE = (
    REPORT_DIR / "europe_growth_risk_input_report.txt"
)

TARGET_CODE = "NGDP_RPCH"
START_YEAR = 1980
END_YEAR = 2025

EXPECTED_COUNTRIES = [
    "AUT", "BEL", "CHE", "CZE", "DEU",
    "DNK", "ESP", "FIN", "FRA", "GBR",
    "GRC", "HUN", "IRL", "ITA", "NLD",
    "NOR", "POL", "PRT", "ROU", "SWE",
]

print("=" * 78)
print("EUROPEAN GROWTH RISK — INPUT PANEL CONSTRUCTION")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "ISO",
    "country",
    "WEO Subject Code",
    "indicator",
    "indicator_description",
    "year",
    "value",
    "value_raw",
    "unit",
    "scale",
    "source",
    "source_vintage",
    "observation_status",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ---------------------------------------------------------
# 1. Extract real GDP growth historical actual observations
# ---------------------------------------------------------

growth = df[
    df["WEO Subject Code"].astype(str).str.strip().eq(TARGET_CODE)
].copy()

growth = growth[
    growth["observation_status"].eq("historical_actual")
].copy()

growth["year"] = pd.to_numeric(
    growth["year"], errors="coerce"
)

growth["growth_pct"] = pd.to_numeric(
    growth["value"], errors="coerce"
)

growth["growth_raw_pct"] = pd.to_numeric(
    growth["value_raw"], errors="coerce"
)

if growth.empty:
    raise ValueError("No historical actual growth records found.")

# Restrict to the inspected historical period.
growth = growth[
    growth["year"].between(START_YEAR, END_YEAR)
].copy()

growth["year"] = growth["year"].astype(int)

# ---------------------------------------------------------
# 2. Structural validation
# ---------------------------------------------------------

if growth["ISO"].isna().any():
    raise ValueError("Missing country identifiers detected.")

if growth["year"].isna().any():
    raise ValueError("Missing or invalid year values detected.")

actual_countries = sorted(growth["ISO"].unique().tolist())

if actual_countries != sorted(EXPECTED_COUNTRIES):
    raise ValueError(
        "Country coverage differs from the expected "
        f"20-country configuration. Found: {actual_countries}"
    )

expected_records = (
    len(EXPECTED_COUNTRIES)
    * (END_YEAR - START_YEAR + 1)
)

if len(growth) != expected_records:
    raise ValueError(
        f"Unexpected record count: {len(growth)}. "
        f"Expected: {expected_records}"
    )

duplicate_count = growth.duplicated(
    subset=["ISO", "year"]
).sum()

if duplicate_count:
    raise ValueError(
        f"Duplicate country-year records: {duplicate_count}"
    )

# ---------------------------------------------------------
# 3. Compare numeric source fields
# ---------------------------------------------------------

comparison = growth[
    ["growth_pct", "growth_raw_pct"]
].dropna()

source_value_mismatches = (
    ~np.isclose(
        comparison["growth_pct"],
        comparison["growth_raw_pct"],
        rtol=0,
        atol=1e-10,
    )
).sum()

# ---------------------------------------------------------
# 4. Construct the modelling input panel
# ---------------------------------------------------------

panel_columns = [
    "ISO",
    "country",
    "year",
    "growth_pct",
    "growth_raw_pct",
    "unit",
    "scale",
    "source",
    "source_vintage",
    "observation_status",
]

panel = growth[panel_columns].copy()

panel["is_missing_growth"] = panel["growth_pct"].isna()

# Negative real GDP growth indicates an annual contraction.
# Missing values remain missing.
panel["is_contraction"] = pd.Series(
    pd.NA, index=panel.index, dtype="boolean"
)

valid_growth = panel["growth_pct"].notna()

panel.loc[valid_growth, "is_contraction"] = (
    panel.loc[valid_growth, "growth_pct"] < 0
)

panel = panel.sort_values(
    ["ISO", "year"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 5. Country coverage report
# ---------------------------------------------------------

coverage = (
    panel.groupby(["ISO", "country"])
    .agg(
        historical_records=("year", "count"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        numeric_observations=("growth_pct", "count"),
        missing_observations=("growth_pct", lambda x: x.isna().sum()),
        contractions=("is_contraction", lambda x: x.eq(True).sum()),
    )
    .reset_index()
)

coverage["numeric_coverage_pct"] = (
    coverage["numeric_observations"]
    / coverage["historical_records"]
    * 100
)

# ---------------------------------------------------------
# 6. Missingness report
# ---------------------------------------------------------

missing = panel[
    panel["growth_pct"].isna()
][
    ["ISO", "country", "year", "observation_status"]
].copy()

# ---------------------------------------------------------
# 7. Common-period validation
# ---------------------------------------------------------

common_window = panel[
    panel["year"].between(2020, 2024)
].copy()

common_numeric = common_window["growth_pct"].notna().sum()

expected_common_records = len(EXPECTED_COUNTRIES) * 5

if len(common_window) != expected_common_records:
    raise ValueError(
        "2020–2024 country-year record coverage is incomplete."
    )

if common_numeric != expected_common_records:
    raise ValueError(
        "2020–2024 contains missing numerical growth observations. "
        "Investigate before using this as a balanced comparison window."
    )

# ---------------------------------------------------------
# 8. Historical range inspection
# ---------------------------------------------------------

valid_values = panel["growth_pct"].dropna()

minimum_growth = valid_values.min()
maximum_growth = valid_values.max()

minimum_rows = panel[
    panel["growth_pct"].eq(minimum_growth)
]

maximum_rows = panel[
    panel["growth_pct"].eq(maximum_growth)
]

# ---------------------------------------------------------
# 9. Write outputs
# ---------------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUTPUT_FILE, index=False)
coverage.to_csv(COVERAGE_FILE, index=False)
missing.to_csv(MISSING_FILE, index=False)

# ---------------------------------------------------------
# 10. Build validation report
# ---------------------------------------------------------

lines = []


def add(value=""):
    lines.append(str(value))


add("=" * 78)
add("EUROPEAN GROWTH RISK INPUT PANEL REPORT")
add("=" * 78)

add(f"Source: {INPUT_FILE}")
add(f"Indicator: {TARGET_CODE}")
add(f"Period: {START_YEAR}-{END_YEAR}")
add(f"Countries: {panel['ISO'].nunique()}")
add(f"Historical records: {len(panel):,}")
add(f"Expected records: {expected_records:,}")
add(f"Duplicate country-year records: {duplicate_count}")
add(f"Missing growth observations: {panel['growth_pct'].isna().sum()}")
add(f"Numeric growth observations: {panel['growth_pct'].notna().sum()}")
add(f"Source value mismatches: {source_value_mismatches}")

add()
add("COUNTRY COVERAGE")
add("-" * 78)
add(coverage.to_string(index=False))

add()
add("MISSING OBSERVATIONS")
add("-" * 78)

if missing.empty:
    add("No missing growth observations.")
else:
    add(missing.to_string(index=False))

add()
add("2020-2024 COMMON WINDOW")
add("-" * 78)
add(f"Records: {len(common_window)}")
add(f"Numeric observations: {common_numeric}")
add(f"Countries: {common_window['ISO'].nunique()}")
add("Balanced numeric coverage: PASS")

add()
add("HISTORICAL GROWTH RANGE")
add("-" * 78)
add(f"Minimum growth: {minimum_growth:.3f}%")
add(minimum_rows[["ISO", "country", "year"]].to_string(index=False))
add(f"Maximum growth: {maximum_growth:.3f}%")
add(maximum_rows[["ISO", "country", "year"]].to_string(index=False))

add()
add("VALIDATION SUMMARY")
add("-" * 78)

checks = {
    "Expected countries present": (
        actual_countries == sorted(EXPECTED_COUNTRIES)
    ),
    "Expected historical record count": (
        len(panel) == expected_records
    ),
    "No duplicate country-year records": (
        duplicate_count == 0
    ),
    "2020-2024 complete numeric panel": (
        common_numeric == expected_common_records
    ),
    "Source values consistent": (
        source_value_mismatches == 0
    ),
}

for name, passed in checks.items():
    add(f"{name}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise ValueError(
        "One or more growth input validation checks failed."
    )

add()
add("INTERPRETATION")
add("-" * 78)
add(
    "This is an actual-only historical input panel. "
    "Missing growth observations have not been imputed."
)
add(
    "Negative annual real GDP growth is identified as a contraction."
)
add(
    "The 2020-2024 period has complete numeric coverage across "
    "all 20 countries."
)
add(
    "No growth risk classifications, composite scores or forecasts "
    "are produced at this stage."
)

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))

print()
print("OUTPUTS CREATED")
print("-" * 78)
print(OUTPUT_FILE)
print(COVERAGE_FILE)
print(MISSING_FILE)
print(REPORT_FILE)

print()
print("VALIDATION PASSED")
