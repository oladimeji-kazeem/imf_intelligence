
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "europe_weo_classified.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_FILE = (
    REPORT_DIR / "europe_growth_risk_input_inspection.txt"
)

TARGET_CODE = "NGDP_RPCH"

print("=" * 78)
print("EUROPEAN GROWTH RISK — CORRECTED INPUT INSPECTION")
print("=" * 78)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
lines = []


def add(value=""):
    lines.append(str(value))


def find_column(candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


# Explicit schema mapping
country_col = find_column(["ISO", "iso"])
indicator_col = find_column(["WEO Subject Code", "weo_code"])
year_col = find_column(["year"])
value_col = find_column(["value"])
status_col = find_column(["observation_status"])

required = {
    "country": country_col,
    "indicator": indicator_col,
    "year": year_col,
    "value": value_col,
    "status": status_col,
}

add("=" * 78)
add("EUROPEAN GROWTH RISK INPUT INSPECTION")
add("=" * 78)

add(f"Input: {INPUT_FILE}")
add(f"Rows: {len(df):,}")
add(f"Columns: {len(df.columns)}")

add()
add("SCHEMA MAPPING")
add("-" * 78)

for name, column in required.items():
    add(f"{name}: {column}")

missing_fields = [
    name for name, column in required.items()
    if column is None
]

if missing_fields:
    add(f"Missing required fields: {missing_fields}")
    raise ValueError(
        f"Required schema fields not found: {missing_fields}"
    )

add()
add("INDICATOR CODE CROSS-CHECK")
add("-" * 78)

for column in ["indicator", "WEO Subject Code", "weo_code"]:
    if column in df.columns:
        matches = df[column].astype(str).eq(TARGET_CODE).sum()
        add(f"{column}: {matches} matches")

growth = df[
    df[indicator_col].astype(str).str.strip().eq(TARGET_CODE)
].copy()

add()
add("TARGET INDICATOR")
add("-" * 78)
add(f"Code: {TARGET_CODE}")
add(f"Rows: {len(growth):,}")

if growth.empty:
    add("ERROR: Target code not found in selected code column.")
    add("Available WEO codes:")
    add(
        df[indicator_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_string(index=False)
    )
else:
    add()
    add("INDICATOR METADATA")
    add("-" * 78)

    for column in [
        "indicator",
        "WEO Subject Code",
        "indicator_description",
        "unit",
        "scale",
        "source_vintage",
    ]:
        if column in growth.columns:
            add(
                f"{column}: "
                f"{growth[column].dropna().unique().tolist()}"
            )

    add()
    add("OBSERVATION STATUS")
    add("-" * 78)

    add(
        growth[status_col]
        .value_counts(dropna=False)
        .to_string()
    )

    add()
    add("COUNTRY COVERAGE")
    add("-" * 78)

    add(f"Unique countries: {growth[country_col].nunique()}")
    add(
        growth[country_col]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    add()
    add("YEAR COVERAGE")
    add("-" * 78)

    growth[year_col] = pd.to_numeric(
        growth[year_col], errors="coerce"
    )

    add(f"Minimum year: {growth[year_col].min()}")
    add(f"Maximum year: {growth[year_col].max()}")
    add(f"Unique years: {growth[year_col].nunique()}")

    add()
    add("HISTORICAL ACTUAL COVERAGE BY COUNTRY")
    add("-" * 78)

    actual = growth[
        growth[status_col].eq("historical_actual")
    ].copy()

    coverage = (
        actual.groupby(country_col)
        .agg(
            observations=(year_col, "count"),
            first_actual_year=(year_col, "min"),
            last_actual_year=(year_col, "max"),
            numeric_values=(value_col, "count"),
        )
        .reset_index()
    )

    add(coverage.to_string(index=False))

    add()
    add("HISTORICAL ACTUAL VALUE QUALITY")
    add("-" * 78)

    add(f"Actual records: {len(actual):,}")
    add(f"Missing numeric values: {actual[value_col].isna().sum()}")
    add(
        f"Non-numeric values: "
        f"{pd.to_numeric(actual[value_col], errors='coerce').isna().sum()}"
    )

    numeric_actual = pd.to_numeric(
        actual[value_col], errors="coerce"
    )

    add(f"Minimum growth: {numeric_actual.min()}")
    add(f"Maximum growth: {numeric_actual.max()}")

    add()
    add("COUNTRY-YEAR DUPLICATE CHECK")
    add("-" * 78)

    duplicate_keys = [country_col, year_col]

    duplicate_count = growth.duplicated(
        subset=duplicate_keys,
        keep=False,
    ).sum()

    add(f"Key: {duplicate_keys}")
    add(f"Duplicate rows: {duplicate_count}")

    if duplicate_count:
        add("Duplicate records:")
        add(
            growth.loc[
                growth.duplicated(
                    subset=duplicate_keys, keep=False
                )
            ].sort_values(duplicate_keys).to_string(index=False)
        )

    add()
    add("HISTORICAL ACTUAL COMMON WINDOW")
    add("-" * 78)

    if not coverage.empty:
        common_start = int(coverage["first_actual_year"].max())
        common_end = int(coverage["last_actual_year"].min())

        add(f"Common start year: {common_start}")
        add(f"Common end year: {common_end}")

        if common_start <= common_end:
            common = actual[
                actual[year_col].between(common_start, common_end)
            ]

            common_counts = common.groupby(country_col)[year_col].nunique()

            add(f"Common-window observations: {len(common)}")
            add(
                f"Countries with observations: "
                f"{common_counts.size}"
            )
            add(
                f"Minimum annual observations per country: "
                f"{common_counts.min()}"
            )
            add(
                f"Maximum annual observations per country: "
                f"{common_counts.max()}"
            )
        else:
            add("No overlapping historical window identified.")

    add()
    add("2020–2024 HISTORICAL ACTUAL COVERAGE")
    add("-" * 78)

    pandemic_window = actual[
        actual[year_col].between(2020, 2024)
    ]

    pandemic_counts = (
        pandemic_window.groupby(country_col)[year_col]
        .nunique()
    )

    add(f"Records: {len(pandemic_window)}")
    add(f"Countries represented: {pandemic_counts.size}")

    if not pandemic_counts.empty:
        add(
            f"Countries with all five years: "
            f"{(pandemic_counts == 5).sum()}"
        )
        add(
            f"Countries with incomplete coverage: "
            f"{(pandemic_counts < 5).sum()}"
        )

    add()
    add("INSPECTION CONCLUSION")
    add("-" * 78)

    if duplicate_count == 0 and not coverage.empty:
        add(
            "Preliminary structural checks passed. "
            "Economic validation and growth-risk design remain pending."
        )
    else:
        add(
            "Input issues detected. Resolve them before building "
            "the growth-risk engine."
        )

REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))
print()
print(f"Report saved to: {REPORT_FILE}")
