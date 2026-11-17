
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/europe_weo_classified.csv"

OUTPUT_PANEL = (
    ROOT / "data/processed/"
    "europe_labour_risk_input_profile.csv"
)

OUTPUT_COUNTRY = (
    ROOT / "outputs/reports/"
    "europe_labour_risk_country_coverage.csv"
)

OUTPUT_MISSING = (
    ROOT / "outputs/reports/"
    "europe_labour_risk_missingness.csv"
)

OUTPUT_COMMON = (
    ROOT / "outputs/reports/"
    "europe_labour_risk_common_period_coverage.csv"
)

OUTPUT_REPORT = (
    ROOT / "outputs/reports/"
    "europe_labour_risk_input_inspection.txt"
)

INDICATOR = "LUR"
EXPECTED_COUNTRIES = 20
COMMON_START = 2020
COMMON_END = 2024


def main():

    print("=" * 76)
    print("EUROPEAN LABOUR RISK INPUT INSPECTION")
    print("=" * 76)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    source = pd.read_csv(INPUT)

    required = {
        "iso",
        "year",
        "weo_code",
        "value",
        "observation_status",
    }

    missing_columns = required - set(source.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    source["year"] = pd.to_numeric(
        source["year"],
        errors="coerce"
    )

    source["value"] = pd.to_numeric(
        source["value"],
        errors="coerce"
    )

    # Retain historical actual observations only.
    df = source[
        (source["weo_code"] == INDICATOR)
        &
        (source["observation_status"] == "historical_actual")
    ].copy()

    if df.empty:
        raise ValueError(
            "No historical LUR observations found."
        )

    df["year"] = df["year"].astype(int)

    df = df.sort_values(
        ["iso", "year"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # STRUCTURAL VALIDATION
    # --------------------------------------------------------

    if df.duplicated(["iso", "year"]).any():
        duplicates = df[
            df.duplicated(["iso", "year"], keep=False)
        ]

        raise ValueError(
            "Duplicate country-year records detected:\n"
            f"{duplicates.head(10)}"
        )

    countries = sorted(df["iso"].unique())

    if len(countries) != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries; "
            f"found {len(countries)}."
        )

    # --------------------------------------------------------
    # COUNTRY COVERAGE
    # --------------------------------------------------------

    country_rows = []

    for iso, group in df.groupby("iso"):

        numeric = group.dropna(subset=["value"])

        country_rows.append({
            "iso": iso,
            "records": len(group),
            "numeric_observations": len(numeric),
            "missing_numeric_observations": (
                group["value"].isna().sum()
            ),
            "first_year": group["year"].min(),
            "last_year": group["year"].max(),
            "first_numeric_year": (
                numeric["year"].min()
                if not numeric.empty else None
            ),
            "last_numeric_year": (
                numeric["year"].max()
                if not numeric.empty else None
            ),
        })

    country_coverage = pd.DataFrame(country_rows)

    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    missingness_rows = []

    for year, group in df.groupby("year"):

        missingness_rows.append({
            "year": year,
            "countries": group["iso"].nunique(),
            "records": len(group),
            "numeric_observations": group["value"].notna().sum(),
            "missing_observations": group["value"].isna().sum(),
        })

    missingness = pd.DataFrame(missingness_rows)

    # --------------------------------------------------------
    # COMMON-PERIOD COVERAGE
    # --------------------------------------------------------

    common = df[
        df["year"].between(COMMON_START, COMMON_END)
    ].copy()

    common_rows = []

    for year, group in common.groupby("year"):

        common_rows.append({
            "year": year,
            "records": len(group),
            "countries": group["iso"].nunique(),
            "numeric_observations": group["value"].notna().sum(),
            "missing_observations": group["value"].isna().sum(),
        })

    common_coverage = pd.DataFrame(common_rows)

    expected_common_records = (
        EXPECTED_COUNTRIES
        * (COMMON_END - COMMON_START + 1)
    )

    common_numeric = common["value"].notna().sum()

    # --------------------------------------------------------
    # VALUE AND EXTREME INSPECTION
    # --------------------------------------------------------

    numeric = df.dropna(subset=["value"])

    value_summary = {
        "minimum": numeric["value"].min(),
        "maximum": numeric["value"].max(),
        "mean": numeric["value"].mean(),
        "median": numeric["value"].median(),
        "negative_values": (numeric["value"] < 0).sum(),
        "zero_values": (numeric["value"] == 0).sum(),
        "values_above_50": (numeric["value"] > 50).sum(),
        "values_above_100": (numeric["value"] > 100).sum(),
    }

    extremes = numeric[
        (numeric["value"] < 0)
        |
        (numeric["value"] > 50)
    ].sort_values("value")

    # --------------------------------------------------------
    # OUTPUT PANEL
    # --------------------------------------------------------

    panel_columns = [
        "iso",
        "year",
        "weo_code",
        "value",
        "observation_status",
    ]

    panel = df[panel_columns].copy()

    OUTPUT_PANEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    panel.to_csv(
        OUTPUT_PANEL,
        index=False
    )

    country_coverage.to_csv(
        OUTPUT_COUNTRY,
        index=False
    )

    missingness.to_csv(
        OUTPUT_MISSING,
        index=False
    )

    common_coverage.to_csv(
        OUTPUT_COMMON,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    lines = []

    lines.append("=" * 76)
    lines.append("EUROPEAN LABOUR RISK INPUT INSPECTION")
    lines.append("=" * 76)

    lines.append("")
    lines.append("INPUT")
    lines.append("-" * 76)
    lines.append(str(INPUT))
    lines.append(f"Source rows: {len(source):,}")

    lines.append("")
    lines.append("INDICATOR")
    lines.append("-" * 76)
    lines.append(f"WEO indicator: {INDICATOR}")
    lines.append("Description: Unemployment rate, percent of labour force")
    lines.append(f"Historical records: {len(df):,}")
    lines.append(f"Countries: {df['iso'].nunique()}")
    lines.append(
        f"Year range: {df['year'].min()}-{df['year'].max()}"
    )
    lines.append(
        f"Numeric observations: {df['value'].notna().sum():,}"
    )
    lines.append(
        f"Missing numeric observations: {df['value'].isna().sum():,}"
    )

    lines.append("")
    lines.append("COUNTRY COVERAGE")
    lines.append("-" * 76)
    lines.append(country_coverage.to_string(index=False))

    lines.append("")
    lines.append("COMMON PERIOD COVERAGE")
    lines.append("-" * 76)
    lines.append(
        f"Period: {COMMON_START}-{COMMON_END}"
    )
    lines.append(
        f"Expected balanced records: {expected_common_records}"
    )
    lines.append(
        f"Actual records: {len(common)}"
    )
    lines.append(
        f"Numeric observations: {common_numeric}"
    )
    lines.append(
        f"Missing observations: {common['value'].isna().sum()}"
    )
    lines.append(common_coverage.to_string(index=False))

    lines.append("")
    lines.append("NUMERIC VALUE INSPECTION")
    lines.append("-" * 76)

    for key, value in value_summary.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("EXTREME OBSERVATIONS")
    lines.append("-" * 76)

    if extremes.empty:
        lines.append("No values below zero or above 50.")
    else:
        lines.append(
            extremes[
                ["iso", "year", "value"]
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("METHODOLOGICAL NOTES")
    lines.append("-" * 76)
    lines.append(
        "Unemployment rate does not measure total labour-market slack."
    )
    lines.append(
        "It excludes underemployment and people outside the labour force."
    )
    lines.append(
        "It does not distinguish youth or long-term unemployment."
    )
    lines.append(
        "Cross-country comparisons can be affected by survey definitions."
    )
    lines.append(
        "Historical actual classification follows the project's "
        "WEO observation-status convention."
    )
    lines.append(
        "Missing values are retained and are not replaced with zero."
    )

    lines.append("")
    lines.append("VALIDATION")
    lines.append("-" * 76)
    lines.append("PASS: Required source columns present.")
    lines.append("PASS: LUR indicator found.")
    lines.append(f"PASS: {len(countries)} countries identified.")
    lines.append("PASS: No duplicate country-year records.")

    if len(common) == expected_common_records:
        lines.append(
            "PASS: Common-period panel has expected record count."
        )
    else:
        lines.append(
            "WARNING: Common-period panel is not balanced."
        )

    if common_numeric == expected_common_records:
        lines.append(
            "PASS: Common-period observations are all numeric."
        )
    else:
        lines.append(
            "WARNING: Common-period numeric coverage is incomplete."
        )

    lines.append("")
    lines.append("OUTPUTS")
    lines.append("-" * 76)
    lines.append(str(OUTPUT_PANEL))
    lines.append(str(OUTPUT_COUNTRY))
    lines.append(str(OUTPUT_MISSING))
    lines.append(str(OUTPUT_COMMON))

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("")
    print("OUTPUTS CREATED")
    print("-" * 76)
    print(OUTPUT_PANEL)
    print(OUTPUT_COUNTRY)
    print(OUTPUT_MISSING)
    print(OUTPUT_COMMON)
    print(OUTPUT_REPORT)

    print("")
    print("INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
