import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eurostat_debt_probe_20261114T082331Z.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "europe_debt_series_audit.txt"
)

EXPECTED_COUNTRIES = {
    "AT", "BE", "CZ", "DE", "DK", "EL", "ES", "FI", "FR",
    "HU", "IE", "IT", "NL", "PL", "PT", "RO", "SE"
}

EXPECTED_YEARS = {
    "2019", "2020", "2021", "2022", "2023", "2024"
}

REQUIRED_DIMENSIONS = {
    "freq": "A",
    "unit": "MIO_NAC",
    "sector": "S13",
    "na_item": "GD",
}

EXPECTED_OBSERVATIONS = (
    len(EXPECTED_COUNTRIES) * len(EXPECTED_YEARS)
)


# ============================================================
# HELPERS
# ============================================================

def get_dimension_categories(dataset, dimension):
    """Return ordered category codes for a JSON-stat dimension."""

    categories = (
        dataset.get("dimension", {})
        .get(dimension, {})
        .get("category", {})
        .get("index", {})
    )

    if isinstance(categories, dict):
        return sorted(categories, key=categories.get)

    if isinstance(categories, list):
        return categories

    raise ValueError(
        f"Unable to read categories for dimension: {dimension}"
    )


def decode_json_stat(dataset):
    """Convert Eurostat JSON-stat observations into a DataFrame."""

    dimensions = dataset.get("id", [])
    sizes = dataset.get("size", [])
    values = dataset.get("value", {})

    if not dimensions or not sizes:
        raise ValueError("JSON-stat dimension metadata is missing.")

    if len(dimensions) != len(sizes):
        raise ValueError("Dimension IDs and sizes do not match.")

    categories = {
        dimension: get_dimension_categories(dataset, dimension)
        for dimension in dimensions
    }

    records = []

    for position, value in values.items():
        index = int(position)

        coordinates = [0] * len(sizes)

        for i in range(len(sizes) - 1, -1, -1):
            coordinates[i] = index % sizes[i]
            index //= sizes[i]

        record = {
            dimensions[i]: categories[dimensions[i]][coordinates[i]]
            for i in range(len(dimensions))
        }

        record["value"] = value
        records.append(record)

    return pd.DataFrame(records), categories


def build_missing_combinations(df):
    """Identify every missing country-year combination."""

    expected = {
        (country, year)
        for country in EXPECTED_COUNTRIES
        for year in EXPECTED_YEARS
    }

    actual = set(
        zip(
            df["geo"].astype(str),
            df["time"].astype(str)
        )
    )

    return sorted(expected - actual)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("EUROSTAT GOVERNMENT DEBT SERIES AUDIT")
    print("=" * 70)

    if not RAW_FILE.exists():
        print(f"ERROR: Raw file not found: {RAW_FILE}")
        sys.exit(1)

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    df, categories = decode_json_stat(dataset)

    # --------------------------------------------------------
    # DATASET METADATA
    # --------------------------------------------------------

    print("\nDATASET METADATA")
    print("-" * 70)

    print("Dataset:", dataset.get("label", "Not supplied"))
    print("Dimensions:", dataset.get("id"))
    print("Dimension sizes:", dataset.get("size"))
    print("Last updated:", dataset.get("updated", "Not supplied"))

    # --------------------------------------------------------
    # FILTER EXACT SERIES
    # --------------------------------------------------------

    print("\nEXACT SERIES")
    print("-" * 70)

    for dimension, required_value in REQUIRED_DIMENSIONS.items():
        print(f"{dimension}: {required_value}")

    required_columns = list(REQUIRED_DIMENSIONS.keys()) + ["geo", "time", "value"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required dimensions missing from decoded data: {missing_columns}"
        )

    filtered_df = df.copy()

    for dimension, required_value in REQUIRED_DIMENSIONS.items():
        filtered_df = filtered_df[
            filtered_df[dimension].astype(str) == required_value
        ]

    # --------------------------------------------------------
    # OBSERVATIONS BY UNIT
    # --------------------------------------------------------

    print("\nOBSERVATIONS BY UNIT")
    print("-" * 70)

    unit_counts = (
        df[df["freq"].astype(str) == "A"]
        .loc[
            (df["sector"].astype(str) == "S13")
            & (df["na_item"].astype(str) == "GD")
        ]
        .groupby("unit")["value"]
        .count()
    )

    for unit in ["MIO_EUR", "MIO_NAC", "PC_GDP"]:
        count = int(unit_counts.get(unit, 0))

        if unit == "MIO_NAC":
            expected = EXPECTED_OBSERVATIONS
            print(f"{unit:<12} {count}/{expected}  REQUIRED")
        else:
            print(f"{unit:<12} {count}  NOT REQUESTED")

    print(
        "\nTOTAL REQUIRED-SERIES OBSERVATIONS:",
        len(filtered_df)
    )

    # --------------------------------------------------------
    # COUNTRY COVERAGE
    # --------------------------------------------------------

    print("\nCOUNTRY COVERAGE")
    print("-" * 70)

    country_counts = (
        filtered_df.groupby("geo")["value"].count().to_dict()
    )

    for country in sorted(EXPECTED_COUNTRIES):
        print(f"{country}: {country_counts.get(country, 0)}")

    unexpected_countries = (
        set(filtered_df["geo"].astype(str).unique())
        - EXPECTED_COUNTRIES
    )

    missing_countries = (
        EXPECTED_COUNTRIES
        - set(filtered_df["geo"].astype(str).unique())
    )

    # --------------------------------------------------------
    # YEAR COVERAGE
    # --------------------------------------------------------

    print("\nYEAR COVERAGE")
    print("-" * 70)

    year_counts = (
        filtered_df.groupby("time")["value"].count().to_dict()
    )

    for year in sorted(EXPECTED_YEARS):
        print(f"{year}: {year_counts.get(year, 0)}")

    missing_years = (
        EXPECTED_YEARS
        - set(filtered_df["time"].astype(str).unique())
    )

    unexpected_years = (
        set(filtered_df["time"].astype(str).unique())
        - EXPECTED_YEARS
    )

    # --------------------------------------------------------
    # MISSING COUNTRY-YEAR COMBINATIONS
    # --------------------------------------------------------

    missing_combinations = build_missing_combinations(filtered_df)

    print("\nMISSING COUNTRY-YEAR COMBINATIONS")
    print("-" * 70)

    if missing_combinations:
        for country, year in missing_combinations:
            print(f"{country}: {year}")
    else:
        print("None")

    # --------------------------------------------------------
    # DUPLICATE AND VALUE CHECKS
    # --------------------------------------------------------

    duplicate_count = int(
        filtered_df.duplicated(
            subset=["geo", "time"],
            keep=False
        ).sum()
    )

    missing_values = int(filtered_df["value"].isna().sum())

    non_numeric_values = int(
        pd.to_numeric(
            filtered_df["value"],
            errors="coerce"
        ).isna().sum()
    )

    numeric_values = pd.to_numeric(
        filtered_df["value"],
        errors="coerce"
    )

    non_positive_values = int((numeric_values <= 0).sum())

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    checks = {
        "Correct observation count":
            len(filtered_df) == EXPECTED_OBSERVATIONS,

        "Complete country coverage":
            not missing_countries and not unexpected_countries,

        "Complete year coverage":
            not missing_years and not unexpected_years,

        "Complete country-year panel":
            len(missing_combinations) == 0,

        "No duplicate country-year records":
            duplicate_count == 0,

        "No missing values":
            missing_values == 0,

        "All values numeric":
            non_numeric_values == 0,

        "All debt values positive":
            non_positive_values == 0,
    }

    print("\nVALIDATION")
    print("=" * 70)

    print("Expected observations:", EXPECTED_OBSERVATIONS)
    print("Retrieved observations:", len(filtered_df))
    print("Duplicate country-year records:", duplicate_count)
    print("Missing values:", missing_values)
    print("Non-numeric values:", non_numeric_values)
    print("Non-positive debt values:", non_positive_values)

    print("\nINDIVIDUAL CHECKS")

    for check, passed in checks.items():
        print(f"{check}: {'PASS' if passed else 'FAIL'}")

    overall_pass = all(checks.values())

    print("\nRESULT")
    print("=" * 70)

    if overall_pass:
        print("DEBT SERIES AUDIT PASSED.")
    else:
        print("COVERAGE EXCEPTIONS REQUIRE INVESTIGATION.")

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_lines = [
        "EUROSTAT GOVERNMENT DEBT SERIES AUDIT",
        "=" * 70,
        f"Raw file: {RAW_FILE}",
        f"Expected observations: {EXPECTED_OBSERVATIONS}",
        f"Retrieved observations: {len(filtered_df)}",
        f"Duplicate country-year records: {duplicate_count}",
        f"Missing values: {missing_values}",
        f"Non-numeric values: {non_numeric_values}",
        f"Non-positive debt values: {non_positive_values}",
        "",
        "VALIDATION CHECKS",
    ]

    for check, passed in checks.items():
        report_lines.append(
            f"{check}: {'PASS' if passed else 'FAIL'}"
        )

    report_lines.extend([
        "",
        "FINAL RESULT: "
        + ("PASS" if overall_pass else "INVESTIGATION REQUIRED")
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nReport saved:", REPORT_FILE)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()