import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

FILES = {
    "interest": RAW_DIR / "eurostat_interest_probe_20261114T073451Z.json",
    "debt": RAW_DIR / "eurostat_debt_probe_20261114T082331Z.json",
    "gdp": RAW_DIR / "eurostat_edp_gdp_20261114T082036Z.json",
}

OUTPUT_FILE = (
    PROCESSED_DIR / "europe_effective_interest_rate_panel.csv"
)

REPORT_FILE = (
    REPORT_DIR / "europe_effective_interest_rate_report.txt"
)

COUNTRIES = {
    "AT", "BE", "CZ", "DE", "DK", "EL", "ES", "FI", "FR",
    "HU", "IE", "IT", "NL", "PL", "PT", "RO", "SE"
}

YEARS = list(range(2020, 2025))


# ============================================================
# JSON-STAT DECODER
# ============================================================

def decode_json_stat(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required source file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    dimensions = data["id"]
    sizes = data["size"]
    values = data.get("value", {})

    categories = {}

    for dimension in dimensions:

        index = (
            data["dimension"][dimension]
            ["category"]["index"]
        )

        categories[dimension] = sorted(
            index,
            key=index.get
        )

    records = []

    for position, value in values.items():

        remainder = int(position)
        coordinates = [0] * len(sizes)

        for i in range(len(sizes) - 1, -1, -1):
            coordinates[i] = remainder % sizes[i]
            remainder //= sizes[i]

        record = {
            dimensions[i]: categories[dimensions[i]][coordinates[i]]
            for i in range(len(dimensions))
        }

        record["value"] = value
        records.append(record)

    return pd.DataFrame(records), data


# ============================================================
# SERIES EXTRACTION
# ============================================================

def extract_series(file_path, filters, series_name):

    df, metadata = decode_json_stat(file_path)

    for dimension, expected_value in filters.items():

        if dimension not in df.columns:
            raise ValueError(
                f"{series_name}: missing dimension {dimension}"
            )

        df = df[
            df[dimension].astype(str) == str(expected_value)
        ]

    df = df[
        df["geo"].astype(str).isin(COUNTRIES)
    ].copy()

    df["year"] = pd.to_numeric(
        df["time"], errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    )

    df = df[df["year"].isin(YEARS if series_name != "debt"
                            else range(2019, 2025))]

    df = df.rename(
        columns={
            "geo": "country_code",
            "value": series_name
        }
    )

    df = df[
        ["country_code", "year", series_name]
    ]

    if df.duplicated(
        ["country_code", "year"]
    ).any():
        raise ValueError(
            f"{series_name}: duplicate country-year observations"
        )

    print(
        f"{series_name}: {len(df)} observations | "
        f"Updated: {metadata.get('updated', 'unknown')}"
    )

    return df


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("=" * 70)
    print("EUROPEAN EFFECTIVE INTEREST RATE PANEL")
    print("=" * 70)

    interest = extract_series(
        FILES["interest"],
        {
            "freq": "A",
            "unit": "MIO_NAC",
            "sector": "S13",
            "na_item": "D41PAY",
        },
        "interest_expenditure"
    )

    debt = extract_series(
        FILES["debt"],
        {
            "freq": "A",
            "unit": "MIO_NAC",
            "sector": "S13",
            "na_item": "GD",
        },
        "debt"
    )

    gdp = extract_series(
        FILES["gdp"],
        {
            "freq": "A",
            "unit": "MIO_NAC",
            "sector": "S1",
            "na_item": "B1GQ",
        },
        "nominal_gdp"
    )

    # --------------------------------------------------------
    # BUILD OPENING AND CLOSING DEBT
    # --------------------------------------------------------

    debt_opening = debt.copy()

    debt_opening["year"] = debt_opening["year"] + 1

    debt_opening = debt_opening.rename(
        columns={"debt": "opening_debt"}
    )

    debt_closing = debt.rename(
        columns={"debt": "closing_debt"}
    )

    # --------------------------------------------------------
    # MERGE SERIES
    # --------------------------------------------------------

    panel = interest.merge(
        debt_opening,
        on=["country_code", "year"],
        how="outer",
        validate="one_to_one"
    )

    panel = panel.merge(
        debt_closing,
        on=["country_code", "year"],
        how="outer",
        validate="one_to_one"
    )

    panel = panel.merge(
        gdp,
        on=["country_code", "year"],
        how="outer",
        validate="one_to_one"
    )

    panel = panel[
        panel["year"].isin(YEARS)
        & panel["country_code"].isin(COUNTRIES)
    ].copy()

    panel["country"] = panel["country_code"].map({
        "AT": "Austria",
        "BE": "Belgium",
        "CZ": "Czechia",
        "DE": "Germany",
        "DK": "Denmark",
        "EL": "Greece",
        "ES": "Spain",
        "FI": "Finland",
        "FR": "France",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SE": "Sweden",
    })

    # --------------------------------------------------------
    # CALCULATE EFFECTIVE INTEREST RATE
    # --------------------------------------------------------

    panel["average_debt"] = (
        panel["opening_debt"] + panel["closing_debt"]
    ) / 2

    panel["effective_interest_rate_pct"] = (
        panel["interest_expenditure"]
        / panel["average_debt"]
    ) * 100

    panel["interest_to_gdp_pct"] = (
        panel["interest_expenditure"]
        / panel["nominal_gdp"]
    ) * 100

    panel["debt_to_gdp_pct"] = (
        panel["closing_debt"]
        / panel["nominal_gdp"]
    ) * 100

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    expected_rows = len(COUNTRIES) * len(YEARS)

    required_columns = [
        "interest_expenditure",
        "opening_debt",
        "closing_debt",
        "average_debt",
        "nominal_gdp",
        "effective_interest_rate_pct",
        "interest_to_gdp_pct",
        "debt_to_gdp_pct",
    ]

    missing_by_column = panel[
        required_columns
    ].isna().sum()

    duplicate_count = panel.duplicated(
        ["country_code", "year"]
    ).sum()

    invalid_debt = (
        (panel["opening_debt"] <= 0)
        | (panel["closing_debt"] <= 0)
    ).sum()

    invalid_gdp = (
        panel["nominal_gdp"] <= 0
    ).sum()

    invalid_interest_rate = (
        panel["effective_interest_rate_pct"] < 0
    ).sum()

    checks = {
        "Expected panel size": len(panel) == expected_rows,
        "No duplicate country-years": duplicate_count == 0,
        "No missing required inputs": missing_by_column.sum() == 0,
        "Positive debt stocks": invalid_debt == 0,
        "Positive nominal GDP": invalid_gdp == 0,
        "Non-negative effective rates": invalid_interest_rate == 0,
    }

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    panel = panel.sort_values(
        ["country_code", "year"]
    )

    columns = [
        "country_code",
        "country",
        "year",
        "interest_expenditure",
        "opening_debt",
        "closing_debt",
        "average_debt",
        "nominal_gdp",
        "effective_interest_rate_pct",
        "interest_to_gdp_pct",
        "debt_to_gdp_pct",
    ]

    PROCESSED_DIR.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    panel[columns].to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    lines = [
        "EUROPEAN EFFECTIVE INTEREST RATE REPORT",
        "=" * 70,
        f"Countries expected: {len(COUNTRIES)}",
        f"Years expected: {len(YEARS)}",
        f"Expected observations: {expected_rows}",
        f"Actual observations: {len(panel)}",
        "",
        "MISSING VALUES BY COLUMN",
    ]

    for column, count in missing_by_column.items():
        lines.append(f"{column}: {count}")

    lines.extend([
        "",
        "VALIDATION CHECKS",
    ])

    for name, passed in checks.items():
        lines.append(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    overall_pass = all(checks.values())

    lines.extend([
        "",
        "RESULT: "
        + ("PASS" if overall_pass else "INVESTIGATION REQUIRED"),
        "",
        f"Output: {OUTPUT_FILE}",
    ])

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nVALIDATION")
    print("=" * 70)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\nREPORT:", REPORT_FILE)
    print("OUTPUT:", OUTPUT_FILE)

    print("\nEFFECTIVE INTEREST RATE SUMMARY")

    print(
        panel.groupby("year")[
            "effective_interest_rate_pct"
        ].agg(["count", "mean", "median", "min", "max"])
    )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
