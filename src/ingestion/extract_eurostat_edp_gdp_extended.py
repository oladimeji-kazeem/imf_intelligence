import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/gov_10dd_edpt1"
)

COUNTRIES = [
    "BE", "CZ", "DK", "DE", "IE", "EL", "ES", "FR", "IT",
    "HU", "NL", "AT", "PL", "PT", "RO", "FI", "SE"
]

START_YEAR = 2019
END_YEAR = 2024
EXPECTED = len(COUNTRIES) * (END_YEAR - START_YEAR + 1)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

RAW_FILE = RAW_DIR / f"eurostat_edp_gdp_extended_{timestamp}.json"
OUTPUT_FILE = PROCESSED_DIR / "europe_eurostat_edp_gdp_2019_2024.csv"
REPORT_FILE = REPORT_DIR / "europe_edp_gdp_extended_report.txt"


def decode_json_stat(data):

    dimensions = data["id"]
    sizes = data["size"]
    values = data.get("value", {})

    categories = {}

    for dimension in dimensions:
        index = data["dimension"][dimension]["category"]["index"]
        categories[dimension] = sorted(index, key=index.get)

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

    return pd.DataFrame(records)


def main():

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    params = [
        ("freq", "A"),
        ("unit", "MIO_NAC"),
        ("sector", "S1"),
        ("na_item", "B1GQ"),
        ("sinceTimePeriod", str(START_YEAR)),
        ("untilTimePeriod", str(END_YEAR)),
    ]

    params.extend(("geo", country) for country in COUNTRIES)

    print("Requesting Eurostat GDP data...")
    response = requests.get(URL, params=params, timeout=120)

    print("HTTP status:", response.status_code)
    print("Request URL:", response.url)
    print("Response size:", len(response.content))

    response.raise_for_status()

    data = response.json()

    with open(RAW_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    df = decode_json_stat(data)

    filters = {
        "freq": "A",
        "unit": "MIO_NAC",
        "sector": "S1",
        "na_item": "B1GQ",
    }

    for dimension, expected_value in filters.items():
        df = df[df[dimension].astype(str) == expected_value]

    df = df[
        df["geo"].astype(str).isin(COUNTRIES)
        & df["time"].astype(str).isin(
            [str(year) for year in range(START_YEAR, END_YEAR + 1)]
        )
    ].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = pd.to_numeric(df["time"], errors="coerce")

    df = df.rename(columns={
        "geo": "country_code",
        "value": "nominal_gdp"
    })

    df = df[["country_code", "year", "nominal_gdp"]]

    duplicate_count = df.duplicated(
        ["country_code", "year"]
    ).sum()

    expected_pairs = {
        (country, year)
        for country in COUNTRIES
        for year in range(START_YEAR, END_YEAR + 1)
    }

    actual_pairs = set(
        zip(df["country_code"], df["year"].astype(int))
    )

    missing_pairs = sorted(expected_pairs - actual_pairs)

    missing_values = int(df["nominal_gdp"].isna().sum())
    non_positive = int((df["nominal_gdp"] <= 0).sum())

    checks = {
        "Expected observation count": len(df) == EXPECTED,
        "Complete country-year coverage": len(missing_pairs) == 0,
        "No duplicate country-years": duplicate_count == 0,
        "No missing GDP values": missing_values == 0,
        "Positive GDP values": non_positive == 0,
    }

    df = df.sort_values(["country_code", "year"])

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f"
    )

    lines = [
        "EUROSTAT EXTENDED GDP EXTRACTION REPORT",
        "=" * 70,
        f"Dataset: gov_10dd_edpt1",
        f"Indicator: B1GQ",
        f"Sector: S1",
        f"Unit: MIO_NAC",
        f"Period: {START_YEAR}-{END_YEAR}",
        f"Expected observations: {EXPECTED}",
        f"Retrieved observations: {len(df)}",
        f"Missing country-years: {len(missing_pairs)}",
        f"Duplicate records: {duplicate_count}",
        f"Missing values: {missing_values}",
        f"Non-positive values: {non_positive}",
        "",
        "MISSING COUNTRY-YEARS:",
        str(missing_pairs if missing_pairs else "None"),
        "",
        "VALIDATION:",
    ]

    for name, passed in checks.items():
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'}")

    overall_pass = all(checks.values())

    lines.extend([
        "",
        "RESULT: " + ("PASS" if overall_pass else "INVESTIGATION REQUIRED"),
        "",
        f"Raw file: {RAW_FILE}",
        f"Processed file: {OUTPUT_FILE}",
    ])

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))

    print("\nReport saved:", REPORT_FILE)

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
