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

START_YEAR = 2020
END_YEAR = 2024
EXPECTED = len(COUNTRIES) * (END_YEAR - START_YEAR + 1)


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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_file = RAW_DIR / f"eurostat_b9_{timestamp}.json"
    output_file = PROCESSED_DIR / "europe_eurostat_b9.csv"
    report_file = REPORT_DIR / "europe_b9_extraction_report.txt"

    params = [
        ("freq", "A"),
        ("unit", "MIO_NAC"),
        ("sector", "S13"),
        ("na_item", "B9"),
        ("sinceTimePeriod", str(START_YEAR)),
        ("untilTimePeriod", str(END_YEAR)),
    ]

    params.extend(("geo", country) for country in COUNTRIES)

    print("Requesting Eurostat B9 data...")

    response = requests.get(URL, params=params, timeout=120)

    print("HTTP status:", response.status_code)
    print("Response size:", len(response.content))

    response.raise_for_status()

    data = response.json()

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    df = decode_json_stat(data)

    filters = {
        "freq": "A",
        "unit": "MIO_NAC",
        "sector": "S13",
        "na_item": "B9",
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
        "value": "b9_net_lending"
    })

    df = df[["country_code", "year", "b9_net_lending"]]

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
    missing_values = int(df["b9_net_lending"].isna().sum())

    checks = {
        "Expected observation count": len(df) == EXPECTED,
        "Complete country-year coverage": len(missing_pairs) == 0,
        "No duplicate country-years": duplicate_count == 0,
        "No missing B9 values": missing_values == 0,
    }

    df = df.sort_values(["country_code", "year"])

    df.to_csv(output_file, index=False, float_format="%.6f")

    lines = [
        "EUROSTAT B9 EXTRACTION REPORT",
        "=" * 65,
        "Dataset: gov_10dd_edpt1",
        "Indicator: B9",
        "Definition: Net lending (+) / net borrowing (-)",
        "Sector: S13",
        "Unit: MIO_NAC",
        f"Period: {START_YEAR}-{END_YEAR}",
        f"Expected observations: {EXPECTED}",
        f"Retrieved observations: {len(df)}",
        f"Missing country-years: {len(missing_pairs)}",
        f"Duplicate records: {duplicate_count}",
        f"Missing values: {missing_values}",
        "",
        "MISSING COUNTRY-YEARS:",
        str(missing_pairs if missing_pairs else "None"),
        "",
        "VALIDATION:",
    ]

    for name, passed in checks.items():
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed = all(checks.values())

    lines.extend([
        "",
        "RESULT: " + ("PASS" if passed else "INVESTIGATION REQUIRED"),
        f"Raw file: {raw_file}",
        f"Processed file: {output_file}",
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
