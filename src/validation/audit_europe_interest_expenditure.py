import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    ROOT / "data" / "raw"
    / "eurostat_interest_probe_20261114T073451Z.json"
)

REPORT_FILE = (
    ROOT / "outputs" / "reports"
    / "europe_interest_expenditure_audit.txt"
)

COUNTRIES = {
    "AT", "BE", "CZ", "DE", "DK", "EL", "ES", "FI", "FR",
    "HU", "IE", "IT", "NL", "PL", "PT", "RO", "SE"
}

YEARS = {str(y) for y in range(2020, 2025)}

FILTERS = {
    "freq": "A",
    "unit": "MIO_NAC",
    "sector": "S13",
    "na_item": "D41PAY",
}

EXPECTED = len(COUNTRIES) * len(YEARS)


def decode_json_stat(data):
    dimensions = data["id"]
    sizes = data["size"]
    values = data.get("value", {})

    categories = {}

    for dim in dimensions:
        index = (
            data["dimension"][dim]["category"]["index"]
        )
        categories[dim] = sorted(index, key=index.get)

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
    print("=" * 70)
    print("EUROSTAT INTEREST EXPENDITURE AUDIT")
    print("=" * 70)

    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    with open(RAW_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    df = decode_json_stat(raw)

    print("\nDATASET:", raw.get("label"))
    print("UPDATED:", raw.get("updated"))

    for dim, value in FILTERS.items():
        df = df[df[dim].astype(str) == value]

    df = df[
        df["geo"].astype(str).isin(COUNTRIES)
        & df["time"].astype(str).isin(YEARS)
    ].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    duplicate_count = int(
        df.duplicated(["geo", "time"]).sum()
    )

    missing_values = int(df["value"].isna().sum())

    negative_values = int((df["value"] < 0).sum())

    actual_pairs = set(
        zip(df["geo"].astype(str), df["time"].astype(str))
    )

    expected_pairs = {
        (country, year)
        for country in COUNTRIES
        for year in YEARS
    }

    missing_pairs = sorted(expected_pairs - actual_pairs)

    unexpected_pairs = sorted(actual_pairs - expected_pairs)

    print("\nEXACT SERIES")
    for dim, value in FILTERS.items():
        print(f"{dim}: {value}")

    print("\nOBSERVATIONS")
    print("Expected:", EXPECTED)
    print("Retrieved:", len(df))

    print("\nCOUNTRY COVERAGE")
    for country in sorted(COUNTRIES):
        count = int((df["geo"] == country).sum())
        print(f"{country}: {count}")

    print("\nYEAR COVERAGE")
    for year in sorted(YEARS):
        count = int((df["time"].astype(str) == year).sum())
        print(f"{year}: {count}")

    print("\nMISSING COUNTRY-YEARS")
    print(missing_pairs if missing_pairs else "None")

    checks = {
        "Observation count": len(df) == EXPECTED,
        "Complete country-year coverage": len(missing_pairs) == 0,
        "No unexpected country-years": len(unexpected_pairs) == 0,
        "No duplicate country-years": duplicate_count == 0,
        "No missing numeric values": missing_values == 0,
        "No negative interest expenditure": negative_values == 0,
    }

    print("\nVALIDATION")
    print("-" * 70)

    print("Duplicates:", duplicate_count)
    print("Missing values:", missing_values)
    print("Negative values:", negative_values)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed = all(checks.values())

    print("\nRESULT:", "PASS" if passed else "INVESTIGATION REQUIRED")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "EUROSTAT INTEREST EXPENDITURE AUDIT",
        "=" * 70,
        f"Expected observations: {EXPECTED}",
        f"Retrieved observations: {len(df)}",
        f"Duplicate country-years: {duplicate_count}",
        f"Missing values: {missing_values}",
        f"Negative values: {negative_values}",
        "",
    ]

    lines.extend(
        f"{name}: {'PASS' if result else 'FAIL'}"
        for name, result in checks.items()
    )

    lines.append(
        "\nFINAL RESULT: "
        + ("PASS" if passed else "INVESTIGATION REQUIRED")
    )

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("Report saved:", REPORT_FILE)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
