
import json
import csv
from pathlib import Path
from itertools import product

FILE = Path(
    "data/raw/eurostat_sfa_probe_20261114T074434Z.json"
)

OUTPUT = Path(
    "outputs/reports/europe_sfa_missingness.csv"
)

TARGET_INDICATORS = [
    "GD_CH",
    "B9_T3",
    "B9FX9",
    "YA3",
    "YA3O",
    "KX",
    "K61",
    "F_ASS",
    "ORADJ",
    "F71_ADJ_LIAB",
    "F8_ADJ_LIAB",
    "FV_ADJ_LIAB",
    "ORINV",
    "ORD41A_ADJ",
    "ORRNV",
    "ORFCD",
]

TARGET_UNITS = ["MIO_NAC", "PC_GDP"]

with FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

dimension_ids = data["id"]
sizes = data["size"]

# Reverse each dimension's category index.
category_maps = {}

for dimension_id in dimension_ids:
    index = data["dimension"][dimension_id]["category"]["index"]

    category_maps[dimension_id] = {
        position: code
        for code, position in index.items()
    }

countries = sorted(
    data["dimension"]["geo"]["category"]["index"],
    key=lambda code: data["dimension"]["geo"]["category"]["index"][code]
)

years = sorted(
    data["dimension"]["time"]["category"]["index"],
    key=lambda code: data["dimension"]["time"]["category"]["index"][code]
)

# Record country-year pairs for every exact series.
observed = {}

for indicator in TARGET_INDICATORS:
    for unit in TARGET_UNITS:
        observed[(indicator, unit)] = set()

for key in data.get("value", {}):
    position = int(key)
    coordinates = {}

    for dimension_id, size in zip(
        reversed(dimension_ids),
        reversed(sizes)
    ):
        coordinates[dimension_id] = position % size
        position //= size

    observation = {
        dimension_id: category_maps[dimension_id][coordinates[dimension_id]]
        for dimension_id in dimension_ids
    }

    if (
        observation.get("freq") != "A"
        or observation.get("sector") != "S13"
    ):
        continue

    indicator = observation.get("na_item")
    unit = observation.get("unit")

    series = (indicator, unit)

    if series in observed:
        observed[series].add(
            (observation["geo"], observation["time"])
        )

expected = set(product(countries, years))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rows = []

print("EUROSTAT SFA MISSINGNESS AUDIT")
print("=" * 90)

for indicator in TARGET_INDICATORS:
    print(f"\n{indicator}")

    for unit in TARGET_UNITS:
        series = (indicator, unit)
        available = observed[series]
        missing = sorted(expected - available)

        coverage = len(available) / len(expected) * 100

        print(
            f"  {unit:10} "
            f"{len(available):3}/{len(expected)} "
            f"({coverage:5.1f}%) "
            f"Missing: {len(missing)}"
        )

        if missing:
            print(
                "    Missing country-years:",
                ", ".join(f"{country}-{year}" for country, year in missing)
            )

        rows.append({
            "indicator": indicator,
            "unit": unit,
            "expected_observations": len(expected),
            "available_observations": len(available),
            "missing_observations": len(missing),
            "coverage_pct": round(coverage, 2),
            "missing_country_years": ";".join(
                f"{country}-{year}" for country, year in missing
            ),
        })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nREPORT SAVED:")
print(OUTPUT)
