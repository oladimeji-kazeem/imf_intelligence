import json
from pathlib import Path
from collections import Counter, defaultdict

FILE = Path(
    "data/raw/eurostat_sfa_probe_20261114T074434Z.json"
)

with FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

dimension_ids = data["id"]
sizes = data["size"]

# Convert dimension positions into category codes.
category_maps = {}

for dimension_id in dimension_ids:
    index = data["dimension"][dimension_id]["category"]["index"]

    category_maps[dimension_id] = {
        position: code
        for code, position in index.items()
    }

counts = Counter()
countries_by_series = defaultdict(set)
years_by_series = defaultdict(set)

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
        observation.get("freq") == "A"
        and observation.get("sector") == "S13"
    ):
        indicator = observation["na_item"]
        unit = observation["unit"]
        country = observation["geo"]
        year = observation["time"]

        series = (indicator, unit)

        counts[series] += 1
        countries_by_series[series].add(country)
        years_by_series[series].add(year)

labels = (
    data["dimension"]["na_item"]["category"]["label"]
)

units = data["dimension"]["unit"]["category"]["index"]

print("EUROSTAT SFA INDICATOR COVERAGE AUDIT")
print("=" * 85)
print(
    f"{'INDICATOR':18} {'UNIT':12} {'OBS':>7} "
    f"{'COUNTRIES':>10} {'YEARS':>7} {'STATUS':>12}"
)

for indicator in sorted(labels):
    for unit in units:
        series = (indicator, unit)
        count = counts[series]
        country_count = len(countries_by_series[series])
        year_count = len(years_by_series[series])

        if count == 85 and country_count == 17 and year_count == 5:
            status = "COMPLETE"
        elif count == 0:
            status = "NO DATA"
        else:
            status = "PARTIAL"

        if count > 0:
            print(
                f"{indicator:18} {unit:12} {count:7} "
                f"{country_count:10} {year_count:7} {status:>12}"
            )

print("\nKEY SFA COMPONENTS")
print("=" * 85)

key_indicators = [
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

for indicator in key_indicators:
    print(f"\n{indicator}: {labels.get(indicator, 'Unknown')}")

    for unit in units:
        series = (indicator, unit)
        count = counts[series]

        print(
            f"  {unit:10} {count:3}/85 "
            f"countries={len(countries_by_series[series])}/17 "
            f"years={len(years_by_series[series])}/5"
        )

print("\nRAW OBSERVATION COUNT:", len(data.get("value", {})))
print("RESULT: SERIES-LEVEL COVERAGE AUDIT COMPLETED.")
