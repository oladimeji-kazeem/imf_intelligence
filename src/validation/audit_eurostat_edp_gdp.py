import json
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/raw/eurostat_debt_probe_20261114T074053Z.json"
)

OUTPUT_FILE = Path(
    "outputs/reports/europe_edp_gdp_coverage.csv"
)

with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

dimension_ids = data["id"]
sizes = data["size"]

# Convert JSON-stat category indexes into position-to-code maps.
category_maps = {}

for dimension_id in dimension_ids:
    index = data["dimension"][dimension_id]["category"]["index"]

    category_maps[dimension_id] = {
        position: code
        for code, position in index.items()
    }

records = []

for key, value in data.get("value", {}).items():

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
        and observation.get("na_item") == "B1GQ"
    ):
        records.append({
            "country": observation["geo"],
            "year": int(observation["time"]),
            "unit": observation["unit"],
            "gdp_value": float(value),
        })

df = pd.DataFrame(
    records,
    columns=["country", "year", "unit", "gdp_value"]
)

if df.empty:
    print("No B1GQ observations found in the existing probe.")
    print("A new Eurostat request will be required.")
    raise SystemExit(0)

if df.duplicated(["country", "year", "unit"]).any():
    raise ValueError("Duplicate GDP observations detected.")

countries = sorted(df["country"].unique())
years = sorted(df["year"].unique())

print("\nEUROSTAT EDP GDP COVERAGE AUDIT")
print("=" * 75)

print("Countries:", len(countries))
print("Years:", years)
print("GDP observations:", len(df))

for unit in sorted(df["unit"].unique()):

    subset = df[df["unit"] == unit]

    expected = len(countries) * len(years)
    observed = subset["gdp_value"].notna().sum()

    print(f"\nUnit: {unit}")
    print("Observed:", observed)
    print("Expected:", expected)
    print("Missing:", expected - observed)

    missing = (
        set((country, year) for country in countries for year in years)
        - set(zip(subset["country"], subset["year"]))
    )

    print("Missing country-years:", sorted(missing))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nGDP coverage report saved:", OUTPUT_FILE)
