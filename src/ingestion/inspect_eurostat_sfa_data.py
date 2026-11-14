from pathlib import Path
from datetime import datetime, timezone
import json
import requests

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/gov_10dd_edpt3"
)

COUNTRIES = [
    "AT", "BE", "CZ", "DE", "DK", "ES", "FI", "FR",
    "EL", "HU", "IE", "IT", "NL", "PL", "PT", "RO", "SE"
]

YEARS = ["2020", "2021", "2022", "2023", "2024"]

# Deliberately do not filter unit or na_item.
# We need to discover all available indicators and units.
params = [
    ("lang", "en"),
    ("freq", "A"),
    ("sector", "S13"),
]

for country in COUNTRIES:
    params.append(("geo", country))

for year in YEARS:
    params.append(("time", year))

print("Requesting Eurostat SFA data...")
print("Dataset: gov_10dd_edpt3")

response = requests.get(
    BASE_URL,
    params=params,
    timeout=180
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()

if data.get("class") != "dataset":
    raise ValueError("Unexpected Eurostat API response.")

# Preserve the original response.
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

output_file = output_dir / (
    f"eurostat_sfa_probe_{timestamp}.json"
)

output_file.write_text(
    json.dumps(data, indent=2),
    encoding="utf-8"
)

print("\nDATASET")
print("Label:", data.get("label"))
print("Last updated:", data.get("updated"))

print("\nDIMENSIONS")

for dimension_id in data.get("id", []):
    dimension = data["dimension"][dimension_id]
    categories = dimension.get("category", {})
    labels = categories.get("label", {})
    index = categories.get("index", {})

    print(f"\n{dimension_id}: {dimension.get('label')}")

    for code, position in sorted(
        index.items(), key=lambda item: item[1]
    ):
        print(f"  {code}: {labels.get(code, '')}")

print("\nOBSERVATIONS")
print("Non-missing observations:", len(data.get("value", {})))

print("\nREQUESTED COVERAGE")
print("Countries:", len(COUNTRIES))
print("Years:", len(YEARS))
print("Maximum country-year combinations:", len(COUNTRIES) * len(YEARS))

print("\nRAW RESPONSE")
print(output_file)
