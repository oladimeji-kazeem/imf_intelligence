from pathlib import Path
from datetime import datetime, timezone
import json
import requests

BASE_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/gov_10a_main"
)

# Eurostat country codes, not the project's ISO-3 codes.
COUNTRIES = [
    "AT", "BE", "CZ", "DE", "DK", "ES", "FI", "FR",
    "EL", "HU", "IE", "IT", "NL", "NO", "PL", "PT",
    "RO", "SE", "CH"
]

YEARS = ["2020", "2021", "2022", "2023", "2024"]

params = [
    ("lang", "en"),
    ("freq", "A"),
    ("sector", "S13"),
    ("na_item", "D41PAY"),
]

for country in COUNTRIES:
    params.append(("geo", country))

for year in YEARS:
    params.append(("time", year))

print("Requesting Eurostat interest expenditure data...")
response = requests.get(
    BASE_URL,
    params=params,
    timeout=120
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()

if data.get("class") != "dataset":
    raise ValueError(
        "Unexpected API response. Inspect the saved response."
    )

# Preserve the unmodified response.
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_file = output_dir / (
    f"eurostat_interest_probe_{timestamp}.json"
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

values = data.get("value", {})
print("\nOBSERVATIONS")
print("Non-missing observations:", len(values))
print("Expected maximum observations:", len(COUNTRIES) * len(YEARS))

print("\nRAW RESPONSE")
print(output_file)
