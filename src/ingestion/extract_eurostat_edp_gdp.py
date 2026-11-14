import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET = "gov_10dd_edpt1"

API_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data/"
    f"{DATASET}"
)

COUNTRIES = [
    "BE", "CZ", "DK", "DE", "IE", "EL", "ES",
    "FR", "IT", "HU", "NL", "AT", "PL", "PT",
    "RO", "FI", "SE"
]

YEARS = list(range(2020, 2025))

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

for directory in [RAW_DIR, PROCESSED_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

RAW_FILE = RAW_DIR / f"eurostat_edp_gdp_{timestamp}.json"

OUTPUT_FILE = (
    PROCESSED_DIR / "europe_eurostat_edp_gdp.csv"
)

REPORT_FILE = (
    REPORT_DIR / "europe_edp_gdp_extraction_report.txt"
)

# --------------------------------------------------
# EUROSTAT REQUEST
# --------------------------------------------------

params = [
    ("freq", "A"),
    ("unit", "MIO_NAC"),
    ("sector", "S1"),
    ("na_item", "B1GQ"),
    ("sinceTimePeriod", "2020"),
    ("untilTimePeriod", "2024"),
]

# Repeated geo parameters select multiple countries.
params.extend([
    ("geo", country) for country in COUNTRIES
])

print("Requesting Eurostat EDP GDP observations...")

response = requests.get(
    API_URL,
    params=params,
    timeout=90
)

print("\nHTTP STATUS:", response.status_code)
print("REQUEST URL:", response.url)
print("CONTENT TYPE:", response.headers.get("Content-Type"))
print("RESPONSE SIZE:", len(response.content))

response.raise_for_status()

data = response.json()

# Preserve the original response.
with RAW_FILE.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Raw response saved:", RAW_FILE)

# --------------------------------------------------
# JSON-STAT DECODING
# --------------------------------------------------

dimension_ids = data["id"]
sizes = data["size"]

category_maps = {}

for dimension_id in dimension_ids:

    index = data["dimension"][dimension_id][
        "category"
    ]["index"]

    if isinstance(index, dict):
        category_maps[dimension_id] = {
            position: code
            for code, position in index.items()
        }

    elif isinstance(index, list):
        category_maps[dimension_id] = {
            position: code
            for position, code in enumerate(index)
        }

    else:
        raise ValueError(
            f"Unsupported category index: {dimension_id}"
        )

values = data.get("value", {})

if isinstance(values, list):
    observations = enumerate(values)

elif isinstance(values, dict):
    observations = (
        (int(position), value)
        for position, value in values.items()
    )

else:
    raise ValueError("Unexpected JSON-stat value structure.")

records = []

for position, value in observations:

    # Missing observations are retained as absent,
    # not converted into zero.
    if value is None:
        continue

    remaining = int(position)
    coordinates = {}

    for dimension_id, size in zip(
        reversed(dimension_ids),
        reversed(sizes)
    ):
        coordinates[dimension_id] = remaining % size
        remaining //= size

    observation = {
        dimension_id: category_maps[dimension_id][
            coordinates[dimension_id]
        ]
        for dimension_id in dimension_ids
    }

    if (
        observation.get("freq") != "A"
        or observation.get("unit") != "MIO_NAC"
        or observation.get("sector") != "S1"
        or observation.get("na_item") != "B1GQ"
    ):
        continue

    records.append({
        "country": observation["geo"],
        "year": int(observation["time"]),
        "indicator": "B1GQ",
        "unit": "MIO_NAC",
        "sector": "S1",
        "gdp_value": float(value),
    })

df = pd.DataFrame(records)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

if not df.empty:

    if df.duplicated(["country", "year"]).any():
        raise ValueError("Duplicate country-year GDP records.")

    unexpected_countries = (
        set(df["country"]) - set(COUNTRIES)
    )

    unexpected_years = (
        set(df["year"]) - set(YEARS)
    )

    if unexpected_countries or unexpected_years:
        raise ValueError(
            f"Unexpected countries: {unexpected_countries}; "
            f"unexpected years: {unexpected_years}"
        )

expected = {
    (country, year)
    for country in COUNTRIES
    for year in YEARS
}

observed = (
    set(zip(df["country"], df["year"]))
    if not df.empty else set()
)

missing = sorted(expected - observed)

if not df.empty:
    df = df.sort_values(
        ["country", "year"]
    ).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

lines = [
    "EUROSTAT EDP GDP EXTRACTION REPORT",
    "=" * 70,
    "",
    f"Dataset: {DATASET}",
    "Indicator: B1GQ - GDP at market prices",
    "Unit: MIO_NAC",
    "Sector: S1",
    "Frequency: Annual",
    "Period: 2020-2024",
    "",
    f"Expected observations: {len(expected)}",
    f"Retrieved observations: {len(observed)}",
    f"Missing observations: {len(missing)}",
    f"Countries represented: {df['country'].nunique() if not df.empty else 0}",
    f"Years represented: {df['year'].nunique() if not df.empty else 0}",
    "",
    "MISSING COUNTRY-YEARS",
]

lines.extend(
    [f"{country}-{year}" for country, year in missing]
    if missing else ["None"]
)

lines.extend([
    "",
    "OUTPUT FILES",
    str(RAW_FILE),
    str(OUTPUT_FILE),
    "",
    "VALIDATION",
    f"Complete panel: {'YES' if len(missing) == 0 else 'NO'}",
    f"Duplicate country-years: {df.duplicated(['country', 'year']).sum() if not df.empty else 0}",
])

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\n".join(lines))
print("\nReport saved:", REPORT_FILE)
