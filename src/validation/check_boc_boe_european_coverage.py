from pathlib import Path
from openpyxl import load_workbook
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"

COUNTRIES = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "CHE": "Switzerland",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SWE": "Sweden",
}

wb = load_workbook(SOURCE, read_only=True, data_only=True)
ws = wb["Debt_2025"]

print("OBSERVATION TABLE: FIRST THREE ROWS")
for row in ws.iter_rows(min_row=65, max_row=67, values_only=True):
    print(row[:18])

# Locate the observation header row.
header_row = 65
headers = [
    str(v).strip() if v is not None else ""
    for v in next(ws.iter_rows(
        min_row=header_row,
        max_row=header_row,
        values_only=True
    ))
]

print("\nCOLUMN HEADERS")
for i, value in enumerate(headers, start=1):
    if value:
        print(f"{i}: {value}")

# Identify likely country and year columns.
country_col = next(
    (i for i, h in enumerate(headers)
     if "COUNTRY" in h.upper() and "GROUP" not in h.upper()),
    None
)
year_col = next(
    (i for i, h in enumerate(headers)
     if h.upper() in {"YEAR", "DEBT_YEAR"}),
    None
)

if country_col is None or year_col is None:
    print("\nSTOP: Could not identify country/year columns.")
    wb.close()
    raise SystemExit(1)

# Collect unique country names and year range.
observed = set()
years = set()
country_rows = 0

for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
    country = row[country_col]
    year = row[year_col]

    if country is not None:
        observed.add(str(country).strip())

    if isinstance(year, (int, float)):
        years.add(int(year))

    if country is not None and year is not None:
        country_rows += 1

print("\nEUROPEAN PEER COVERAGE")
for code, name in COUNTRIES.items():
    matches = [
        item for item in observed
        if item.casefold() == name.casefold()
    ]
    print(f"{code} | {name} | {'FOUND' if matches else 'NOT FOUND'}")

print("\nDATASET SUMMARY")
print(f"Observation rows with country and year: {country_rows}")
print(f"Unique country labels: {len(observed)}")
print(f"Year range: {min(years) if years else 'N/A'}–{max(years) if years else 'N/A'}")

print("\nCOUNTRY LABELS")
print("; ".join(sorted(observed)))

wb.close()