from pathlib import Path
from collections import defaultdict
from openpyxl import load_workbook
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PEERS = {
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

# Historical labels are flagged for investigation, never automatically
# allocated to modern successor countries.
HISTORICAL_LABELS = {
    "CZE": ["Czechoslovakia"],
}

UNALLOCATED_HISTORICAL_LABELS = [
    "USSR/Russian Federation",
    "Yugoslavia",
]

wb = load_workbook(SOURCE, read_only=True, data_only=True)
ws = wb["Debt_2025"]

headers = [
    str(v).strip() if v is not None else ""
    for v in next(ws.iter_rows(min_row=65, max_row=65, values_only=True))
]

country_idx = headers.index("DEBT_COUNTRY")
year_idx = headers.index("DEBT_YEAR")
total_idx = headers.index("DEBT_TOTAL_2025")

stats = defaultdict(lambda: {
    "rows": 0,
    "years": set(),
    "missing_total": 0,
    "zero_total": 0,
    "positive_total": 0,
})

for row in ws.iter_rows(min_row=66, values_only=True):
    country = row[country_idx]
    year = row[year_idx]

    if country is None:
        continue

    country = str(country).strip()
    item = stats[country]
    item["rows"] += 1

    if isinstance(year, (int, float)):
        item["years"].add(int(year))

    value = row[total_idx]

    if value is None:
        item["missing_total"] += 1
    elif value == 0:
        item["zero_total"] += 1
    elif value > 0:
        item["positive_total"] += 1

wb.close()

# Historical labels are reported separately and never allocated to
# modern successor countries.
HISTORICAL_SERIES = {
    "Czechoslovakia": "SHARED_PREDECESSOR_CZECHIA_SLOVAKIA",
    "USSR/Russian Federation": "COMBINED_HISTORICAL_ENTITY_UNALLOCATED",
    "Yugoslavia": "PREDECESSOR_STATE_UNALLOCATED",
}

historical_output = []

for label, attribution_status in HISTORICAL_SERIES.items():
    item = stats.get(label)
    if item is None:
        historical_output.append({
            "historical_label": label,
            "attribution_status": attribution_status,
            "record_status": "NO_MATCHING_LABEL",
            "observation_rows": 0,
            "first_year": "",
            "last_year": "",
            "missing_total_rows": 0,
            "zero_total_rows": 0,
            "positive_total_rows": 0,
        })
        continue

    years = item["years"]
    historical_output.append({
        "historical_label": label,
        "attribution_status": attribution_status,
        "record_status": "HISTORICAL_RECORD_FOUND",
        "observation_rows": item["rows"],
        "first_year": min(years) if years else "",
        "last_year": max(years) if years else "",
        "missing_total_rows": item["missing_total"],
        "zero_total_rows": item["zero_total"],
        "positive_total_rows": item["positive_total"],
    })

historical_csv_path = PROCESSED / "boc_boe_historical_label_coverage.csv"
with historical_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=historical_output[0].keys())
    writer.writeheader()
    writer.writerows(historical_output)

output = []

for code, name in PEERS.items():
    item = stats.get(name, {
        "rows": 0,
        "years": set(),
        "missing_total": 0,
        "zero_total": 0,
        "positive_total": 0,
    })

    years = item["years"]

    historical = [
        label for label in HISTORICAL_LABELS.get(code, [])
        if label in stats
    ]

    if years:
        coverage_interpretation = "DIRECT_COUNTRY_LABEL_MATCH"
    elif historical:
        coverage_interpretation = "HISTORICAL_PREDECESSOR_ONLY_ATTRIBUTION_UNRESOLVED"
    else:
        coverage_interpretation = "NO_MATCHING_COUNTRY_OR_CONFIGURED_PREDECESSOR_LABEL"

    output.append({
        "iso3": code,
        "country": name,
        "coverage_status": "DIRECT_MATCH" if years else "NO_DIRECT_MATCH",
        "coverage_interpretation": coverage_interpretation,
        "observation_rows": item["rows"],
        "first_year": min(years) if years else "",
        "last_year": max(years) if years else "",
        "missing_total_rows": item["missing_total"],
        "zero_total_rows": item["zero_total"],
        "positive_total_rows": item["positive_total"],
        "historical_labels_for_review": "; ".join(historical),
    })

csv_path = PROCESSED / "boc_boe_europe_peer_coverage.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=output[0].keys())
    writer.writeheader()
    writer.writerows(output)

direct = sum(x["coverage_status"] == "DIRECT_MATCH" for x in output)

report = [
    "BOC–BOE EUROPEAN PEER COVERAGE AUDIT",
    "=" * 60,
    f"Peer countries: {len(PEERS)}",
    f"Directly represented: {direct}",
    f"No direct match: {len(PEERS) - direct}",
    f"Historical predecessor only, attribution unresolved: "
    f"{sum(x['coverage_interpretation'] == 'HISTORICAL_PREDECESSOR_ONLY_ATTRIBUTION_UNRESOLVED' for x in output)}",
    "",
    "COUNTRY COVERAGE",
    "-" * 60,
]

for x in output:
    report.append(
        f"{x['iso3']} | {x['country']} | {x['coverage_status']} | "
        f"interpretation={x['coverage_interpretation']} | "
        f"rows={x['observation_rows']} | "
        f"years={x['first_year']}-{x['last_year']} | "
        f"positive={x['positive_total_rows']} | "
        f"zero={x['zero_total_rows']} | "
        f"missing={x['missing_total_rows']} | "
        f"historical_review={x['historical_labels_for_review']}"
    )

report.extend([
    "",
    "HISTORICAL LABEL COVERAGE — NOT ALLOCATED TO MODERN PEERS",
    "-" * 60,
])

for x in historical_output:
    report.append(
        f"{x['historical_label']} | {x['record_status']} | "
        f"attribution={x['attribution_status']} | "
        f"rows={x['observation_rows']} | "
        f"years={x['first_year']}-{x['last_year']} | "
        f"positive={x['positive_total_rows']} | "
        f"zero={x['zero_total_rows']} | "
        f"missing={x['missing_total_rows']}"
    )

txt_path = REPORTS / "boc_boe_europe_peer_coverage.txt"
txt_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nCSV: {csv_path}")
print(f"Historical CSV: {historical_csv_path}")
print(f"Report: {txt_path}")
