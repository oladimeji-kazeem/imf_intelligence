import json
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/raw/eurostat_sfa_probe_20261114T074434Z.json"
)

OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "europe_eurostat_sfa_reconciliation.csv"
REPORT_FILE = REPORT_DIR / "europe_eurostat_sfa_reconciliation_report.txt"

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

with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

dimension_ids = data["id"]
sizes = data["size"]

# Build reverse category maps.
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
        observation.get("freq") != "A"
        or observation.get("sector") != "S13"
        or observation.get("na_item") not in TARGET_INDICATORS
        or observation.get("unit") not in TARGET_UNITS
    ):
        continue

    records.append({
        "country": observation["geo"],
        "year": int(observation["time"]),
        "indicator": observation["na_item"],
        "unit": observation["unit"],
        "value": float(value),
    })

long_df = pd.DataFrame(records)

if long_df.duplicated(
    ["country", "year", "indicator", "unit"]
).any():
    raise ValueError("Duplicate Eurostat observations detected.")

wide = long_df.pivot(
    index=["country", "year", "unit"],
    columns="indicator",
    values="value"
).reset_index()

# Preserve the raw components. Do not replace missing values with zero.
required = ["GD_CH", "B9_T3"]

for column in required:
    if column not in wide.columns:
        raise ValueError(f"Required indicator missing: {column}")

wide["sfa_calculated"] = (
    wide["GD_CH"] - wide["B9_T3"]
)

wide["sfa_available"] = (
    wide["GD_CH"].notna() & wide["B9_T3"].notna()
)

wide["reconciliation_status"] = wide["sfa_available"].map({
    True: "calculated",
    False: "missing_required_component"
})

# Save detailed panel.
wide.to_csv(OUTPUT_FILE, index=False)

# Unit consistency test.
unit_results = {}

for unit in TARGET_UNITS:
    subset = wide[wide["unit"] == unit].copy()

    valid = subset.dropna(
        subset=["GD_CH", "B9_T3", "sfa_calculated"]
    )

    unit_results[unit] = {
        "records": len(subset),
        "valid_sfa": len(valid),
        "missing_sfa": len(subset) - len(valid),
        "countries": subset["country"].nunique(),
        "years": subset["year"].nunique(),
    }

# Compare the independently calculated SFA in both units.
# Compare the direction of SFA across units.
nac = wide[wide["unit"] == "MIO_NAC"][
    ["country", "year", "sfa_calculated"]
].rename(columns={"sfa_calculated": "sfa_nac"})

gdp = wide[wide["unit"] == "PC_GDP"][
    ["country", "year", "sfa_calculated"]
].rename(columns={"sfa_calculated": "sfa_pct_gdp"})

comparison = nac.merge(
    gdp,
    on=["country", "year"],
    how="inner"
)

comparison = comparison.dropna(
    subset=["sfa_nac", "sfa_pct_gdp"]
)

comparison["same_direction"] = (
    (comparison["sfa_nac"] > 0)
    == (comparison["sfa_pct_gdp"] > 0)
)

sign_mismatches = int(
    (~comparison["same_direction"]).sum()
)


# Report.
lines = [
    "EUROSTAT SFA RECONCILIATION REPORT",
    "=" * 65,
    "",
    "Formula: SFA = GD_CH - B9_T3",
    "Sector: S13",
    "Period: 2020-2024",
    "",
    "UNIT COVERAGE",
]

for unit, result in unit_results.items():
    lines.extend([
        f"\nUnit: {unit}",
        f"Records: {result['records']}",
        f"Calculated SFA: {result['valid_sfa']}",
        f"Missing SFA: {result['missing_sfa']}",
        f"Countries: {result['countries']}",
        f"Years: {result['years']}",
    ])

lines.extend([
    "",
    "UNIT COMPARISON",
    f"Matched country-years: {len(comparison)}",
    f"Direction mismatches: {sign_mismatches}",
    "",
    "IMPORTANT LIMITATIONS",
    "- SFA is calculated only where GD_CH and B9_T3 are both present.",
    "- Missing observations are not treated as zero.",
    "- Monetary and percentage-of-GDP values are retained separately.",
    "- Detailed adjustment components are not added together here.",
    "- This is an accounting reconciliation, not causal identification.",
    "",
    f"Dataset: {OUTPUT_FILE}",
])

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\n".join(lines))
print("\nReport saved:", REPORT_FILE)
