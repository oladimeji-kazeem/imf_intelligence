
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
OUTPUT = ROOT / "data/processed"
REPORT = ROOT / "outputs/reports/boc_boe_full_panel_series_scope_audit.txt"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

SHEET = "Debt_2025"
HEADER_ROW = 64

ID_FIELDS = [
    "k",
    "DEBT_COUNTRY",
    "DEBT_COUNTRY_GROUP",
    "DEBT_YEAR",
]

PEERS = {
    "AUT": "Austria", "BEL": "Belgium", "CHE": "Switzerland",
    "CZE": "Czech Republic", "DEU": "Germany", "DNK": "Denmark",
    "ESP": "Spain", "FIN": "Finland", "FRA": "France",
    "GBR": "United Kingdom", "GRC": "Greece", "HUN": "Hungary",
    "IRL": "Ireland", "ITA": "Italy", "NLD": "Netherlands",
    "NOR": "Norway", "POL": "Poland", "PRT": "Portugal",
    "ROU": "Romania", "SWE": "Sweden",
}

def write_report(lines):
    REPORT.write_text("\n".join(lines), encoding="utf-8")

print("Reading full BoC–BoE observation panel...")

df = pd.read_excel(
    WORKBOOK,
    sheet_name=SHEET,
    header=HEADER_ROW,
)

df.columns = [str(c).strip() for c in df.columns]

series_fields = [c for c in df.columns if c not in ID_FIELDS]

# Read metadata catalogue
wb = load_workbook(WORKBOOK, read_only=True, data_only=True)
ws = wb[SHEET]

metadata = []

for row in ws.iter_rows(min_row=12, max_row=62, values_only=True):
    if row[0] is not None:
        metadata.append({
            "series_id": str(row[0]).strip(),
            "label": str(row[1]).strip() if row[1] is not None else "",
            "description": str(row[2]).strip() if row[2] is not None else "",
        })

catalogue = pd.DataFrame(metadata)

data_set = set(series_fields)
metadata_set = set(catalogue["series_id"])

metadata_only = sorted(metadata_set - data_set)
data_only = sorted(data_set - metadata_set)

# Numeric conversion
for field in series_fields:
    df[field] = pd.to_numeric(df[field], errors="coerce")

# Identify individual peer-country rows
peer_names = {v.upper() for v in PEERS.values()}

df["row_scope"] = np.where(
    df["DEBT_COUNTRY"].astype(str).str.upper().isin(peer_names),
    "INDIVIDUAL_PEER_COUNTRY",
    "NON_PEER_OR_AGGREGATE",
)

# Full-panel series availability
profile = []

for field in series_fields:
    s = df[field]
    profile.append({
        "series_id": field,
        "available": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "availability_pct": round(s.notna().mean() * 100, 2),
        "positive": int((s > 0).sum()),
        "zero": int((s == 0).sum()),
        "negative": int((s < 0).sum()),
        "minimum": s.min(),
        "maximum": s.max(),
    })

profile_df = pd.DataFrame(profile)
profile_df.to_csv(
    OUTPUT / "boc_boe_full_panel_series_profile.csv",
    index=False,
)

# Availability by row scope
scope_rows = []

for scope, group in df.groupby("row_scope"):
    for field in series_fields:
        s = group[field]
        scope_rows.append({
            "row_scope": scope,
            "series_id": field,
            "rows": len(group),
            "available": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "availability_pct": round(s.notna().mean() * 100, 2),
            "positive": int((s > 0).sum()),
            "zero": int((s == 0).sum()),
        })

scope_df = pd.DataFrame(scope_rows)
scope_df.to_csv(
    OUTPUT / "boc_boe_full_panel_series_by_scope.csv",
    index=False,
)

# Availability by country-group label
group_rows = []

for group_name, group in df.groupby("DEBT_COUNTRY_GROUP", dropna=False):
    for field in series_fields:
        s = group[field]
        group_rows.append({
            "country_group": group_name,
            "series_id": field,
            "rows": len(group),
            "available": int(s.notna().sum()),
            "availability_pct": round(s.notna().mean() * 100, 2),
        })

group_df = pd.DataFrame(group_rows)
group_df.to_csv(
    OUTPUT / "boc_boe_full_panel_series_by_country_group.csv",
    index=False,
)

# Compare the two total fields
total_comparison = {}

if {"DEBT_TOTAL_2025", "DEBT_TOTAL_DEBT_2025"}.issubset(df.columns):
    a = df["DEBT_TOTAL_2025"]
    b = df["DEBT_TOTAL_DEBT_2025"]
    both = a.notna() & b.notna()

    total_comparison = {
        "both_available": int(both.sum()),
        "equal_values": int((a[both] == b[both]).sum()),
        "different_values": int((a[both] != b[both]).sum()),
        "max_absolute_difference": (
            float((a[both] - b[both]).abs().max())
            if both.any() else None
        ),
    }

    comparison_df = df.loc[
        both,
        ID_FIELDS + ["DEBT_TOTAL_2025", "DEBT_TOTAL_DEBT_2025"],
    ].copy()

    comparison_df["absolute_difference"] = (
        comparison_df["DEBT_TOTAL_2025"]
        - comparison_df["DEBT_TOTAL_DEBT_2025"]
    ).abs()

    comparison_df.to_csv(
        OUTPUT / "boc_boe_total_series_comparison.csv",
        index=False,
    )

# Country labels
country_labels = (
    df["DEBT_COUNTRY"]
    .dropna()
    .astype(str)
    .value_counts()
    .rename_axis("country_label")
    .reset_index(name="row_count")
)

country_labels.to_csv(
    OUTPUT / "boc_boe_full_panel_country_labels.csv",
    index=False,
)

# Report
lines = [
    "BOC–BOE FULL PANEL SERIES SCOPE AUDIT",
    "=" * 75,
    "",
    f"Observation rows: {len(df)}",
    f"Observation variables: {len(series_fields)}",
    f"Metadata series: {len(catalogue)}",
    f"Metadata-only series: {len(metadata_only)}",
    f"Observation-only series: {len(data_only)}",
    "",
    "METADATA / OBSERVATION RECONCILIATION",
    "-" * 75,
    f"Metadata-only: {metadata_only}",
    f"Observation-only: {data_only}",
    "",
    "ROW SCOPE",
    "-" * 75,
]

for scope, group in df.groupby("row_scope"):
    lines.append(f"{scope}: {len(group)} rows")

lines += [
    "",
    "FULL-PANEL SERIES AVAILABILITY",
    "-" * 75,
]

for _, row in profile_df.sort_values(
    "availability_pct", ascending=False
).iterrows():
    lines.append(
        f"{row['series_id']} | "
        f"available={row['available']} | "
        f"availability={row['availability_pct']}% | "
        f"positive={row['positive']} | "
        f"zero={row['zero']}"
    )

lines += [
    "",
    "TOTAL SERIES COMPARISON",
    "-" * 75,
]

for key, value in total_comparison.items():
    lines.append(f"{key}: {value}")

lines += [
    "",
    "COUNTRY-GROUP LABELS",
    "-" * 75,
]

for value, count in df["DEBT_COUNTRY_GROUP"].value_counts(dropna=False).items():
    lines.append(f"{value}: {count}")

lines += [
    "",
    "INTERPRETATION",
    "-" * 75,
    "Full-panel availability does not establish individual-country suitability.",
    "Aggregate and country-group rows must not be mistaken for sovereign rows.",
    "Identical total-series values do not establish semantic equivalence.",
    "Metadata-only fields require investigation before exclusion.",
]

write_report(lines)

print("\n".join(lines))

print("\nOUTPUT FILES")
for filename in [
    "boc_boe_full_panel_series_profile.csv",
    "boc_boe_full_panel_series_by_scope.csv",
    "boc_boe_full_panel_series_by_country_group.csv",
    "boc_boe_full_panel_country_labels.csv",
    "boc_boe_total_series_comparison.csv",
]:
    print(OUTPUT / filename)

print(REPORT)
