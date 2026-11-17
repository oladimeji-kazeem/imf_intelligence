
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
OUTPUT = ROOT / "data/processed"
REPORT = ROOT / "outputs/reports/boc_boe_aggregate_series_rows_audit.txt"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    WORKBOOK,
    sheet_name="Debt_2025",
    header=64
)

df.columns = [str(c).strip() for c in df.columns]

AGGREGATE_FIELDS = [
    "DEBT_ADVANCED_ECON_2025",
    "DEBT_EME_2025",
    "DEBT_HIPC_2025",
    "DEBT_OTHER_2025",
    "DEBT_EXAE_2025",
    "DEBT_SSA_2025",
    "DEBT_DEFRATE_SOVEREIGNS_2025",
    "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
    "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
    "DEBT_DEFRATE_WORLD_GDP_2025",
    "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
    "DEBT_ARREARS_DEFRATE_CONV_DEF_2025",
    "DEBT_ARREARS_DEFRATE_WORLD_GDP_2025",
    "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
    "DEBT_GROSS_WORLD_PUBLIC_DEBT_2025",
    "DEBT_DEFRATE_EME_OTHER_GDP_2025",
    "DEBT_DEFRATE_WORLD_PUBLIC_DEBT_2025",
    "DEBT_DEFRATE_CHINA_2025",
    "DEBT_GROSS_EME_OTHER_GDP_2025",
    "DEBT_GROSS_CHINA_2025",
    "DEBT_DEFRATE_PARIS_CLUB_DEBT_2025",
    "DEBT_PARIS_CLUB_DEBT_2025",
    "DEBT_GROSS_WORLD_PRODUCT_2025",
]

AGGREGATE_FIELDS = [
    c for c in AGGREGATE_FIELDS if c in df.columns
]

for field in AGGREGATE_FIELDS:
    df[field] = pd.to_numeric(df[field], errors="coerce")

# Keep rows with at least one aggregate indicator
mask = df[AGGREGATE_FIELDS].notna().any(axis=1)
aggregate_rows = df.loc[mask].copy()

# Detailed row-level evidence
detail_fields = [
    c for c in [
        "k",
        "DEBT_COUNTRY",
        "DEBT_COUNTRY_GROUP",
        "DEBT_YEAR",
    ] + AGGREGATE_FIELDS
    if c in aggregate_rows.columns
]

aggregate_rows[detail_fields].to_csv(
    OUTPUT / "boc_boe_aggregate_indicator_observations.csv",
    index=False
)

# Count observations by country label
aggregate_rows["_indicators_populated"] = aggregate_rows[AGGREGATE_FIELDS].notna().sum(axis=1)

country_summary = (
    aggregate_rows.groupby(
        ["DEBT_COUNTRY", "DEBT_COUNTRY_GROUP"],
        dropna=False
    )
    .agg(
        rows=("DEBT_YEAR", "size"),
        first_year=("DEBT_YEAR", "min"),
        last_year=("DEBT_YEAR", "max"),
        indicators_populated=("_indicators_populated", "sum"),
    )
    .reset_index()
)

country_summary.to_csv(
    OUTPUT / "boc_boe_aggregate_indicator_country_summary.csv",
    index=False
)

# Availability by aggregate series and country label
availability = (
    aggregate_rows
    .groupby("DEBT_COUNTRY", dropna=False)[AGGREGATE_FIELDS]
    .agg(lambda x: int(x.notna().sum()))
)

availability.to_csv(
    OUTPUT / "boc_boe_aggregate_indicator_availability_by_country.csv"
)

# Individual series detail
series_rows = []

for field in AGGREGATE_FIELDS:
    populated = aggregate_rows.loc[
        aggregate_rows[field].notna()
    ]

    series_rows.append({
        "series_id": field,
        "populated_rows": len(populated),
        "country_labels": "; ".join(
            sorted(populated["DEBT_COUNTRY"].dropna().astype(str).unique())
        ),
        "country_group_labels": "; ".join(
            sorted(
                populated["DEBT_COUNTRY_GROUP"]
                .dropna().astype(str).unique()
            )
        ),
        "first_year": populated["DEBT_YEAR"].min(),
        "last_year": populated["DEBT_YEAR"].max(),
    })

series_summary = pd.DataFrame(series_rows)

series_summary.to_csv(
    OUTPUT / "boc_boe_aggregate_indicator_series_summary.csv",
    index=False
)

lines = [
    "BOC–BOE AGGREGATE SERIES ROWS AUDIT",
    "=" * 75,
    "",
    f"Full panel rows: {len(df)}",
    f"Rows with at least one aggregate indicator: {len(aggregate_rows)}",
    f"Aggregate variables inspected: {len(AGGREGATE_FIELDS)}",
    "",
    "COUNTRY LABELS CONTAINING AGGREGATE INDICATORS",
    "-" * 75,
]

for _, row in country_summary.iterrows():
    lines.append(
        f"{row['DEBT_COUNTRY']} | "
        f"group={row['DEBT_COUNTRY_GROUP']} | "
        f"rows={row['rows']} | "
        f"years={row['first_year']}–{row['last_year']} | "
        f"populated_cells={row['indicators_populated']}"
    )

lines += [
    "",
    "SERIES-LEVEL COVERAGE",
    "-" * 75,
]

for _, row in series_summary.iterrows():
    lines.append(
        f"{row['series_id']} | "
        f"rows={row['populated_rows']} | "
        f"countries={row['country_labels']} | "
        f"groups={row['country_group_labels']} | "
        f"years={row['first_year']}–{row['last_year']}"
    )

REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

print("\nOUTPUT FILES")
print(OUTPUT / "boc_boe_aggregate_indicator_observations.csv")
print(OUTPUT / "boc_boe_aggregate_indicator_country_summary.csv")
print(OUTPUT / "boc_boe_aggregate_indicator_availability_by_country.csv")
print(OUTPUT / "boc_boe_aggregate_indicator_series_summary.csv")
print(REPORT)
