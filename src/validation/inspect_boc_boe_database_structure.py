
from pathlib import Path
from openpyxl import load_workbook
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "external" / "boc_boe_sovereign_default_database_2025.xlsx"
REPORTS = ROOT / "outputs" / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

print("Opening BoC-BoE workbook...", flush=True)

wb = load_workbook(
    SOURCE,
    read_only=True,
    data_only=True
)

report = []
metadata_rows = []

# --------------------------------------------------
# SHEET 1: ANNUAL SUMMARY
# --------------------------------------------------

print("Inspecting annual summary sheet...", flush=True)

ws = wb["2025 in Progress"]

report.extend([
    "=" * 100,
    "SHEET: 2025 in Progress",
    "=" * 100,
    f"Dimensions: {ws.max_row} rows x {ws.max_column} columns",
])

for row_number, row in enumerate(
    ws.iter_rows(
        min_row=1,
        max_row=min(ws.max_row, 100),
        values_only=True
    ),
    start=1
):
    populated = [
        f"{index + 1}={value}"
        for index, value in enumerate(row)
        if value is not None
    ]

    if populated:
        report.append(
            f"ROW {row_number}: " + " | ".join(populated)
        )

# --------------------------------------------------
# SHEET 2: DEBT DATA
# --------------------------------------------------

print("Inspecting Debt_2025 sheet...", flush=True)

ws = wb["Debt_2025"]

debt_max_row = ws.max_row
debt_max_column = ws.max_column

report.extend([
    "",
    "=" * 100,
    "SHEET: Debt_2025",
    "=" * 100,
    f"Dimensions: {debt_max_row} rows x {debt_max_column} columns",
    "FIRST 80 ROWS",
])

column_counts = [0] * debt_max_column

for row_number, row in enumerate(
    ws.iter_rows(values_only=True),
    start=1
):
    # Count populated cells in each column.
    for index, value in enumerate(row):
        if value is not None:
            column_counts[index] += 1

    # Record first 80 rows.
    if row_number <= 80:
        populated = [
            f"{index + 1}={value}"
            for index, value in enumerate(row)
            if value is not None
        ]

        if populated:
            report.append(
                f"ROW {row_number}: " + " | ".join(populated)
            )

        metadata_rows.append({
            "row_number": row_number,
            "values": " | ".join(
                str(value) if value is not None else ""
                for value in row
            )
        })

report.extend([
    "",
    "COLUMN POPULATION COUNTS",
])

for index, count in enumerate(column_counts, start=1):
    report.append(
        f"COLUMN {index}: {count} nonempty cells"
    )

wb.close()

# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

txt_path = REPORTS / "boc_boe_database_structure_inspection.txt"
csv_path = REPORTS / "boc_boe_database_first_80_rows.csv"

txt_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["row_number", "values"]
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print("\n" + "=" * 90)
print("BOC-BOE DATABASE STRUCTURE INSPECTION")
print("=" * 90)
print(f"Annual summary dimensions: 1396 rows x 85 columns")
print(f"Debt data dimensions: {debt_max_row} rows x {debt_max_column} columns")
print(f"Report: {txt_path}")
print(f"CSV: {csv_path}")
print("Inspection completed.", flush=True)
