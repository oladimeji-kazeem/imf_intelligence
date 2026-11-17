
from pathlib import Path
from openpyxl import load_workbook
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"
REPORTS = ROOT / "outputs" / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

FILES = [
    "boc_boe_sovereign_default_database_2025.xlsx",
    "kiel_monthly_sovereign_default_restructuring_2020.xlsx",
    "kiel_monthly_sovereign_default_1970_2025.xlsx.xlsx",
    "esm_domestic_sovereign_defaults.xlsx",
]

inventory = []
report = []

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for filename in FILES:
    path = EXTERNAL / filename

    report.append("=" * 90)
    report.append(f"SOURCE FILE: {filename}")
    report.append("=" * 90)

    if not path.exists():
        report.append("STATUS: FILE NOT FOUND")
        continue

    report.append(f"SIZE_BYTES: {path.stat().st_size}")
    report.append(f"SHA256: {sha256(path)}")

    try:
        wb = load_workbook(
            path,
            read_only=True,
            data_only=True
        )

        report.append(f"SHEETS: {len(wb.sheetnames)}")
        report.append(f"SHEET_NAMES: {wb.sheetnames}")

        for ws in wb.worksheets:
            report.append("")
            report.append(f"WORKSHEET: {ws.title}")
            report.append(f"MAX_ROW: {ws.max_row}")
            report.append(f"MAX_COLUMN: {ws.max_column}")

            nonempty_rows = []

            for row in ws.iter_rows(values_only=True):
                values = list(row)

                if any(value is not None for value in values):
                    nonempty_rows.append(values)

                if len(nonempty_rows) >= 12:
                    break

            header_row = None

            for index, row in enumerate(nonempty_rows, start=1):
                if sum(value is not None for value in row) >= 2:
                    header_row = index
                    break

            headers = (
                nonempty_rows[header_row - 1]
                if header_row else []
            )

            report.append(f"FIRST_NONEMPTY_HEADER_ROW: {header_row}")
            report.append(f"HEADER_CANDIDATES: {headers}")

            report.append("SAMPLE_ROWS:")
            for row in nonempty_rows[:8]:
                report.append(str(row))

            inventory.append({
                "filename": filename,
                "sha256": sha256(path),
                "sheet_name": ws.title,
                "max_row": ws.max_row,
                "max_column": ws.max_column,
                "header_row_candidate": header_row,
                "header_candidates": " | ".join(
                    str(x) if x is not None else ""
                    for x in headers
                ),
            })

        wb.close()

    except Exception as exc:
        report.append(f"INSPECTION_ERROR: {type(exc).__name__}: {exc}")

    report.append("")

csv_path = REPORTS / "sovereign_default_workbook_inventory.csv"
txt_path = REPORTS / "sovereign_default_workbook_inspection.txt"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "sha256",
            "sheet_name",
            "max_row",
            "max_column",
            "header_row_candidate",
            "header_candidates",
        ],
    )
    writer.writeheader()
    writer.writerows(inventory)

txt_path.write_text("\n".join(report), encoding="utf-8")

print("=" * 90)
print("SOVEREIGN DEFAULT WORKBOOK INSPECTION")
print("=" * 90)
print(f"Workbooks requested: {len(FILES)}")
print(f"Worksheet records inspected: {len(inventory)}")
print(f"CSV: {csv_path}")
print(f"REPORT: {txt_path}")
print("Inspection completed.")
