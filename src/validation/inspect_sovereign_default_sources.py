
from pathlib import Path
import pandas as pd
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"
OUTPUT = ROOT / "outputs" / "reports"

OUTPUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SOURCES = {
    "BoC-BoE sovereign default database": [
        "boc_boe_sovereign_default_database"
    ],
    "Kiel monthly sovereign default dataset": [
        "kiel_monthly_sovereign_default_1970_2020"
    ],
    "ESM domestic sovereign defaults": [
        "esm_domestic_sovereign_defaults"
    ],
}

SUPPORTED = {".xlsx", ".xlsm", ".xls", ".csv"}

def locate_source(stems):
    for stem in stems:
        matches = [
            p for p in EXTERNAL.glob("*")
            if p.is_file()
            and p.stem.lower() == stem.lower()
            and p.suffix.lower() in SUPPORTED
        ]
        if matches:
            return matches[0]
    return None

def inspect_excel(path):
    result = []

    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        return [{
            "sheet": "",
            "rows": None,
            "columns": None,
            "column_names": "",
            "sample": "",
            "inspection_error": str(exc),
        }]

    for sheet in workbook.sheet_names:
        try:
            preview = pd.read_excel(
                path,
                sheet_name=sheet,
                nrows=5
            )

            # Count worksheet rows without loading the entire sheet.
            row_count = None
            if path.suffix.lower() in {".xlsx", ".xlsm"}:
                try:
                    from openpyxl import load_workbook
                    wb = load_workbook(
                        path,
                        read_only=True,
                        data_only=True
                    )
                    ws = wb[sheet]
                    row_count = max(ws.max_row - 1, 0)
                    wb.close()
                except Exception:
                    pass

            sample = preview.head(5).to_dict(orient="records")

            result.append({
                "sheet": sheet,
                "rows": row_count,
                "columns": len(preview.columns),
                "column_names": " | ".join(
                    str(c) for c in preview.columns
                ),
                "sample": json.dumps(
                    sample,
                    default=str,
                    ensure_ascii=False
                ),
                "inspection_error": "",
            })

        except Exception as exc:
            result.append({
                "sheet": sheet,
                "rows": None,
                "columns": None,
                "column_names": "",
                "sample": "",
                "inspection_error": str(exc),
            })

    return result

def inspect_csv(path):
    try:
        preview = pd.read_csv(path, nrows=5)
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            row_count = max(sum(1 for _ in f) - 1, 0)

        return [{
            "sheet": "CSV",
            "rows": row_count,
            "columns": len(preview.columns),
            "column_names": " | ".join(
                str(c) for c in preview.columns
            ),
            "sample": json.dumps(
                preview.head(5).to_dict(orient="records"),
                default=str,
                ensure_ascii=False
            ),
            "inspection_error": "",
        }]
    except Exception as exc:
        return [{
            "sheet": "CSV",
            "rows": None,
            "columns": None,
            "column_names": "",
            "sample": "",
            "inspection_error": str(exc),
        }]

inventory = []
report = [
    "=" * 78,
    "SOVEREIGN DEFAULT SOURCE INSPECTION REPORT",
    "=" * 78,
    "",
    f"External data directory: {EXTERNAL}",
    "",
]

for source_name, stems in EXPECTED_SOURCES.items():
    path = locate_source(stems)

    report.extend([
        source_name.upper(),
        "-" * 78,
    ])

    if path is None:
        report.append("STATUS: FILE NOT FOUND")
        report.append("")
        inventory.append({
            "source_name": source_name,
            "file": "",
            "status": "file_not_found",
            "sheet": "",
            "rows": None,
            "columns": None,
            "column_names": "",
            "sample": "",
            "inspection_error": "",
        })
        continue

    report.append(f"File: {path.name}")
    report.append(f"Size (MB): {path.stat().st_size / (1024**2):.3f}")

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = inspect_excel(path)
    else:
        sheets = inspect_csv(path)

    for item in sheets:
        inventory.append({
            "source_name": source_name,
            "file": str(path),
            "status": (
                "inspection_error"
                if item["inspection_error"]
                else "inspected"
            ),
            **item,
        })

        report.extend([
            "",
            f"Sheet: {item['sheet']}",
            f"Rows (excluding header, where available): {item['rows']}",
            f"Columns: {item['columns']}",
            f"Column names: {item['column_names']}",
            "Sample records:",
            item["sample"],
        ])

        if item["inspection_error"]:
            report.append(
                f"INSPECTION ERROR: {item['inspection_error']}"
            )

    report.append("")

inventory_df = pd.DataFrame(inventory)

inventory_path = (
    OUTPUT / "sovereign_default_source_inventory.csv"
)
report_path = (
    OUTPUT / "sovereign_default_source_inspection.txt"
)

inventory_df.to_csv(
    inventory_path,
    index=False,
    encoding="utf-8"
)

report.append("=" * 78)
report.append("INSPECTION SUMMARY")
report.append("=" * 78)

for source_name in EXPECTED_SOURCES:
    subset = inventory_df[
        inventory_df["source_name"] == source_name
    ]
    found = (
        len(subset) > 0
        and (subset["status"] != "file_not_found").any()
    )
    report.append(
        f"{source_name}: "
        f"{'FOUND' if found else 'NOT FOUND'}"
    )

report.extend([
    "",
    "OUTPUT FILES",
    "-" * 78,
    str(inventory_path),
    str(report_path),
])

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n".join(report))
print("\nInspection completed.")
