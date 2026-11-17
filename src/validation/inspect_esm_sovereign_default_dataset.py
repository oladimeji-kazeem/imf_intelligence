
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data" / "external"
    / "esm_domestic_sovereign_defaults.xlsx"
)

REPORT = (
    ROOT / "outputs" / "reports"
    / "esm_sovereign_default_source_inspection.txt"
)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Download the ESM workbook first: {SOURCE}"
    )

xls = pd.ExcelFile(SOURCE)

lines = [
    "=" * 75,
    "ESM SOVEREIGN DEFAULT DATASET — SOURCE INSPECTION",
    "=" * 75,
    "",
    f"Source: {SOURCE}",
    f"Worksheets: {len(xls.sheet_names)}",
    "",
]

for sheet in xls.sheet_names:
    df = pd.read_excel(SOURCE, sheet_name=sheet)

    lines.extend([
        "=" * 75,
        f"WORKSHEET: {sheet}",
        "=" * 75,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "COLUMN NAMES",
        "-" * 75,
    ])

    for column in df.columns:
        lines.append(f"{column} | {df[column].dtype}")

    lines.extend([
        "",
        "MISSINGNESS",
        "-" * 75,
    ])

    missing = df.isna().sum()

    for column, count in missing.items():
        if count:
            lines.append(f"{column}: {count}")

    lines.extend([
        "",
        "SAMPLE RECORDS",
        "-" * 75,
        df.head(5).to_string(index=False),
        "",
    ])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nInspection report saved: {REPORT}")
