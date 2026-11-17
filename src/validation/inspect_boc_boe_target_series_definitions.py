from openpyxl import load_workbook
from pathlib import Path

path = Path("data/external/boc_boe_sovereign_default_database_2025.xlsx")

targets = [
    "DEBT_EME_2025",
    "DEBT_EXAE_2025",
    "DEBT_GROSS_EME_OTHER_GDP_2025",
    "DEBT_GROSS_WORLD_PRODUCT_2025",
]

wb = load_workbook(path, read_only=False, data_only=False)

print("=" * 90)
print("WORKBOOK SHEETS")
print("=" * 90)

for ws in wb.worksheets:
    print(f"{ws.title}: {ws.max_row} rows x {ws.max_column} columns")

print("\n" + "=" * 90)
print("README / DEFINITIONS")
print("=" * 90)

for ws in wb.worksheets:
    if "readme" in ws.title.lower() or "definition" in ws.title.lower():
        for row in ws.iter_rows():
            values = [
                f"{cell.coordinate}={cell.value}"
                for cell in row if cell.value is not None
            ]
            if values:
                print(" | ".join(values))

ws = wb["Debt_2025"]

print("\n" + "=" * 90)
print("METADATA ROWS 1-64")
print("=" * 90)

for row in ws.iter_rows(min_row=1, max_row=64):
    values = [
        f"{cell.coordinate}={cell.value}"
        for cell in row if cell.value is not None
    ]
    if values:
        print(" | ".join(values))

print("\n" + "=" * 90)
print("TARGET SERIES COMMENTS AND HYPERLINKS")
print("=" * 90)

for row in ws.iter_rows(min_row=1, max_row=64):
    for cell in row:
        if cell.value is not None and any(
            term in str(cell.value) for term in targets
        ):
            print("\nSERIES:", cell.value)
            print("CELL:", cell.coordinate)
            print("COMMENT:", cell.comment.text if cell.comment else None)
            print("HYPERLINK:", cell.hyperlink.target if cell.hyperlink else None)

wb.close()
print("\nINSPECTION COMPLETED.")
