from openpyxl import load_workbook
from pathlib import Path

path = Path("data/external/boc_boe_sovereign_default_database_2025.xlsx")
wb = load_workbook(path, data_only=False)
ws = wb["2025 in Progress"]

print("=" * 100)
print("YEAR HEADERS / TOP-OF-SHEET STRUCTURE")
print("=" * 100)

for r in range(1, 8):
    print(f"\nROW {r}")
    for c in range(1, ws.max_column + 1):
        value = ws.cell(r, c).value
        if value is not None:
            print(f"{ws.cell(r, c).coordinate} = {value}", end=" | ")
    print()

print("\n" + "=" * 100)
print("TARGET ROWS: LABELS, VALUES, FORMULAS AND NUMBER FORMATS")
print("=" * 100)

target_rows = list(range(20, 38)) + list(range(58, 66))

for r in target_rows:
    print(f"\nROW {r}")
    for c in range(3, ws.max_column + 1):
        cell = ws.cell(r, c)
        if cell.value is not None:
            print(
                f"{cell.coordinate}: value={cell.value!r}; "
                f"format={cell.number_format!r}",
                end=" | "
            )
    print()

print("\n" + "=" * 100)
print("INSPECTION COMPLETED")
print("=" * 100)

wb.close()
