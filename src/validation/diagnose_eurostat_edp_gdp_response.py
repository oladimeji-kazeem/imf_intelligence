import json
from pathlib import Path

INPUT_FILE = Path(
    "data/raw/eurostat_edp_gdp_20261114T080925Z.json"
)

with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("\nEUROSTAT GDP RESPONSE DIAGNOSTIC")
print("=" * 75)

print("\nDATASET METADATA")
print("Label:", data.get("label"))
print("ID:", data.get("id"))
print("Size:", data.get("size"))
print("Updated:", data.get("updated"))

print("\nDIMENSION CATEGORIES")

for dimension_id in data.get("id", []):

    dimension = data["dimension"][dimension_id]
    category = dimension.get("category", {})

    index = category.get("index", {})
    labels = category.get("label", {})

    print(f"\n{dimension_id}")
    print("Label:", dimension.get("label"))
    print("Available categories:")

    if isinstance(index, dict):
        for code, position in sorted(
            index.items(), key=lambda x: x[1]
        ):
            print(f"  {code}: {labels.get(code, '')}")

    elif isinstance(index, list):
        for position, code in enumerate(index):
            print(f"  {code}: {labels.get(code, '')}")

print("\nOBSERVATION DATA")
values = data.get("value", {})

print("Value type:", type(values).__name__)
print("Number of observations:", len(values))

if isinstance(values, dict):
    print("First observations:", list(values.items())[:20])

elif isinstance(values, list):
    print("First observations:", values[:20])

print("\nSTATUS DATA")
status = data.get("status", {})
print("Status type:", type(status).__name__)
print("Status entries:", len(status))

if isinstance(status, dict):
    print("First status entries:", list(status.items())[:20])

elif isinstance(status, list):
    print("First status entries:", status[:20])

print("\nRESPONSE DIMENSION CHECKS")

for dimension_id in ["freq", "unit", "sector", "na_item", "geo", "time"]:

    if dimension_id not in data.get("dimension", {}):
        print(f"{dimension_id}: NOT PRESENT")
        continue

    category = data["dimension"][dimension_id]["category"]
    index = category.get("index", {})

    if isinstance(index, dict):
        codes = list(index.keys())
    else:
        codes = list(index)

    print(f"{dimension_id}: {codes}")

print("\nDIAGNOSTIC COMPLETE")
