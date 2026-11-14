
import json
from pathlib import Path
from collections import Counter

FILE = Path(
    "data/raw/eurostat_interest_probe_20261114T073451Z.json"
)

with FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

dimension_ids = data["id"]
sizes = data["size"]

unit_dimension = data["dimension"]["unit"]
unit_index = unit_dimension["category"]["index"]
unit_labels = unit_dimension["category"]["label"]

# Decode each observation's position in the JSON-stat array.
counts = Counter()

for key in data.get("value", {}):
    position = int(key)
    coordinates = {}

    for dimension_id, size in zip(
        reversed(dimension_ids),
        reversed(sizes)
    ):
        coordinates[dimension_id] = position % size
        position //= size

    unit_position = coordinates["unit"]

    unit_code = next(
        code for code, index in unit_index.items()
        if index == unit_position
    )

    counts[unit_code] += 1

print("EUROSTAT INTEREST DATA AUDIT")
print("=" * 55)

for unit_code in unit_index:
    count = counts[unit_code]
    print(
        f"{unit_code:10} "
        f"{unit_labels[unit_code]:45} "
        f"{count:3}/95"
    )

print("\nTOTAL OBSERVATIONS:", sum(counts.values()))

if all(counts[code] == 95 for code in unit_index):
    print("RESULT: ALL THREE UNITS HAVE COMPLETE COVERAGE.")
else:
    print("RESULT: INVESTIGATE UNIT-LEVEL MISSING OBSERVATIONS.")
