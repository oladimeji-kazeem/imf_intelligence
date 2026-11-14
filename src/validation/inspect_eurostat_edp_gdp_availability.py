
import json
from pathlib import Path

FILES = {
    "debt": Path("data/raw/eurostat_debt_probe_20261114T074053Z.json"),
    "sfa": Path("data/raw/eurostat_sfa_probe_20261114T074434Z.json"),
}

for name, filepath in FILES.items():

    print("\n" + "=" * 75)
    print(f"{name.upper()} DATASET DIMENSION AUDIT")
    print("=" * 75)

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Dataset:", data.get("label"))
    print("Last updated:", data.get("updated"))
    print("Dimensions:", data.get("id"))
    print("Dimension sizes:", data.get("size"))

    for dimension_id in data["id"]:

        dimension = data["dimension"][dimension_id]

        print(f"\nDimension: {dimension_id}")
        print("Label:", dimension.get("label"))

        index = dimension.get("category", {}).get("index", {})

        if dimension_id.lower() in [
            "na_item", "unit", "geo", "time"
        ]:
            labels = dimension.get("category", {}).get("label", {})

            for code, position in sorted(
                index.items(), key=lambda x: x[1]
            ):
                print(
                    f"  {code}: {labels.get(code, 'No label')}"
                )
