
from pathlib import Path
import pandas as pd

ROOT = Path(".")
SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"

OUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    SOURCE,
    sheet_name="Debt_2025",
    header=64
)

required = [
    "DEBT_COUNTRY",
    "DEBT_COUNTRY_GROUP",
    "DEBT_YEAR",
    "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
    "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
    "DEBT_DEFRATE_SOVEREIGNS_2025",
    "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
    "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing source columns: {missing}")

world = df[
    (df["DEBT_COUNTRY"] == "World") &
    (df["DEBT_COUNTRY_GROUP"] == "World")
].copy()

world["DEBT_YEAR"] = pd.to_numeric(
    world["DEBT_YEAR"], errors="coerce"
)

for col in required[3:]:
    world[col] = pd.to_numeric(world[col], errors="coerce")

# General sovereign default count / rate
world["recalculated_default_rate"] = (
    world["DEBT_TOTAL_DEF_SOVEREIGNS_2025"] /
    world["DEBT_TOTAL_NUMB_SOVEREIGNS_2025"] * 100
)

world["default_rate_abs_difference"] = (
    world["recalculated_default_rate"] -
    world["DEBT_DEFRATE_SOVEREIGNS_2025"]
).abs()

# Arrears count / rate
world["recalculated_arrears_rate"] = (
    world["DEBT_ARREARS_TOTAL_SOVEREIGNS_2025"] /
    world["DEBT_TOTAL_NUMB_SOVEREIGNS_2025"] * 100
)

world["arrears_rate_abs_difference"] = (
    world["recalculated_arrears_rate"] -
    world["DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025"]
).abs()

TOLERANCE = 0.0051

world["default_rate_match"] = (
    world["default_rate_abs_difference"] <= TOLERANCE
)

world["arrears_rate_match"] = (
    world["arrears_rate_abs_difference"] <= TOLERANCE
)

default_rows = world.dropna(subset=[
    "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
    "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
    "DEBT_DEFRATE_SOVEREIGNS_2025",
])

arrears_rows = world.dropna(subset=[
    "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
    "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
    "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
])

results = [
    {
        "relationship": "TOTAL_DEFAULT_COUNT_TO_DEFAULT_RATE",
        "numerator": "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
        "denominator": "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
        "derived_indicator": "DEBT_DEFRATE_SOVEREIGNS_2025",
        "valid_observations": len(default_rows),
        "matches": int(default_rows["default_rate_match"].sum()),
        "non_matches": int((~default_rows["default_rate_match"]).sum()),
        "maximum_absolute_difference": (
            default_rows["default_rate_abs_difference"].max()
        ),
        "status": (
            "FORMULA_RECONCILED"
            if default_rows["default_rate_match"].all()
            else "REVIEW_REQUIRED"
        ),
    },
    {
        "relationship": "ARREARS_COUNT_TO_ARREARS_RATE",
        "numerator": "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
        "denominator": "DEBT_TOTAL_NUMB_SOVEREIGNS_2025",
        "derived_indicator": "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
        "valid_observations": len(arrears_rows),
        "matches": int(arrears_rows["arrears_rate_match"].sum()),
        "non_matches": int((~arrears_rows["arrears_rate_match"]).sum()),
        "maximum_absolute_difference": (
            arrears_rows["arrears_rate_abs_difference"].max()
        ),
        "status": (
            "FORMULA_RECONCILED"
            if arrears_rows["arrears_rate_match"].all()
            else "REVIEW_REQUIRED"
        ),
    },
]

summary = pd.DataFrame(results)

world.to_csv(
    OUT / "boc_boe_aggregate_formula_reconciliation.csv",
    index=False
)

summary.to_csv(
    OUT / "boc_boe_aggregate_formula_relationship_register.csv",
    index=False
)

report = [
    "=" * 78,
    "BoC-BoE AGGREGATE FORMULA RELATIONSHIP AUDIT",
    "=" * 78,
    "",
    f"World observations: {len(world)}",
    f"Year range: {int(world['DEBT_YEAR'].min())}–"
    f"{int(world['DEBT_YEAR'].max())}",
    f"Rounding tolerance: {TOLERANCE} percentage points",
    "",
    "FORMULA RECONCILIATION",
    "-" * 78,
]

for item in results:
    report.extend([
        f"Relationship: {item['relationship']}",
        f"Numerator: {item['numerator']}",
        f"Denominator: {item['denominator']}",
        f"Derived indicator: {item['derived_indicator']}",
        f"Valid observations: {item['valid_observations']}",
        f"Matches: {item['matches']}",
        f"Non-matches: {item['non_matches']}",
        f"Maximum absolute difference: "
        f"{item['maximum_absolute_difference']:.8f}",
        f"Status: {item['status']}",
        "",
    ])

report.extend([
    "INTERPRETATION",
    "-" * 78,
    "- The two rate indicators reconcile empirically to their count "
    "and denominator fields.",
    "- The rates should not be treated as independent signals from "
    "their corresponding counts and common denominator.",
    "- The count indicators remain distinct measures.",
    "- Formula reconciliation does not establish the full methodological "
    "definition of either count.",
    "- No variables are deleted or overwritten by this audit.",
])

report_path = REPORTS / "boc_boe_aggregate_formula_relationship_audit.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nOUTPUTS")
print(OUT / "boc_boe_aggregate_formula_reconciliation.csv")
print(OUT / "boc_boe_aggregate_formula_relationship_register.csv")
print(report_path)

if not summary["status"].eq("FORMULA_RECONCILED").all():
    raise SystemExit("AUDIT REVIEW REQUIRED: one or more formula checks failed.")

print("\nAUDIT COMPLETED SUCCESSFULLY.")
