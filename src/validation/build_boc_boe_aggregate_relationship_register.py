
from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

relationships = [
    {
        "relationship_id": "REL_001",
        "source_variables": "DEBT_TOTAL_DEF_SOVEREIGNS_2025",
        "target_variable": "DEBT_DEFRATE_SOVEREIGNS_2025",
        "relationship_type": "COUNT_DERIVED_RATE",
        "formula": "Defaulted sovereign count / total sovereign count * 100",
        "audit_status": "FORMULA_RECONCILED",
        "observations_tested": 65,
        "modelling_implication": "Avoid treating count and derived rate as independent predictors",
    },
    {
        "relationship_id": "REL_002",
        "source_variables": "DEBT_ARREARS_TOTAL_SOVEREIGNS_2025",
        "target_variable": "DEBT_ARREARS_DEFRATE_SOVEREIGNS_2025",
        "relationship_type": "COUNT_DERIVED_RATE",
        "formula": "Sovereigns with arrears / total sovereign count * 100",
        "audit_status": "FORMULA_RECONCILED",
        "observations_tested": 46,
        "modelling_implication": "Avoid treating count and derived rate as independent predictors",
    },
    {
        "relationship_id": "REL_003",
        "source_variables": (
            "DEBT_EME_2025 + DEBT_HIPC_2025 + "
            "DEBT_OTHER_2025"
        ),
        "target_variable": "DEBT_EXAE_2025",
        "relationship_type": "EXACT_COMPONENT_SUM",
        "formula": "EME + HIPC + Other developing",
        "audit_status": "FORMULA_RECONCILED",
        "observations_tested": 65,
        "modelling_implication": "Do not include aggregate and all components together in a standard linear model",
    },
    {
        "relationship_id": "REL_004",
        "source_variables": (
            "DEBT_GROSS_EME_OTHER_GDP_2025 / "
            "DEBT_GROSS_WORLD_PRODUCT_2025"
        ),
        "target_variable": "EM_DEVELOPING_GDP_SHARE",
        "relationship_type": "DERIVED_RATIO",
        "formula": "EM and developing GDP / Gross World Product * 100",
        "audit_status": "CALCULATED_45_OBSERVATIONS",
        "observations_tested": 45,
        "modelling_implication": "Use as relative composition feature; not a direct risk measure",
    },
    {
        "relationship_id": "REL_005",
        "source_variables": (
            "DEBT_GROSS_EME_OTHER_GDP_2025; "
            "DEBT_GROSS_WORLD_PRODUCT_2025"
        ),
        "target_variable": "",
        "relationship_type": "NESTED_NOMINAL_GDP_AGGREGATES",
        "formula": "EM and developing GDP is a subset of world product",
        "audit_status": "SEMANTICALLY_SUPPORTED_NUMERICAL_NESTING",
        "observations_tested": 45,
        "modelling_implication": "Avoid interpreting both nominal levels as independent risk signals",
    },
]

df = pd.DataFrame(relationships)

output = DATA / "boc_boe_aggregate_relationship_register.csv"
df.to_csv(output, index=False)

lines = [
    "=" * 80,
    "BOC-BOE AGGREGATE RELATIONSHIP REGISTER",
    "=" * 80,
    f"Relationships registered: {len(df)}",
    "",
    "RELATIONSHIP COUNTS",
    "-" * 80,
]

for relationship_type, count in df["relationship_type"].value_counts().items():
    lines.append(f"{relationship_type}: {count}")

lines += [
    "",
    "MODELLING GOVERNANCE",
    "-" * 80,
    "1. Preserve original source observations.",
    "2. Keep derived variables separately identifiable and traceable.",
    "3. Do not treat formula-derived measures as independent signals.",
    "4. Do not infer causation from correlation or arithmetic relationships.",
    "5. Retain provisional-year status and source coverage metadata.",
    "",
    f"Output: {output}",
    "REGISTER BUILD COMPLETED.",
]

report = "\n".join(lines)

(REPORTS / "boc_boe_aggregate_relationship_register.txt").write_text(
    report, encoding="utf-8"
)

print(report)
