from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import load_workbook

ROOT = Path(".")
SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"

OUTPUT = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

wb = load_workbook(SOURCE, data_only=True, read_only=True)
ws = wb["2025 in Progress"]

# Column E onwards contains annual observations.
records = []

for col in range(5, ws.max_column + 1):
    year = ws.cell(1, col).value

    if year is None:
        continue

    records.append({
        "year": str(year),
        "eme": ws.cell(25, col).value,
        "hipc": ws.cell(26, col).value,
        "other_developing": ws.cell(27, col).value,
        "exae": ws.cell(28, col).value,
        "em_developing_gdp": ws.cell(61, col).value,
        "world_product": ws.cell(62, col).value,
    })

wb.close()

df = pd.DataFrame(records)

numeric_cols = [
    "eme", "hipc", "other_developing", "exae",
    "em_developing_gdp", "world_product"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# EXAE arithmetic reconciliation.
df["exae_calculated"] = (
    df["eme"] + df["hipc"] + df["other_developing"]
)

df["exae_difference"] = df["exae"] - df["exae_calculated"]

df["exae_reconciliation_status"] = np.where(
    df[["eme", "hipc", "other_developing", "exae"]]
    .notna().all(axis=1),
    np.where(
        df["exae_difference"].abs() <= 0.0051,
        "RECONCILED",
        "NOT_RECONCILED"
    ),
    "INCOMPLETE"
)

# GDP composition ratio.
df["em_developing_share_world_gdp_pct"] = (
    df["em_developing_gdp"] / df["world_product"] * 100
)

# Save year-level reconciliation.
df.to_csv(
    OUTPUT / "boc_boe_group_gdp_relationships.csv",
    index=False
)

valid_exae = df[
    df["exae_reconciliation_status"] != "INCOMPLETE"
]

valid_gdp = df[
    df["em_developing_share_world_gdp_pct"].notna()
]

lines = []
lines.append("=" * 80)
lines.append("BOC-BOE GROUP AND GDP RELATIONSHIP AUDIT")
lines.append("=" * 80)

lines.append(f"Annual observations: {len(df)}")

lines.append("\nEXAE COMPOSITION RECONCILIATION")
lines.append("-" * 80)
lines.append(f"Comparable observations: {len(valid_exae)}")
lines.append(
    f"Reconciled: "
    f"{(valid_exae['exae_reconciliation_status'] == 'RECONCILED').sum()}"
)
lines.append(
    f"Not reconciled: "
    f"{(valid_exae['exae_reconciliation_status'] == 'NOT_RECONCILED').sum()}"
)
lines.append(
    f"Incomplete: "
    f"{(df['exae_reconciliation_status'] == 'INCOMPLETE').sum()}"
)

if len(valid_exae):
    lines.append(
        "Maximum absolute difference: "
        f"{valid_exae['exae_difference'].abs().max():,.10f}"
    )

lines.append("\nGDP COMPOSITION")
lines.append("-" * 80)
lines.append(f"Available observations: {len(valid_gdp)}")

if len(valid_gdp):
    lines.append(
        f"First available year: {valid_gdp.iloc[0]['year']}"
    )
    lines.append(
        f"Last available year: {valid_gdp.iloc[-1]['year']}"
    )
    lines.append(
        "First available share: "
        f"{valid_gdp.iloc[0]['em_developing_share_world_gdp_pct']:.4f}%"
    )
    lines.append(
        "Latest share: "
        f"{valid_gdp.iloc[-1]['em_developing_share_world_gdp_pct']:.4f}%"
    )
    lines.append(
        "Mean share: "
        f"{valid_gdp['em_developing_share_world_gdp_pct'].mean():.4f}%"
    )

lines.append("\nINTERPRETATION")
lines.append("-" * 80)
lines.append(
    "Arithmetic reconciliation establishes numerical composition, "
    "not the full methodology or country membership."
)
lines.append(
    "GDP share is a derived relative-composition indicator, "
    "not a sovereign default-risk measure."
)
lines.append(
    "Nominal GDP levels remain exposed to inflation and exchange-rate effects."
)

report = "\n".join(lines)

(REPORTS / "boc_boe_group_gdp_relationship_audit.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print("\nAUDIT COMPLETED.")