
from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"

DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

records = [
    {
        "evidence_id": "METH_001",
        "topic": "Database purpose and scope",
        "finding": "Estimates stocks of sovereign government obligations in default, including bonds, marketable securities, bank loans and official loans.",
        "status": "DOCUMENTED",
        "source_version": "2025 release note",
        "source_section": "Introduction",
        "source_url": "https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/",
        "limitation": "The database is not simply an event register."
    },
    {
        "evidence_id": "METH_002",
        "topic": "Operational default definition",
        "finding": "Debt distress/effective default includes interruption of scheduled debt service or sovereign renegotiation of existing contractual terms.",
        "status": "DOCUMENTED",
        "source_version": "2025 release note",
        "source_section": "Introduction",
        "source_url": "https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/",
        "limitation": "Broader than a missed-payment-only definition."
    },
    {
        "evidence_id": "METH_003",
        "topic": "Restructuring treatment",
        "finding": "Restructuring can involve principal write-downs, interest-rate reductions or maturity extensions; restructured debt is subsequently reclassified as performing.",
        "status": "DOCUMENTED",
        "source_version": "2025 release note",
        "source_section": "Introduction",
        "source_url": "https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/",
        "limitation": "A restructuring episode and a payment default are not interchangeable concepts."
    },
    {
        "evidence_id": "METH_004",
        "topic": "Annual observation semantics",
        "finding": "Annual totals include sovereigns in default at any point during the year and the amounts of debt affected.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Section 3",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "Do not count each positive country-year observation as a new default event."
    },
    {
        "evidence_id": "METH_005",
        "topic": "Valuation basis",
        "finding": "Country and global default amounts are expressed in nominal US dollars.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Section 3",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "Nominal values are affected by inflation and exchange-rate movements."
    },
    {
        "evidence_id": "METH_006",
        "topic": "Creditor coverage",
        "finding": "Categories include IMF, IBRD, IDA, IADB, Paris Club, China, other official creditors, foreign-currency bank loans, foreign-currency bonds, other private creditors, local-currency debt and domestic arrears.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Section 3",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "Not every category has complete country-year coverage."
    },
    {
        "evidence_id": "METH_007",
        "topic": "Domestic arrears definition",
        "finding": "Domestic arrears include overdue government payments such as salaries, pensions, tax refunds, services, capital outlays and sovereign local-law bonds.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Section 2 and domestic arrears subsection",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "The legal threshold for arrears varies, commonly 30, 60 or 90 days."
    },
    {
        "evidence_id": "METH_008",
        "topic": "Domestic arrears estimation",
        "finding": "Domestic arrears are often reported as flows; the database uses available flow and stock information to derive stock estimates.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Domestic arrears subsection",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "The derived stock series is not equivalent to directly observed annual arrears stocks."
    },
    {
        "evidence_id": "METH_009",
        "topic": "Source estimation and completeness",
        "finding": "The database combines IMF, World Bank, IADB, Paris Club, research datasets, national sources and other information; some values are estimated.",
        "status": "DOCUMENTED",
        "source_version": "2025 release note and updated methodology",
        "source_section": "Data sources and estimation",
        "source_url": "https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/",
        "limitation": "The authors acknowledge possible omissions, contradictory sources and future revisions."
    },
    {
        "evidence_id": "METH_010",
        "topic": "Debtor-group classification",
        "finding": "The current methodology describes advanced economies using IMF definitions, HIPC using IMF definitions, emerging/frontier markets using the J.P. Morgan EMBI Global Diversified Index, and other developing countries as the remaining sovereigns with default data.",
        "status": "PARTIALLY_DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Summary data",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "Exact historical membership and applicability to the 2025 workbook remain to be verified."
    },
    {
        "evidence_id": "METH_011",
        "topic": "GDP aggregate definition",
        "finding": "The current methodology identifies the emerging-market and developing-country GDP aggregate as excluding China, alongside world GDP.",
        "status": "PARTIALLY_DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Summary data",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "Confirm that the same definition and country coverage apply to the 2025 workbook's historical series."
    },
    {
        "evidence_id": "METH_012",
        "topic": "Debtor-group arithmetic",
        "finding": "EXAE equals EME plus HIPC plus Other Developing in all 65 years tested in the workbook.",
        "status": "EMPIRICALLY_RECONCILED",
        "source_version": "User's 2025 workbook audit",
        "source_section": "Rows 25-28 of 2025 in Progress",
        "source_url": "",
        "limitation": "Arithmetic identity does not independently establish country membership or classification stability."
    },
    {
        "evidence_id": "METH_013",
        "topic": "GDP composition ratio",
        "finding": "EM and developing GDP divided by Gross World Product produces a derived share for 1980-2024p.",
        "status": "EMPIRICALLY_CALCULATED",
        "source_version": "User's 2025 workbook audit",
        "source_section": "Rows 61-62 of 2025 in Progress",
        "source_url": "",
        "limitation": "Nominal GDP share is not a direct sovereign default-risk measure."
    },
    {
        "evidence_id": "METH_014",
        "topic": "Provisional observation",
        "finding": "The workbook labels its final annual observation 2024p.",
        "status": "OBSERVED_IN_WORKBOOK",
        "source_version": "2025 workbook",
        "source_section": "Year header",
        "source_url": "",
        "limitation": "Preserve provisional status; do not silently treat 2024p as final."
    },
    {
        "evidence_id": "METH_015",
        "topic": "Methodology vintage alignment",
        "finding": "The 2025 release note describes data through 2024, while the currently linked methodology document was updated in October 2026 and describes data through 2025.",
        "status": "UNRESOLVED",
        "source_version": "2025 workbook versus 2026 methodology",
        "source_section": "Document version comparison",
        "source_url": "https://www.bankofcanada.ca/2025/10/staff-analytical-note-2025-24/",
        "limitation": "Do not apply 2026-only revisions retrospectively without checking the 2025 methodology version."
    },
    {
        "evidence_id": "METH_016",
        "topic": "Country-level reliability",
        "finding": "The methodology describes a country-level reliability framework based on default years, debt types, restructuring characteristics and source consistency.",
        "status": "DOCUMENTED",
        "source_version": "Methodology updated 2026-10-06",
        "source_section": "Section 5",
        "source_url": "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr124.pdf",
        "limitation": "The actual country-level scores in the 2025 workbook still need to be extracted and assessed."
    },
]

df = pd.DataFrame(records)

# Guard against accidental duplicate evidence identifiers.
assert df["evidence_id"].is_unique

# Preserve a consistent column order.
columns = [
    "evidence_id",
    "topic",
    "finding",
    "status",
    "source_version",
    "source_section",
    "source_url",
    "limitation",
]
df = df[columns]

csv_path = DATA / "boc_boe_methodology_evidence_register.csv"
txt_path = REPORTS / "boc_boe_methodology_evidence_register.txt"

df.to_csv(csv_path, index=False)

lines = [
    "=" * 84,
    "BOC-BOE METHODOLOGY EVIDENCE REGISTER",
    "=" * 84,
    f"Evidence records: {len(df)}",
    "",
    "STATUS DISTRIBUTION",
    "-" * 84,
]

for status, count in df["status"].value_counts().items():
    lines.append(f"{status}: {count}")

lines.extend([
    "",
    "KEY VERSION-CONTROL WARNING",
    "-" * 84,
    "The 2025 workbook and 2026-updated methodology are not automatically",
    "assumed to be methodologically identical.",
    "",
    "UNRESOLVED ITEMS",
    "-" * 84,
])

for _, row in df[df["status"] == "UNRESOLVED"].iterrows():
    lines.append(f"{row['evidence_id']} | {row['topic']}")
    lines.append(f"  {row['limitation']}")

lines.extend([
    "",
    "OUTPUT FILES",
    "-" * 84,
    str(csv_path),
    str(txt_path),
    "",
    "REGISTER BUILD COMPLETED.",
])

report = "\n".join(lines)
txt_path.write_text(report, encoding="utf-8")

print(report)
