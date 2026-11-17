
from pathlib import Path
import pandas as pd

DATA = Path("data/processed")
REPORTS = Path("outputs/reports")

DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Official country-level scores from BoC Technical Report No. 125.
# Last updated 7 October 2025.
records = [
    ("GRC", "Greece", 1),
    ("HUN", "Hungary", 2),
    ("IRL", "Ireland", 1),
    ("POL", "Poland", 2),
    ("PRT", "Portugal", 1),
    ("ROU", "Romania", 2),
    ("GBR", "United Kingdom", 1),
]

df = pd.DataFrame(
    records,
    columns=["iso3", "country", "official_reliability_score"]
)

df["reliability_category"] = df["official_reliability_score"].map({
    1: "HIGHEST_RELIABILITY",
    2: "RELATIVELY_HIGH_RELIABILITY",
    3: "MODERATE_RELIABILITY",
    4: "LOWEST_RELIABILITY",
})

df["score_direction"] = "LOWER_IS_BETTER"
df["score_scope"] = "COUNTRY_LEVEL_OVERALL"
df["score_is_probability"] = False
df["score_is_annual"] = False
df["source_document"] = "BoC Technical Report No. 125"
df["source_updated"] = "2025-10-07"
df["source_url"] = (
    "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr125.pdf"
)

df["methodology_dimensions"] = (
    "Default-year identification; debt-type coverage; "
    "restructuring characteristics; source consistency"
)

df["governance_note"] = (
    "Subjective ordinal country-level assessment; "
    "not an annual observation or component-specific quality score"
)

csv_path = DATA / "boc_boe_europe_peer_reliability.csv"
report_path = REPORTS / "boc_boe_europe_peer_reliability_audit.txt"

df.to_csv(csv_path, index=False)

lines = [
    "=" * 84,
    "BOC-BOE EUROPEAN PEER RELIABILITY AUDIT",
    "=" * 84,
    f"Countries assessed: {len(df)}",
    f"Mean official score: {df['official_reliability_score'].mean():.2f}",
    f"Minimum score: {df['official_reliability_score'].min()}",
    f"Maximum score: {df['official_reliability_score'].max()}",
    "",
    "SCORE DISTRIBUTION",
    "-" * 84,
]

for score, count in (
    df["official_reliability_score"].value_counts().sort_index().items()
):
    lines.append(f"Score {score}: {count} countries")

lines.extend([
    "",
    "COUNTRY SCORES",
    "-" * 84,
])

for _, row in df.iterrows():
    lines.append(
        f"{row['iso3']} | {row['country']} | "
        f"{row['official_reliability_score']} | "
        f"{row['reliability_category']}"
    )

lines.extend([
    "",
    "INTERPRETATION AND GOVERNANCE",
    "-" * 84,
    "1. Lower scores indicate higher reliability.",
    "2. Scores are subjective ordinal country-level assessments.",
    "3. Scores are not probabilities or annual observations.",
    "4. Scores do not establish completeness of every creditor series.",
    "5. Do not use the score as a direct multiplier on default amounts.",
    "6. Preserve the official score separately from internally calculated",
    "   coverage and missingness indicators.",
    "",
    "SOURCE",
    "-" * 84,
    "BoC Technical Report No. 125, updated 7 October 2025.",
    "https://www.bankofcanada.ca/wp-content/uploads/2023/07/tr125.pdf",
    "",
    "OUTPUT",
    "-" * 84,
    str(csv_path),
    str(report_path),
    "",
    "PEER RELIABILITY AUDIT COMPLETED.",
])

report = "\n".join(lines)
report_path.write_text(report, encoding="utf-8")
print(report)
