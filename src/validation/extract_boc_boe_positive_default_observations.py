
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/external/boc_boe_sovereign_default_database_2025.xlsx"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PEER_COUNTRIES = {
    "United Kingdom",
    "Greece",
    "Hungary",
    "Ireland",
    "Poland",
    "Portugal",
    "Romania",
}

COMPONENT_FIELDS = [
    "DEBT_IMF_2025",
    "DEBT_IBRD_2025",
    "DEBT_IDA_2025",
    "DEBT_IADB_2025",
    "DEBT_PARIS_CLUB_2025",
    "DEBT_CHINA_2025",
    "DEBT_OTHER_OFFICIAL_CREDITORS_2025",
    "DEBT_PRIVATE_CREDITORS_2025",
    "DEBT_FC_BANK_LOANS_2025",
    "DEBT_FC_BONDS_2025",
    "DEBT_LC_DEBT_2025",
    "DEBT_FISCAL_ARREARS_2025",
]

DETAIL_FILE = (
    OUTPUT_DIR /
    "boc_boe_europe_positive_default_observations.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "boc_boe_europe_positive_default_country_summary.csv"
)

REPORT_FILE = (
    REPORT_DIR /
    "boc_boe_europe_positive_default_observations.txt"
)


def main():

    print("Loading BoC–BoE 2025 database...")

    # Excel row 65 contains the observation table headers.
    df = pd.read_excel(
        SOURCE,
        sheet_name="Debt_2025",
        header=64
    )

    required = {
        "DEBT_COUNTRY",
        "DEBT_YEAR",
        "DEBT_TOTAL_2025",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    df["DEBT_COUNTRY"] = (
        df["DEBT_COUNTRY"]
        .astype("string")
        .str.strip()
    )

    df["DEBT_YEAR"] = pd.to_numeric(
        df["DEBT_YEAR"],
        errors="coerce"
    )

    df["DEBT_TOTAL_2025"] = pd.to_numeric(
        df["DEBT_TOTAL_2025"],
        errors="coerce"
    )

    available_components = [
        col for col in COMPONENT_FIELDS
        if col in df.columns
    ]

    for col in available_components:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Restrict analysis to the seven directly matched peers.
    peers = df[
        df["DEBT_COUNTRY"].isin(PEER_COUNTRIES)
    ].copy()

    represented = set(peers["DEBT_COUNTRY"].dropna())

    if represented != PEER_COUNTRIES:
        missing_countries = PEER_COUNTRIES - represented
        raise ValueError(
            f"Unexpected missing peer countries: "
            f"{sorted(missing_countries)}"
        )

    # Preserve all seven countries' annual observations.
    # Positive observations are extracted separately.
    positive = peers[
        peers["DEBT_TOTAL_2025"] > 0
    ].copy()

    if positive.empty:
        raise ValueError(
            "No positive debt-in-default observations found."
        )

    positive = positive.sort_values(
        ["DEBT_COUNTRY", "DEBT_YEAR"]
    )

    positive.to_csv(
        DETAIL_FILE,
        index=False
    )

    # Country-level profile.
    summary_rows = []

    for country, group in peers.groupby("DEBT_COUNTRY"):

        valid = group[
            group["DEBT_TOTAL_2025"].notna()
        ]

        positive_group = valid[
            valid["DEBT_TOTAL_2025"] > 0
        ]

        row = {
            "country": country,
            "annual_observation_rows": len(group),
            "valid_total_rows": len(valid),
            "missing_total_rows": int(
                group["DEBT_TOTAL_2025"].isna().sum()
            ),
            "positive_years": len(positive_group),
            "zero_years": int(
                (valid["DEBT_TOTAL_2025"] == 0).sum()
            ),
            "first_positive_year": (
                int(positive_group["DEBT_YEAR"].min())
                if not positive_group.empty else None
            ),
            "last_positive_year": (
                int(positive_group["DEBT_YEAR"].max())
                if not positive_group.empty else None
            ),
            "maximum_annual_debt_in_default": (
                positive_group["DEBT_TOTAL_2025"].max()
                if not positive_group.empty else None
            ),
        }

        for component in available_components:
            row[f"{component}_positive_years"] = int(
                (positive_group[component] > 0).sum()
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        "country"
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # Human-readable report.
    lines = [
        "BOC–BOE EUROPEAN POSITIVE DEFAULT OBSERVATIONS",
        "=" * 65,
        "",
        f"Peer countries assessed: {len(PEER_COUNTRIES)}",
        f"Annual observations across peers: {len(peers)}",
        f"Positive country-year observations: {len(positive)}",
        "",
        "COUNTRY PROFILE",
        "-" * 65,
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['country']} | "
            f"positive_years={row['positive_years']} | "
            f"zero_years={row['zero_years']} | "
            f"missing={row['missing_total_rows']} | "
            f"first={row['first_positive_year']} | "
            f"last={row['last_positive_year']} | "
            f"maximum={row['maximum_annual_debt_in_default']}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 65,
        "Positive observations represent annual debt-in-default stocks.",
        "They are not independent default events.",
        "Persistent exposures may appear in multiple consecutive years.",
        "Annual stock values must not be summed across years as unique",
        "defaulted debt or cumulative creditor losses.",
        "",
        "OUTPUT FILES",
        "-" * 65,
        str(DETAIL_FILE),
        str(SUMMARY_FILE),
        str(REPORT_FILE),
    ])

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
