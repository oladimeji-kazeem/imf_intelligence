from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    PROCESSED /
    "greek_psi_source_evidence_register.csv"
)

OUTPUT_REPORT = (
    REPORTS /
    "greek_psi_source_evidence_register.txt"
)


EVENT_IDS = [
    "EU_SOV_GRC_PSI_2011_2012_BONDS_GUARANTEED_BONDS",
    "EU_SOV_GRC_PSI_2011_2012_BONDS_GOVERNMENT_BONDS",
]

EPISODE_ID = "EU_SOV_GRC_PSI_2011_2012"


# Each source is assigned only to the claims it actually supports.
SOURCES = [

    {
        "evidence_id": "GRC_PSI_SRC_001",
        "episode_id": EPISODE_ID,
        "event_ids": ";".join(EVENT_IDS),

        "source_organisation": "International Monetary Fund",
        "source_title": (
            "Issues in Restructuring of Sovereign Domestic Debt "
            "— Background Paper"
        ),
        "publication_date": "2021-12-01",
        "source_url": (
            "https://www.imf.org/-/media/files/publications/"
            "pp/2021/english/ppea2021071.pdf"
        ),

        "evidence_dimension": "SOVEREIGN_RESTRUCTURING",

        "source_finding": (
            "The Greek debt exchange was completed on 12 March 2012. "
            "The IMF reports a 53.5 percent nominal face-value reduction."
        ),

        "payment_default_finding": (
            "The IMF describes this debt exchange as pre-emptive, "
            "with no payments missed during the exchange."
        ),

        "restructuring_finding": (
            "Sovereign debt restructuring is established for the "
            "March 2012 Greek PSI transaction."
        ),

        "cds_credit_event_finding": "NOT_ASSESSED_BY_THIS_SOURCE",

        "evidence_assessment": "IMF_DOCUMENTED",

        "limitation": (
            "The finding concerns the PSI exchange, not every Greek "
            "sovereign obligation or every date in the Greek debt crisis."
        ),
    },

    {
        "evidence_id": "GRC_PSI_SRC_002",
        "episode_id": EPISODE_ID,
        "event_ids": ";".join(EVENT_IDS),

        "source_organisation": (
            "International Swaps and Derivatives Association"
        ),
        "source_title": (
            "Restructuring Credit Event Has Occurred with Respect "
            "to The Hellenic Republic"
        ),
        "publication_date": "2012-03-09",
        "source_url": (
            "https://www.isda.org/2012/03/09/"
            "isda-emea-determinations-committee-restructuring-credit-event-has-occurred-with-respect-to-the-hellenic-republic/"
        ),

        "evidence_dimension": "CDS_CREDIT_EVENT",

        "source_finding": (
            "ISDA's EMEA Determinations Committee unanimously "
            "determined that a Restructuring Credit Event had occurred "
            "under the applicable CDS definitions."
        ),

        "payment_default_finding": "NOT_ESTABLISHED_BY_THIS_DETERMINATION",

        "restructuring_finding": (
            "The determination followed the exercise of collective "
            "action clauses that reduced affected bondholders' "
            "rights to receive payments."
        ),

        "cds_credit_event_finding": (
            "RESTRUCTURING_CREDIT_EVENT_CONFIRMED_BY_ISDA"
        ),

        "evidence_assessment": "PRIMARY_DETERMINATION",

        "limitation": (
            "A CDS restructuring credit event is a contractual "
            "classification under derivatives definitions. "
            "It must not automatically be equated with a missed-payment default."
        ),
    },

    {
        "evidence_id": "GRC_PSI_SRC_003",
        "episode_id": EPISODE_ID,
        "event_ids": ";".join(EVENT_IDS),

        "source_organisation": "International Monetary Fund",
        "source_title": (
            "Sovereign Debt Restructuring — Recent Developments "
            "and Implications for the Fund's Legal and Policy Framework"
        ),
        "publication_date": "2013-04-26",
        "source_url": (
            "https://www.imf.org/external/np/pp/eng/2013/042613.pdf"
        ),

        "evidence_dimension": "CDS_AND_RESTRUCTURING_FRAMEWORK",

        "source_finding": (
            "The IMF explains that restructuring can qualify as a "
            "CDS credit event even when the debt exchange occurs "
            "before a payment default."
        ),

        "payment_default_finding": (
            "A pre-default debt exchange can trigger a CDS credit event."
        ),

        "restructuring_finding": (
            "The Greek restructuring involved collective action "
            "procedures that bound affected creditors."
        ),

        "cds_credit_event_finding": (
            "The Greek CDS was triggered following activation "
            "of the collective action procedure."
        ),

        "evidence_assessment": "IMF_POLICY_ANALYSIS",

        "limitation": (
            "This is an explanatory policy paper, not the ISDA "
            "Determinations Committee's primary decision."
        ),
    },
]


def main():

    df = pd.DataFrame(SOURCES)

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    required_columns = [
        "evidence_id",
        "episode_id",
        "event_ids",
        "source_organisation",
        "source_title",
        "publication_date",
        "source_url",
        "evidence_dimension",
        "source_finding",
        "payment_default_finding",
        "restructuring_finding",
        "cds_credit_event_finding",
        "evidence_assessment",
        "limitation",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["evidence_id"].duplicated().any():
        raise ValueError("Duplicate evidence IDs found.")

    if df[required_columns].isna().any().any():
        raise ValueError("Missing required evidence values.")

    if not df["episode_id"].eq(EPISODE_ID).all():
        raise ValueError("Unexpected episode identifier.")

    if not df["event_ids"].eq(";".join(EVENT_IDS)).all():
        raise ValueError("Unexpected event linkage.")

    # --------------------------------------------------------
    # Save register
    # --------------------------------------------------------

    df.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as report:

        report.write("=" * 76 + "\n")
        report.write("GREEK PSI SOURCE-EVIDENCE REGISTER\n")
        report.write("=" * 76 + "\n\n")

        report.write(f"Canonical episode: {EPISODE_ID}\n")
        report.write(f"Linked event records: {len(EVENT_IDS)}\n")
        report.write(f"Evidence sources: {len(df)}\n\n")

        report.write("EVIDENCE DIMENSIONS\n")
        report.write("-" * 76 + "\n")

        for _, row in df.iterrows():

            report.write(
                f"\n{row['evidence_id']}\n"
            )

            report.write(
                f"Organisation: {row['source_organisation']}\n"
            )

            report.write(
                f"Title: {row['source_title']}\n"
            )

            report.write(
                f"Dimension: {row['evidence_dimension']}\n"
            )

            report.write(
                f"Assessment: {row['evidence_assessment']}\n"
            )

            report.write(
                f"Finding: {row['source_finding']}\n"
            )

            report.write(
                f"Limitation: {row['limitation']}\n"
            )

        report.write("\n" + "=" * 76 + "\n")
        report.write("EPISODE-LEVEL INTERPRETATION\n")
        report.write("=" * 76 + "\n")

        report.write(
            "Sovereign restructuring: ESTABLISHED\n"
        )

        report.write(
            "Payment default during the PSI exchange: "
            "NOT ESTABLISHED; IMF describes the exchange as pre-emptive.\n"
        )

        report.write(
            "CDS restructuring credit event: "
            "CONFIRMED BY ISDA.\n"
        )

        report.write(
            "General sovereign default status for all Greek obligations: "
            "NOT DETERMINED BY THIS REGISTER.\n"
        )

        report.write(
            "\nNo composite default label or severity score is assigned.\n"
        )

    print("=" * 76)
    print("GREEK PSI SOURCE-EVIDENCE REGISTER")
    print("=" * 76)

    print(f"Canonical episode: {EPISODE_ID}")
    print(f"Linked event records: {len(EVENT_IDS)}")
    print(f"Evidence sources: {len(df)}")

    print("\nEVIDENCE DIMENSIONS")

    print(
        df[
            [
                "evidence_id",
                "evidence_dimension",
                "evidence_assessment",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUTS")
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)

    print("\nSOURCE-EVIDENCE REGISTER COMPLETED.")


if __name__ == "__main__":
    main()