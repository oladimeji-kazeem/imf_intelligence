
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "data/external/"
    "boc_boe_sovereign_default_database_2025.xlsx"
)

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

CATALOGUE_OUTPUT = (
    OUTPUT_DIR / "boc_boe_full_series_catalogue.csv"
)

PROFILE_OUTPUT = (
    OUTPUT_DIR / "boc_boe_full_series_profile.csv"
)

COUNTRY_OUTPUT = (
    OUTPUT_DIR / "boc_boe_full_series_country_coverage.csv"
)

REPORT_OUTPUT = (
    REPORT_DIR / "boc_boe_full_series_inventory_audit.txt"
)


def classify_series(series_id):
    name = series_id.upper()

    if "DEFRATE" in name:
        return "DEFAULT_RATE"

    if "ARREARS" in name:
        return "ARREARS"

    if "NUMB_SOVEREIGNS" in name or "DEF_SOVEREIGNS" in name:
        return "SOVEREIGN_COUNTS"

    if "GROSS_" in name:
        return "GROSS_DEBT_OR_MACRO_AGGREGATE"

    if any(x in name for x in [
        "IMF", "IBRD", "IDA", "IADB", "PARIS_CLUB",
        "CHINA", "OFFICIAL_CREDITORS", "PRIVATE_CREDITORS"
    ]):
        return "CREDITOR"

    if any(x in name for x in [
        "FC_BANK", "FC_BONDS", "LC_DEBT"
    ]):
        return "INSTRUMENT"

    if any(x in name for x in [
        "ADVANCED_ECON", "EME", "HIPC", "EXAE", "SSA"
    ]):
        return "COUNTRY_GROUP"

    if "TOTAL" in name:
        return "TOTAL_OR_AGGREGATE"

    return "OTHER"


def main():

    print("Reading workbook metadata...")

    workbook = load_workbook(
        SOURCE,
        read_only=True,
        data_only=True
    )

    ws = workbook["Debt_2025"]

    catalogue_rows = []

    # Metadata rows 12-62.
    for row_num in range(12, 63):

        series_id = ws.cell(row_num, 1).value
        label = ws.cell(row_num, 2).value
        description = ws.cell(row_num, 3).value

        if not series_id:
            continue

        catalogue_rows.append({
            "metadata_row": row_num,
            "series_id": str(series_id).strip(),
            "label": label,
            "description": description,
            "analytical_group": classify_series(str(series_id)),
        })

    workbook.close()

    catalogue = pd.DataFrame(catalogue_rows)

    print("Reading observation table...")

    df = pd.read_excel(
        SOURCE,
        sheet_name="Debt_2025",
        header=64
    )

    identifiers = {
        "k",
        "DEBT_COUNTRY",
        "DEBT_COUNTRY_GROUP",
        "DEBT_YEAR",
    }

    series_fields = [
        col for col in df.columns
        if col not in identifiers
    ]

    missing_catalogue_fields = (
        set(series_fields) - set(catalogue["series_id"])
    )

    if missing_catalogue_fields:
        raise ValueError(
            "Observation fields missing from metadata catalogue: "
            f"{sorted(missing_catalogue_fields)}"
        )

    df["DEBT_COUNTRY"] = (
        df["DEBT_COUNTRY"].astype("string").str.strip()
    )

    df["DEBT_YEAR"] = pd.to_numeric(
        df["DEBT_YEAR"], errors="coerce"
    )

    peers = df[
        df["DEBT_COUNTRY"].isin(PEER_COUNTRIES)
    ].copy()

    if set(peers["DEBT_COUNTRY"].dropna()) != PEER_COUNTRIES:
        raise ValueError(
            "The expected seven peer countries were not all found."
        )

    profile_rows = []
    country_rows = []

    for field in series_fields:

        values = pd.to_numeric(
            peers[field], errors="coerce"
        )

        profile_rows.append({
            "series_id": field,
            "analytical_group": classify_series(field),
            "observations": len(values),
            "available": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "availability_pct": round(
                values.notna().mean() * 100, 2
            ),
            "positive": int((values > 0).sum()),
            "zero": int((values == 0).sum()),
            "negative": int((values < 0).sum()),
            "minimum": (
                values.min() if values.notna().any() else None
            ),
            "maximum": (
                values.max() if values.notna().any() else None
            ),
        })

        for country, group in peers.groupby("DEBT_COUNTRY"):

            country_values = pd.to_numeric(
                group[field], errors="coerce"
            )

            available_years = group.loc[
                country_values.notna(), "DEBT_YEAR"
            ].dropna()

            country_rows.append({
                "country": country,
                "series_id": field,
                "observations": len(group),
                "available": int(country_values.notna().sum()),
                "missing": int(country_values.isna().sum()),
                "availability_pct": round(
                    country_values.notna().mean() * 100, 2
                ),
                "positive": int((country_values > 0).sum()),
                "zero": int((country_values == 0).sum()),
                "negative": int((country_values < 0).sum()),
                "first_available_year": (
                    int(available_years.min())
                    if not available_years.empty else None
                ),
                "last_available_year": (
                    int(available_years.max())
                    if not available_years.empty else None
                ),
            })

    profile = pd.DataFrame(profile_rows)
    country_coverage = pd.DataFrame(country_rows)

    catalogue.to_csv(CATALOGUE_OUTPUT, index=False)
    profile.to_csv(PROFILE_OUTPUT, index=False)
    country_coverage.to_csv(COUNTRY_OUTPUT, index=False)

    lines = [
        "BOC–BOE FULL SERIES INVENTORY AUDIT",
        "=" * 75,
        "",
        f"Metadata series identified: {len(catalogue)}",
        f"Observation variables assessed: {len(series_fields)}",
        f"Peer countries assessed: {peers['DEBT_COUNTRY'].nunique()}",
        f"Peer annual observations: {len(peers)}",
        "",
        "SERIES COVERAGE BY ANALYTICAL GROUP",
        "-" * 75,
    ]

    for group_name, group in profile.groupby("analytical_group"):

        lines.append(
            f"{group_name} | "
            f"series={len(group)} | "
            f"mean_availability="
            f"{group['availability_pct'].mean():.2f}%"
        )

    lines.extend([
        "",
        "SERIES-LEVEL AVAILABILITY",
        "-" * 75,
    ])

    for _, row in profile.sort_values(
        ["analytical_group", "series_id"]
    ).iterrows():

        lines.append(
            f"{row['analytical_group']} | "
            f"{row['series_id']} | "
            f"available={row['available']} | "
            f"missing={row['missing']} | "
            f"positive={row['positive']} | "
            f"zero={row['zero']} | "
            f"negative={row['negative']} | "
            f"availability={row['availability_pct']}%"
        )

    lines.extend([
        "",
        "HIGH-PRIORITY SERIES",
        "-" * 75,
    ])

    priority = profile[
        profile["analytical_group"].isin([
            "DEFAULT_RATE",
            "ARREARS",
            "SOVEREIGN_COUNTS",
        ])
    ]

    for _, row in priority.sort_values(
        ["analytical_group", "series_id"]
    ).iterrows():

        lines.append(
            f"{row['series_id']} | "
            f"available={row['available']}/"
            f"{row['observations']} | "
            f"availability={row['availability_pct']}% | "
            f"min={row['minimum']} | "
            f"max={row['maximum']}"
        )

    lines.extend([
        "",
        "INTERPRETATION LIMITATIONS",
        "-" * 75,
        "Metadata labels do not establish statistical definitions or units.",
        "Missing values are not treated as zero.",
        "Positive values are not automatically default events.",
        "Rate denominators and calculation methods require documentation.",
        "Country coverage does not establish complete historical coverage.",
        "",
        "OUTPUT FILES",
        "-" * 75,
        str(CATALOGUE_OUTPUT),
        str(PROFILE_OUTPUT),
        str(COUNTRY_OUTPUT),
        str(REPORT_OUTPUT),
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
