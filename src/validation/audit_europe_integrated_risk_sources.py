from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "fiscal": "data/processed/europe_fiscal_risk_signals.csv",
    "growth": "data/processed/europe_growth_risk_classifications.csv",
    "inflation": "data/processed/europe_inflation_risk_classifications.csv",
    "external": "data/processed/europe_external_risk_classifications.csv",
    "labour": "data/processed/europe_labour_risk_classifications.csv",
}

EXPECTED_COUNTRIES = 20

REPORT_DIR = ROOT / "outputs/reports"
REPORT_PATH = REPORT_DIR / "europe_integrated_risk_source_audit.txt"


def inspect_file(engine, relative_path):
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"{engine}: file not found: {path}")

    df = pd.read_csv(path)

    print(f"\n{'=' * 76}")
    print(f"ENGINE: {engine.upper()}")
    print("=" * 76)

    print(f"File: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMN NAMES")
    for column in df.columns:
        print(f"  {column}")

    print("\nCOUNTRY COVERAGE")

    if "iso" not in df.columns:
        raise ValueError(f"{engine}: missing iso column")

    countries = sorted(df["iso"].dropna().unique())

    print(f"Unique countries: {len(countries)}")
    print(f"Countries: {', '.join(countries)}")

    if len(countries) != EXPECTED_COUNTRIES:
        print(
            f"WARNING: Expected {EXPECTED_COUNTRIES} countries; "
            f"found {len(countries)}"
        )

    print("\nMISSING VALUES")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())

    print("\nDUPLICATE COUNTRY RECORDS")
    print(f"Duplicate ISO rows: {df['iso'].duplicated().sum()}")

    if "risk_dimension" in df.columns:
        keys = ["iso", "risk_dimension"]

        if "measurement_type" in df.columns:
            keys.append("measurement_type")

        print(
            "Duplicate dimension keys:",
            df.duplicated(keys).sum()
        )

    if "dimension" in df.columns:
        print(
            "Duplicate dimension keys:",
            df.duplicated(["iso", "dimension"]).sum()
        )

    # Inspect classification labels
    classification_columns = [
        col for col in df.columns
        if "class" in col.lower()
        or "classification" in col.lower()
        or "signal" in col.lower()
    ]

    print("\nCLASSIFICATION VALUES")

    for column in classification_columns:
        values = sorted(
            df[column].dropna().astype(str).unique()
        )
        print(f"{column}: {values}")

    # Inspect percentile ranges
    percentile_columns = [
        col for col in df.columns
        if "percentile" in col.lower()
    ]

    print("\nPERCENTILE RANGES")

    for column in percentile_columns:
        values = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        if not values.empty:
            print(
                f"{column}: min={values.min():.4f}, "
                f"max={values.max():.4f}, "
                f"missing={df[column].isna().sum()}"
            )

    return {
        "engine": engine,
        "rows": len(df),
        "columns": len(df.columns),
        "countries": countries,
        "country_count": len(countries),
        "duplicate_iso_rows": int(df["iso"].duplicated().sum()),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for engine, relative_path in FILES.items():
        result = inspect_file(engine, relative_path)
        summaries.append(result)

    country_sets = {
        result["engine"]: set(result["countries"])
        for result in summaries
    }

    reference = country_sets["fiscal"]

    print(f"\n{'=' * 76}")
    print("CROSS-ENGINE COUNTRY CONSISTENCY")
    print("=" * 76)

    all_consistent = True

    for engine, countries in country_sets.items():
        missing = sorted(reference - countries)
        additional = sorted(countries - reference)

        print(f"\n{engine.upper()}")
        print(f"Missing against fiscal: {missing}")
        print(f"Additional against fiscal: {additional}")

        if missing or additional:
            all_consistent = False

    print("\nSOURCE SUMMARY")
    for result in summaries:
        print(
            f"{result['engine']}: "
            f"{result['rows']} rows, "
            f"{result['country_count']} countries, "
            f"{result['columns']} columns"
        )

    print("\nFINAL AUDIT STATUS")

    if all_consistent:
        print("PASS: All five engines have matching country sets.")
    else:
        print("WARNING: Country coverage differs between engines.")

    report_lines = [
        "EUROPEAN INTEGRATED RISK SOURCE AUDIT",
        "=" * 76,
        "",
        f"Country sets consistent: {all_consistent}",
        "",
    ]

    for result in summaries:
        report_lines.extend([
            f"ENGINE: {result['engine']}",
            f"Rows: {result['rows']}",
            f"Columns: {result['columns']}",
            f"Countries: {result['country_count']}",
            f"Duplicate ISO rows: {result['duplicate_iso_rows']}",
            "",
        ])

    report_lines.append(
        "All five country sets consistent."
        if all_consistent
        else "Country-set differences require investigation."
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(f"\nAudit report: {REPORT_PATH}")


if __name__ == "__main__":
    main()