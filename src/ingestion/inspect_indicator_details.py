
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
EXPORT_DIR = ROOT / "outputs" / "exports"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD WEO
# --------------------------------------------------

def load_data():

    files = list(RAW_DIR.glob("*.xlsx"))

    if len(files) != 1:
        raise ValueError(
            "Expected exactly one Excel file in data/raw."
        )

    df = pd.read_excel(
        files[0],
        sheet_name="Countries"
    )

    df.columns = [str(c).strip() for c in df.columns]

    column_mapping = {
        "INDICATOR.ID": "WEO Subject Code",
        "INDICATOR": "Subject Descriptor",
        "UNIT": "Units",
        "SCALE": "Scale",
        "COUNTRY.ID": "ISO"
    }
    for modern_col, legacy_col in column_mapping.items():
        if modern_col in df.columns and legacy_col not in df.columns:
            df[legacy_col] = df[modern_col]

    return df


# --------------------------------------------------
# INSPECT CANDIDATE CODES
# --------------------------------------------------

def inspect_codes(data):

    codes_to_check = [
        "NGDP",
        "NGDPD",
        "PPPGDP",
        "NGDPPC",
        "NGDPDPC",
        "NGDP_RPCH",
        "PCPIPCH",
        "GGXCNL",
        "GGXCNL_NGDP",
        "GGXONLB",
        "GGXONLB_NGDP",
        "GGXWDG_NGDP",
        "LUR",
        "BCA",
        "BCA_NGDPD",
        "TX_RPCH",
        "TM_RPCH"
    ]

    columns = [
        "WEO Subject Code",
        "Subject Descriptor",
        "Units",
        "Scale"
    ]

    missing = [
        col for col in columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    catalogue = (
        data[columns]
        .drop_duplicates()
        .copy()
    )

    catalogue["WEO Subject Code"] = (
        catalogue["WEO Subject Code"]
        .astype(str)
        .str.strip()
    )

    result = catalogue[
        catalogue["WEO Subject Code"].isin(
            codes_to_check
        )
    ].sort_values("WEO Subject Code")

    output = (
        EXPORT_DIR / "weo_indicator_details.csv"
    )

    result.to_csv(output, index=False)

    print("\nINDICATOR DETAILS")
    print("=" * 90)

    if result.empty:
        print("No candidate codes found.")
    else:
        print(result.to_string(index=False))

    print("\nCodes not found:")

    found = set(result["WEO Subject Code"])

    for code in codes_to_check:
        if code not in found:
            print("-", code)

    print("\nSaved:", output)


# --------------------------------------------------
# ADDITIONAL KEYWORD SEARCH
# --------------------------------------------------

def search_descriptions(data):

    keywords = [
        "GDP, constant prices",
        "GDP, current prices",
        "consumer prices",
        "gross debt",
        "general government debt",
        "net lending",
        "net borrowing",
        "unemployment",
        "current account",
        "exports",
        "imports"
    ]

    catalogue = data[
        [
            "WEO Subject Code",
            "Subject Descriptor",
            "Units",
            "Scale"
        ]
    ].drop_duplicates()

    print("\nADDITIONAL DESCRIPTION SEARCH")
    print("=" * 90)

    for keyword in keywords:

        matches = catalogue[
            catalogue["Subject Descriptor"]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                regex=False,
                na=False
            )
        ]

        print(f"\nSEARCH: {keyword}")

        if matches.empty:
            print("No matches.")
        else:
            print(matches.to_string(index=False))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    data = load_data()

    print("WEO records:", len(data))
    print("WEO columns:", len(data.columns))

    inspect_codes(data)
    search_descriptions(data)


if __name__ == "__main__":
    main()
