from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "B9": ROOT / "data/processed/europe_eurostat_b9.csv",
    "GDP": ROOT / "data/processed/europe_eurostat_edp_gdp_2019_2024.csv",
    "Interest rate panel": ROOT / "data/processed/europe_effective_interest_rate_panel.csv",
}

for name, path in FILES.items():

    print("\n" + "=" * 75)
    print(name.upper())
    print("=" * 75)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path)

    print("File:", path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    if "year" in df.columns:
        print("\nYear coverage:")
        print(df["year"].min(), "to", df["year"].max())

    country_column = next(
        (c for c in ["country_code", "geo", "iso"] if c in df.columns),
        None
    )

    if country_column:
        print("Countries:", df[country_column].nunique())

    print("\nMissing values:")
    print(df.isna().sum().to_string())
