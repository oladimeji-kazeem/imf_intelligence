from pathlib import Path
import pandas as pd

FILE = Path("data/processed/europe_fiscal_risk_input_panel.csv")

df = pd.read_csv(FILE)

print("\nFISCAL RISK INPUT VALIDATION")
print("=" * 55)

# 1. Structural checks
print("\n1. STRUCTURE")
print("Records:", len(df))
print("Countries:", df["iso"].nunique())
print("Years:", df["year"].min(), "-", df["year"].max())

duplicates = df.duplicated(["iso", "year"]).sum()
print("Duplicate country-year records:", duplicates)

# 2. Country coverage
print("\n2. COUNTRY COVERAGE")
coverage = (
    df.groupby(["iso", "country"])
    .agg(
        observations=("year", "count"),
        first_year=("year", "min"),
        last_year=("year", "max")
    )
    .reset_index()
)

print(coverage.to_string(index=False))

# 3. 2024 comparison
print("\n3. 2024 FISCAL SNAPSHOT")

snapshot = df[df["year"] == 2024].copy()

columns = [
    "iso",
    "country",
    "government_debt",
    "fiscal_balance",
    "primary_balance",
    "debt_change_pp",
    "nominal_gdp_growth_pct"
]

print(snapshot[columns].sort_values(
    "government_debt",
    ascending=False
).to_string(index=False))

# 4. Economic plausibility checks
print("\n4. PLAUSIBILITY CHECKS")

checks = {
    "Negative government debt": (
        df["government_debt"] < 0
    ).sum(),
    "Extreme government debt above 300% GDP": (
        df["government_debt"] > 300
    ).sum(),
    "Fiscal balance outside +/- 50% GDP": (
        df["fiscal_balance"].abs() > 50
    ).sum(),
    "Primary balance outside +/- 50% GDP": (
        df["primary_balance"].abs() > 50
    ).sum(),
    "Nominal GDP growth below -50%": (
        df["nominal_gdp_growth_pct"] < -50
    ).sum(),
    "Nominal GDP growth above 100%": (
        df["nominal_gdp_growth_pct"] > 100
    ).sum(),
}

for name, count in checks.items():
    print(f"{name}: {count}")

# 5. Derived debt change verification
print("\n5. DEBT CHANGE FORMULA CHECK")

grouped = df.sort_values(["iso", "year"]).groupby("iso")

previous_debt = grouped["government_debt"].shift(1)
previous_year = grouped["year"].shift(1)

consecutive = (df.sort_values(
    ["iso", "year"]
)["year"] - previous_year) == 1

expected_change = (
    df.sort_values(["iso", "year"])["government_debt"]
    - previous_debt
).where(consecutive)

actual_change = df.sort_values(
    ["iso", "year"]
)["debt_change_pp"]

mismatch = (
    (expected_change - actual_change).abs() > 0.000001
).fillna(False).sum()

print("Debt change mismatches:", mismatch)

# 6. Save country coverage
coverage.to_csv(
    "outputs/reports/europe_fiscal_risk_country_coverage.csv",
    index=False
)

print("\nValidation completed.")
