from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/europe_eurostat_sfa_reconciliation.csv"
)

OUTPUT_FILE = Path(
    "outputs/reports/europe_sfa_unit_mismatches.csv"
)

df = pd.read_csv(INPUT_FILE)

nac = df[df["unit"] == "MIO_NAC"].copy()
gdp = df[df["unit"] == "PC_GDP"].copy()

ROUNDING_TOLERANCE = 0.05

valid["nac_positive"] = valid["sfa_nac"] > 0
valid["pct_positive"] = (
    valid["sfa_pct_gdp"] > ROUNDING_TOLERANCE
)

valid["nac_negative"] = valid["sfa_nac"] < 0
valid["pct_negative"] = (
    valid["sfa_pct_gdp"] < -ROUNDING_TOLERANCE
)

valid["rounding_equivalent_zero"] = (
    valid["sfa_pct_gdp"].abs() <= ROUNDING_TOLERANCE
)

valid["genuine_direction_conflict"] = (
    (valid["nac_positive"] & valid["pct_negative"])
    | (valid["nac_negative"] & valid["pct_positive"])
)

print("Valid observations:", len(valid))
print(
    "Rounded-to-zero observations:",
    valid["rounding_equivalent_zero"].sum()
)
print(
    "Genuine direction conflicts:",
    valid["genuine_direction_conflict"].sum()
)

comparison = nac[["country", "year", "sfa_calculated"]].rename(
    columns={"sfa_calculated": "sfa_nac"}
).merge(
    gdp[["country", "year", "sfa_calculated"]].rename(
        columns={"sfa_calculated": "sfa_pct_gdp"}
    ),
    on=["country", "year"],
    how="outer",
)

valid = comparison.dropna(
    subset=["sfa_nac", "sfa_pct_gdp"]
).copy()

valid["same_direction"] = (
    (valid["sfa_nac"] > 0)
    == (valid["sfa_pct_gdp"] > 0)
)

print("Valid matched observations:", len(valid))
print(
    "Direction disagreements:",
    (~valid["same_direction"]).sum()
)

disagreements = valid.loc[
    ~valid["same_direction"],
    ["country", "year", "sfa_nac", "sfa_pct_gdp"]
]

print(disagreements.to_string(index=False))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
disagreements.to_csv(OUTPUT_FILE, index=False)

comparison = nac[["country", "year", "sfa_calculated"]].rename(
    columns={"sfa_calculated": "sfa_nac"}
).merge(
    gdp[["country", "year", "sfa_calculated"]].rename(
        columns={"sfa_calculated": "sfa_pct_gdp"}
    ),
    on=["country", "year"],
    how="outer",
)

valid = comparison.dropna(
    subset=["sfa_nac", "sfa_pct_gdp"]
).copy()

valid["same_direction"] = (
    (valid["sfa_nac"] > 0)
    == (valid["sfa_pct_gdp"] > 0)
)

print("Valid matched observations:", len(valid))
print(
    "Direction disagreements:",
    (~valid["same_direction"]).sum()
)

disagreements = valid.loc[
    ~valid["same_direction"],
    ["country", "year", "sfa_nac", "sfa_pct_gdp"]
]

print(disagreements.to_string(index=False))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
disagreements.to_csv(OUTPUT_FILE, index=False)
