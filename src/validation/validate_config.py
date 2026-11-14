from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDICATOR_FILE = PROJECT_ROOT / "config" / "indicators.yaml"
COUNTRY_FILE = PROJECT_ROOT / "config" / "countries.yaml"


def load_yaml(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_indicators(config):

    indicators = config.get("indicators", {})

    required_fields = [
        "name",
        "category",
        "unit",
        "frequency",
        "direction_of_risk",
        "required"
    ]

    for indicator, details in indicators.items():

        for field in required_fields:

            if field not in details:
                raise ValueError(
                    f"{indicator} is missing {field}"
                )

    print(f"Indicators registered: {len(indicators)}")


def validate_countries(config):

    regions = config.get("regions", {})

    for region, details in regions.items():

        countries = details.get("countries", [])

        if not countries:
            raise ValueError(
                f"No countries configured for {region}"
            )

        if len(countries) != len(set(countries)):
            raise ValueError(
                f"Duplicate country codes in {region}"
            )

        for code in countries:

            if len(code) != 3 or not code.isalpha():
                raise ValueError(
                    f"Invalid country code: {code}"
                )

        print(
            f"{region}: {len(countries)} countries configured"
        )


def main():

    indicators = load_yaml(INDICATOR_FILE)
    countries = load_yaml(COUNTRY_FILE)

    validate_indicators(indicators)
    validate_countries(countries)

    print("\nConfiguration validation successful.")


if __name__ == "__main__":
    main()
