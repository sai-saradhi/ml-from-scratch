from pathlib import Path
import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_california_housing,
)

# ----------------------------
# Create folders
# ----------------------------

DATA_DIR = Path("datasets/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(data, name):
    """
    Convert sklearn dataset to CSV
    """

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    if hasattr(data, "target"):
        X["target"] = data.target

    output_path = DATA_DIR / f"{name}.csv"

    X.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# ----------------------------
# Iris
# ----------------------------

save_dataset(
    load_iris(),
    "iris"
)

# ----------------------------
# Wine
# ----------------------------

save_dataset(
    load_wine(),
    "wine"
)

# ----------------------------
# Breast Cancer
# ----------------------------

save_dataset(
    load_breast_cancer(),
    "breast_cancer"
)

# ----------------------------
# California Housing
# ----------------------------

housing = fetch_california_housing()

housing_df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

housing_df["target"] = housing.target

housing_df.to_csv(
    DATA_DIR / "california_housing.csv",
    index=False
)

print("Saved: california_housing.csv")

print("\nAll datasets downloaded successfully.")