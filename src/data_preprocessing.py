"""
data_preprocessing.py
---------------------
Loading, cleaning and feature-engineering utilities for the Customer Churn
prediction project.

The dataset ships as two files: a training set and a separate testing set.
Both share the same schema:

    CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls,
    Payment Delay, Subscription Type, Contract Length, Total Spend,
    Last Interaction, Churn (target: 1 = churned, 0 = retained)
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, so the scripts work no matter
# what the current working directory is).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
TEST_CSV = DATA_DIR / "customer_churn_dataset-testing-master.csv"

TARGET = "Churn"
ID_COL = "CustomerID"

NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
CATEGORICAL_FEATURES = ["Gender", "Subscription Type", "Contract Length"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw():
    """Return the raw (train, test) DataFrames straight from disk."""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop malformed rows and coerce the target to integer.

    The published training file contains a single fully-empty trailing row;
    dropping rows without a target removes it cleanly. Numeric columns are
    stored as floats in the CSV, which is fine for scikit-learn.
    """
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def get_xy(df: pd.DataFrame):
    """Split a cleaned DataFrame into the feature matrix X and target y."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: scale numerics, one-hot encode categoricals.

    Scaling is only strictly required by Logistic Regression, but applying it
    inside a shared pipeline keeps every model reproducible and leak-free
    (the scaler is fit on training data only).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_clean_split(test_size: float = 0.2, random_state: int = 42):
    """Return a stratified train/validation split of the *training* file.

    Important data note
    -------------------
    The dataset ships two files, but they do NOT come from the same
    distribution: within the training file the labels follow near-deterministic
    rules (e.g. ``Age > 50`` or a ``Monthly`` contract imply ~100% churn),
    whereas in the testing file those same conditions only churn ~50-60% of the
    time. Treating the testing file as a hold-out set therefore measures data
    drift, not model quality, and makes every model look broken.

    The honest, in-distribution evaluation is a stratified split of the training
    file (done here). The separate testing file is loaded via
    :func:`load_scoring_set` and used as an *unseen scoring set* for the
    Power BI dashboard.
    """
    train, _ = load_raw()
    train = clean(train)
    X, y = get_xy(train)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_scoring_set():
    """Return (X, y) from the provided testing file - unseen customers to score.

    Because of the distribution drift described in :func:`load_clean_split`,
    this set is used only to demonstrate scoring/deployment (feeding the Power BI
    dashboard), not to report headline model metrics.
    """
    _, test = load_raw()
    test = clean(test)
    return get_xy(test)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_clean_split()
    print("Train split:", X_train.shape, "| Validation split:", X_test.shape)
    print("Churn rate (train): {:.1%}".format(y_train.mean()))
    print("Churn rate (val):   {:.1%}".format(y_test.mean()))
