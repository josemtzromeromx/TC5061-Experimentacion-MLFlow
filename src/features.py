"""Feature definition and preprocessing for the Personal Loan baseline.

The feature set intentionally preserves the prior notebook baseline. ZIP Code is
excluded because it is a high-cardinality nominal variable, and ID is used only as
an identifier. StandardScaler is returned as the preprocessing transformer so it
can be fitted exclusively on the training partition inside an sklearn Pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Personal Loan"

FEATURE_COLUMNS = (
    "Age",
    "Experience",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
)


def prepare_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and y using exactly the variables selected in the baseline notebook."""

    missing = [column for column in (*FEATURE_COLUMNS, TARGET_COLUMN) if column not in dataframe]
    if missing:
        raise ValueError(f"Columns required for modeling are missing: {missing}")

    X = dataframe.loc[:, FEATURE_COLUMNS].astype(float)
    y = dataframe[TARGET_COLUMN].astype(int)

    return X, y


def build_preprocessor() -> StandardScaler:
    """Return the baseline StandardScaler used before Logistic Regression.

    The scaler is deliberately not fitted here. Fitting occurs when the complete
    sklearn Pipeline is fitted on X_train, preventing information from X_test from
    influencing the learned means and standard deviations.
    """

    return StandardScaler()
