"""Dataset loading, validation, and deterministic cleaning.

This module contains only data-ingestion and data-quality responsibilities.
It intentionally excludes exploratory analysis, feature scaling, model training,
and MLflow tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_COLUMN = "Personal Loan"
ID_COLUMN = "ID"

REQUIRED_COLUMNS = {
    "ID",
    "Age",
    "Experience",
    "Income",
    "ZIP Code",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Personal Loan",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
}

BINARY_COLUMNS = (
    "Personal Loan",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
)


@dataclass(frozen=True)
class CleaningSummary:
    """Audit information produced by the deterministic cleaning stage."""

    rows_loaded: int
    negative_experience_removed: int
    duplicate_rows_removed: int
    rows_final: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the bank personal-loan CSV and validate its schema.

    Parameters
    ----------
    data_path:
        Path to ``Bank_Personal_Loan_Modelling.csv``.

    Returns
    -------
    pandas.DataFrame
        Raw dataframe exactly as read from disk.
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place Bank_Personal_Loan_Modelling.csv "
            "under data/raw/ or pass --data-path."
        )

    dataframe = pd.read_csv(path)
    validate_schema(dataframe)
    return dataframe


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Validate columns and basic domain constraints used by the baseline."""

    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset schema is incomplete. Missing columns: "
            + ", ".join(missing_columns)
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ValueError("ID contains missing values and cannot be used as an index.")

    duplicated_ids = int(dataframe[ID_COLUMN].duplicated().sum())
    if duplicated_ids:
        raise ValueError(f"ID must be unique; found {duplicated_ids} duplicated IDs.")

    if not dataframe["Family"].isin([1, 2, 3, 4]).all():
        raise ValueError("Family contains values outside the expected domain 1-4.")

    if not dataframe["Education"].isin([1, 2, 3]).all():
        raise ValueError("Education contains values outside the expected domain 1-3.")

    if ((dataframe["Age"] < 18) | (dataframe["Age"] > 100)).any():
        raise ValueError("Age contains values outside the validation range 18-100.")

    for column in BINARY_COLUMNS:
        observed = set(dataframe[column].dropna().unique().tolist())
        if not observed.issubset({0, 1}):
            raise ValueError(
                f"{column} must be binary (0/1). Observed values: {sorted(observed)}"
            )


def clean_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    """Apply the same deterministic cleaning decisions used in the baseline notebook.

    The baseline notebook removes records with negative ``Experience`` because
    they represent invalid years of work experience. It also checks for duplicate
    records and uses ``ID`` as the dataframe index.
    """

    cleaned = dataframe.copy()
    rows_loaded = len(cleaned)

    negative_experience_removed = int((cleaned["Experience"] < 0).sum())
    cleaned = cleaned.loc[cleaned["Experience"] >= 0].copy()

    duplicate_rows_removed = int(cleaned.duplicated().sum())
    if duplicate_rows_removed:
        cleaned = cleaned.drop_duplicates().copy()

    cleaned = cleaned.set_index(ID_COLUMN)

    summary = CleaningSummary(
        rows_loaded=rows_loaded,
        negative_experience_removed=negative_experience_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        rows_final=len(cleaned),
    )

    return cleaned, summary
