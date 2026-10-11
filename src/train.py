"""CLI entry point for reproducible Logistic Regression experiments with MLflow.

Example
-------
python src/train.py --C 1.0 --solver lbfgs --class-weight none \
    --max-iter 1000 --random-state 4284
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataset import clean_dataset, load_dataset
from features import FEATURE_COLUMNS, build_preprocessor, prepare_features


DEFAULT_DATA_PATH = "data/raw/Bank_Personal_Loan_Modelling.csv"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "TC5061_Bank_Personal_Loan_Experiments"


def parse_class_weight(value: str) -> str | None:
    """Translate CLI-friendly values into sklearn's class_weight values."""

    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized == "balanced":
        return "balanced"
    raise argparse.ArgumentTypeError("--class-weight must be 'none' or 'balanced'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and track the Bank Personal Loan Logistic Regression "
            "experiment in MLflow. Defaults reproduce the prior notebook baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Input CSV path.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Optimization algorithm for LogisticRegression.",
    )
    parser.add_argument(
        "--class-weight",
        type=parse_class_weight,
        default=None,
        metavar="{none,balanced}",
        help="Class weighting strategy. Use 'balanced' to compensate for class imbalance.",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=4284,
        help="Seed used for the train/test split and compatible model solvers.",
    )
    parser.add_argument("--test-size", type=float, default=0.33, help="Fraction reserved for testing.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow Tracking URI.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT, help="MLflow experiment name.")
    parser.add_argument("--run-name", default=None, help="Optional explicit MLflow run name.")

    args = parser.parse_args()

    if args.C <= 0:
        parser.error("--C must be greater than zero.")
    if args.max_iter <= 0:
        parser.error("--max-iter must be greater than zero.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")

    return args


def set_reproducibility_seed(random_state: int) -> None:
    """Fix Python and NumPy seeds in addition to sklearn random_state."""

    random.seed(random_state)
    np.random.seed(random_state)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform the stratified split used by the previous notebook."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model(
    *,
    C: float,
    solver: str,
    class_weight: str | None,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Build the complete preprocessing + Logistic Regression pipeline."""

    classifier = LogisticRegression(
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("scaler", build_preprocessor()),
            ("logistic_regression", classifier),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluation metrics, predictions, probabilities, and confusion matrix."""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, y_pred, y_proba, cm


def create_artifacts(
    artifact_dir: Path,
    *,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    cm: np.ndarray,
    metrics: dict[str, float],
) -> list[Path]:
    """Create files that are explicitly uploaded with mlflow.log_artifact()."""

    artifact_dir.mkdir(parents=True, exist_ok=True)

    report_path = artifact_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            file,
            indent=2,
        )

    metrics_path = artifact_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cm_path = artifact_dir / "confusion_matrix.png"
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"]).plot(
        ax=ax_cm,
        colorbar=False,
    )
    ax_cm.set_title("Matriz de confusión - Regresión Logística")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close(fig_cm)

    roc_path = artifact_dir / "roc_curve.png"
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        name="Regresión Logística",
        ax=ax_roc,
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_title("Curva ROC - Regresión Logística")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close(fig_roc)

    return [report_path, metrics_path, cm_path, roc_path]


def make_run_name(args: argparse.Namespace) -> str:
    """Create a descriptive run name when the user does not provide one."""

    if args.run_name:
        return args.run_name

    weight = args.class_weight if args.class_weight is not None else "none"
    return f"lr_C-{args.C:g}_solver-{args.solver}_weight-{weight}_seed-{args.random_state}"


def main() -> None:
    args = parse_args()
    set_reproducibility_seed(args.random_state)

    raw_df = load_dataset(args.data_path)
    clean_df, cleaning_summary = clean_dataset(raw_df)
    X, y = prepare_features(clean_df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_model(
        C=args.C,
        solver=args.solver,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    metrics, y_pred, y_proba, cm = evaluate_model(model, X_test, y_test)

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.set_experiment(args.experiment_name)
    run_name = make_run_name(args)

    params_to_log = {
        "model_type": "LogisticRegression",
        "preprocessing": "StandardScaler",
        "C": args.C,
        "solver": args.solver,
        "class_weight": str(args.class_weight),
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "stratify": True,
        "n_features": len(FEATURE_COLUMNS),
        "rows_loaded": cleaning_summary.rows_loaded,
        "negative_experience_removed": cleaning_summary.negative_experience_removed,
        "duplicate_rows_removed": cleaning_summary.duplicate_rows_removed,
        "rows_final": cleaning_summary.rows_final,
    }

    with tempfile.TemporaryDirectory(prefix="tc5061_mlflow_") as temp_dir:
        artifact_paths = create_artifacts(
            Path(temp_dir),
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            cm=cm,
            metrics=metrics,
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params_to_log)
            mlflow.log_metrics(metrics)

            mlflow.set_tags(
                {
                    "dataset": Path(args.data_path).name,
                    "task": "binary_classification",
                    "target": "Personal Loan",
                    "project": "TC5061_MLFlow_Experimentation",
                    "source_notebook": "TC5061_Notebook_Baseline_MartinezJose.ipynb",
                }
            )

            for artifact_path in artifact_paths:
                mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")

            input_example = X_train.head(5)
            output_example = model.predict(input_example)
            signature = infer_signature(input_example, output_example)

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="personal_loan_model",
                input_example=input_example,
                signature=signature,
            )

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

    print("\nExperiment completed successfully")
    print("-" * 60)
    print(f"Experiment       : {experiment.name}")
    print(f"Experiment ID    : {experiment_id}")
    print(f"Run name         : {run_name}")
    print(f"Run ID           : {run_id}")
    print(f"Model URI        : {model_info.model_uri}")
    print(f"Training rows    : {len(X_train)}")
    print(f"Test rows        : {len(X_test)}")
    print(f"Features         : {len(FEATURE_COLUMNS)}")
    print("\nMetrics")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:10s}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
