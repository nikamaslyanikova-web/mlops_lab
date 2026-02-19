import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # TotalCharges often stored as string -> convert to numeric, invalid -> NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_pipeline(cat_cols, num_cols, n_estimators, max_depth, min_samples_leaf, random_state):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])


def save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default="data/raw/telco.csv")
    ap.add_argument("--experiment-name", type=str, default="Telco_Churn_LR1")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-samples-leaf", type=int, default=2)

    args = ap.parse_args()

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = load_data(csv_path)

    # Target
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn'")

    y = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    pipe = build_pipeline(
        cat_cols, num_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    mlflow.set_experiment(args.experiment_name)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        # params
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })

        # train
        pipe.fit(X_train, y_train)

        # predict
        y_pred = pipe.predict(X_test)

        # metrics
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        # roc_auc if possible
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
            mlflow.log_metric("test_roc_auc", auc)

        # artifact: confusion matrix
        cm_path = artifacts_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path))

        # model
        mlflow.sklearn.log_model(pipe, artifact_path="model")


if __name__ == "__main__":
    main()
