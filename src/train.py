import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


def load_data(path: Path):
    df = pd.read_csv(path)

    # convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_pipeline(cat_cols, num_cols, n_estimators, max_depth, min_samples_leaf, random_state):

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipe


def save_confusion_matrix(y_true, y_pred, output_path: Path):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", type=str, default="data/processed/train.csv")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv")

    parser.add_argument("--experiment-name", type=str, default="Telco_Churn_LR2")

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=2)

    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    y_train = train_df["Churn"].map({"Yes": 1, "No": 0})
    X_train = train_df.drop(columns=["Churn"])

    y_test = test_df["Churn"].map({"Yes": 1, "No": 0})
    X_test = test_df.drop(columns=["Churn"])

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pipe = build_pipeline(
        cat_cols,
        num_cols,
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
        args.random_state
    )

    mlflow.set_experiment(args.experiment_name)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with mlflow.start_run():

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state
        })

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            mlflow.log_metric("test_roc_auc", auc)

        cm_path = artifacts_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)

        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(pipe, artifact_path="model")


if __name__ == "__main__":
    main()