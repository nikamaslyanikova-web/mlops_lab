import os
import json
import pandas as pd


# -------------------------
# PRE-TRAIN TEST
# -------------------------
def test_data_schema_basic():
    path = "data/processed/train.csv"
    assert os.path.exists(path), "Train data not found"

    df = pd.read_csv(path)

    # перевірка колонок
    assert "Churn" in df.columns, "Missing target column"

    # перевірка розміру
    assert df.shape[0] > 50, "Too few rows"

    # перевірка NaN
    assert df["Churn"].notna().all(), "Target has NaN"


# -------------------------
# POST-TRAIN TEST
# -------------------------
def test_artifacts_exist():
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("confusion_matrix.png"), "confusion matrix missing"


def test_quality_gate():
    threshold = float(os.getenv("F1_THRESHOLD", "0.7"))

    with open("metrics.json") as f:
        metrics = json.load(f)

    f1 = metrics["f1"]

    assert f1 >= threshold, f"F1 too low: {f1}"
