import optuna
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import json


train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

y_train = train["Churn"]
X_train = train.drop(columns=["Churn"])

y_test = test["Churn"]
X_test = test.drop(columns=["Churn"])

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])


def objective(trial):

    C = trial.suggest_float("C", 1e-3, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 1000)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver
    )

    pipeline = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    with mlflow.start_run(nested=True):

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        f1 = f1_score(y_test, preds, pos_label="Yes")

        mlflow.log_params({
            "C": C,
            "max_iter": max_iter,
            "solver": solver
        })

        mlflow.log_metric("f1_score", f1)

    return f1


mlflow.set_experiment("telco_hpo")

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=20)

print("Best params:", study.best_params)
print("Best score:", study.best_value)

with open("artifacts/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)