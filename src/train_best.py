import json
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

y_train = train["Churn"]
X_train = train.drop(columns=["Churn"])

y_test = test["Churn"]
X_test = test.drop(columns=["Churn"])


with open("artifacts/best_params.json") as f:
    best_params = json.load(f)


cat_cols = X_train.select_dtypes(include="object").columns
num_cols = X_train.select_dtypes(exclude="object").columns


preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])


model = LogisticRegression(**best_params)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])


pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

f1 = f1_score(y_test, preds, pos_label="Yes")

print("FINAL F1:", f1)