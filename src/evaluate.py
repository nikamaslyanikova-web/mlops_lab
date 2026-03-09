import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--y-true", type=str, required=True)
    ap.add_argument("--y-pred", type=str, required=True)

    args = ap.parse_args()

    y_true = pd.read_csv(args.y_true)
    y_pred = pd.read_csv(args.y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", acc)
    print("F1:", f1)


if __name__ == "__main__":
    main()