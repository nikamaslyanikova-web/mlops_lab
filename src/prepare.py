import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/raw/telco.csv")
    ap.add_argument("--output-dir", type=str, default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # очистка TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["Churn"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)


if __name__ == "__main__":
    main()
