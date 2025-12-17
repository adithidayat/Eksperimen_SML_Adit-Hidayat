import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load dataset mentah
    """
    return pd.read_csv("breast_cancer.csv")


def preprocess_data(df):
    """
    Preprocessing data sesuai eksperimen
    """
    # Drop kolom tidak relevan
    df = df.drop(columns=["id", "Unnamed: 32"])

    # Encode target
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isna().sum() != 0:
        raise ValueError("Target diagnosis mengandung NaN")

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def save_data(X_train, X_test, y_train, y_test, columns):
    """
    Simpan hasil preprocessing
    """
    output_dir = "preprocessing/namadataset_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=columns).to_csv(
        f"{output_dir}/X_train.csv", index=False
    )
    pd.DataFrame(X_test, columns=columns).to_csv(
        f"{output_dir}/X_test.csv", index=False
    )

    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, columns = preprocess_data(df)
    save_data(X_train, X_test, y_train, y_test, columns)

    print("Preprocessing otomatis selesai.")
    print("Output ada di: preprocessing/breast_cancer_preprocessing")


if __name__ == "__main__":
    main()

