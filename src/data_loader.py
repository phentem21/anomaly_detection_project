import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path="data/dataset.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    # Separate label if present
    if "label" in df.columns:
        y = df["label"].values
        df = df.drop(columns=["label"])
    else:
        y = None

    # Drop non-numeric columns
    df = df.select_dtypes(include=["number"])

    # Remove columns with non-scalar values
    df = df.loc[:, df.applymap(np.isscalar).all()]

    # Handle NaNs
    df = df.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X, y