import pandas as pd
import numpy as np

def load_data(path="data/my_dataset.csv"):
    df = pd.read_csv(path)

    # Example: assume first column is timestamp, rest are features
    df = df.drop(columns=["timestamp"], errors="ignore")

    # If you have labels (0=normal, 1=anomaly), separate them
    if "label" in df.columns:
        y = df["label"].values
        X = df.drop(columns=["label"]).values
    else:
        y = np.zeros(len(df))   # no labels → assume all normal
        X = df.values

    return X, y