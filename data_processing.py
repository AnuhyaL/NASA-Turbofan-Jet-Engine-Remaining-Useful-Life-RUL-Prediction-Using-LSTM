import pandas as pd
import numpy as np
import os

# Column names for NASA CMAPSS FD datasets
COLS = ["unit", "cycle"] + [f"op_{i+1}" for i in range(3)] + [f"sensor_{i+1}" for i in range(21)]


def load_cmapss(path: str) -> pd.DataFrame:
    """
    Load NASA C-MAPSS dataset from a text file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # NASA files are whitespace-separated
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLS
    return df


def safe_load_dataset(path: str):
    """
    Safely load dataset with exception handling.
    """
    try:
        return load_cmapss(path)
    except FileNotFoundError:
        print("ERROR: Dataset file not found.")
        return None


def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each row.

    RUL = (max cycle for that unit) - (current cycle)
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    if cap is not None:
        df["RUL"] = np.minimum(df["RUL"], cap)

    return df


def create_sliding_windows(df: pd.DataFrame, seq_len: int = 50):
    """
    Create fixed-length time-series sequences for LSTM training.
    """
    # ✅ List comprehension (Part-2 requirement)
    feature_cols = [c for c in df.columns if c.startswith("op_") or c.startswith("sensor_")]

    # Normalize features before sequence generation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = [], []

    for unit_id in df["unit"].unique():
        sub = df[df["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)

        data = sub[feature_cols].values
        rul = sub["RUL"].values

        if len(sub) < seq_len:
            continue

        for i in range(seq_len - 1, len(sub)):
            X.append(data[i - seq_len + 1 : i + 1])
            y.append(rul[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def batch_generator(X, y, batch_size):
    """
    Generator that yields mini-batches of data.
    (Part-2 advanced requirement: generator function)
    """
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


class CMAPSSDataset:
    """
    Handles loading and preprocessing of CMAPSS dataset.
    Demonstrates composition with the RULPredictor model.
    """

    def __init__(self, path: str, rul_cap: int = 125):
        self.path = path
        self.rul_cap = rul_cap
        self.data = None

    def load_data(self):
        """
        Load dataset and compute RUL.
        """
        self.data = load_cmapss(self.path)
        self.data = add_rul(self.data, cap=self.rul_cap)
        return self.data
