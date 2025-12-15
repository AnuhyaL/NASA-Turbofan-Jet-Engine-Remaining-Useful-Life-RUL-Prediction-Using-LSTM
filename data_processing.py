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

    # Use whitespace delimiter; NASA files are space-separated
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLS
    return df



def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each row.

    RUL = (max cycle for that unit) - (current cycle)

    Optionally cap RUL at a maximum value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ["unit", "cycle", ...].
    cap : int or None
        Maximum RUL value. If None, no capping is applied.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added 'RUL' column.
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if cap is not None:
        df["RUL"] = np.minimum(df["RUL"], cap)
    return df


def create_sliding_windows(df: pd.DataFrame, seq_len: int = 50):
    feature_cols = [c for c in df.columns if c.startswith("op_") or c.startswith("sensor_")]

    
    # Normalize features before sequences
   
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

