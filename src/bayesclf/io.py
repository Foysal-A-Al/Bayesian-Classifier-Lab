import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv_numeric(path, target):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names are not allowed.")
    df = pd.read_csv(Path(path))
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    X, y = df.drop(columns=[target]), df[target]
    if df.empty or X.shape[1] == 0:
        raise ValueError("Provide at least one row and one predictor column.")
    bad = X.select_dtypes(exclude="number").columns.tolist()
    if bad:
        raise ValueError(f"Base pipeline expects numeric predictors. Encode first: {bad}")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values detected; impute inside a CV-safe pipeline.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Predictors contain infinite values.")
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    return X, y
