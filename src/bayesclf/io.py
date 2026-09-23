from pathlib import Path
import pandas as pd

def load_csv_numeric(path,target):
    df=pd.read_csv(Path(path))
    if target not in df.columns: raise ValueError(f"Target column '{target}' not found.")
    X,y=df.drop(columns=[target]),df[target]
    bad=X.select_dtypes(exclude="number").columns.tolist()
    if bad: raise ValueError(f"Base pipeline expects numeric predictors. Encode first: {bad}")
    if X.isna().any().any() or y.isna().any(): raise ValueError("Missing values detected; impute inside a CV-safe pipeline.")
    return X,y
