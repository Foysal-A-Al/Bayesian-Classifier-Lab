from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
ROOT=Path(__file__).resolve().parents[1]
X,y=make_classification(n_samples=600,n_features=12,n_informative=7,n_redundant=2,random_state=42)
df=pd.DataFrame(X,columns=[f"feature_{i+1}" for i in range(X.shape[1])]); df["target"]=y
path=ROOT/"data"/"example_dataset.csv"; path.parent.mkdir(exist_ok=True); df.to_csv(path,index=False); print(path)
