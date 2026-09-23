from __future__ import annotations
import time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

def make_splits(X,y,n_splits=10,n_repeats=10,random_state=42):
    return list(RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state).split(X,y))

def _prob_metrics(model,X_test,y_test):
    roc_auc,ll=np.nan,np.nan
    if not hasattr(model,"predict_proba"): return roc_auc,ll
    try:
        p=model.predict_proba(X_test); classes=np.asarray(model.classes_)
        ll=log_loss(y_test,p,labels=classes)
        roc_auc=roc_auc_score(y_test,p[:,1]) if len(classes)==2 else roc_auc_score(y_test,p,labels=classes,average="macro",multi_class="ovr")
    except Exception: pass
    return roc_auc,ll

def evaluate_models(X,y,models,dataset_name="dataset",n_splits=10,n_repeats=10,random_state=42):
    X,y=np.asarray(X),np.asarray(y); rows=[]
    splits=make_splits(X,y,n_splits,n_repeats,random_state)
    for idx,(tr,te) in enumerate(splits):
        repeat=idx//n_splits+1; fold=idx%n_splits+1
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        for name,est in models.items():
            model=clone(est)
            t0=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-t0
            t1=time.perf_counter(); pred=model.predict(Xte); pred_s=time.perf_counter()-t1
            auc,ll=_prob_metrics(model,Xte,yte)
            rows.append({"dataset":dataset_name,"model":name,"repeat":repeat,"fold":fold,
              "accuracy":accuracy_score(yte,pred),"balanced_accuracy":balanced_accuracy_score(yte,pred),
              "precision_macro":precision_score(yte,pred,average="macro",zero_division=0),
              "recall_macro":recall_score(yte,pred,average="macro",zero_division=0),
              "f1_macro":f1_score(yte,pred,average="macro",zero_division=0),
              "roc_auc":auc,"log_loss":ll,"train_seconds":train_s,"predict_seconds":pred_s,
              "n_train":len(tr),"n_test":len(te),"test_fraction":len(te)/(len(tr)+len(te))})
    return pd.DataFrame(rows)
