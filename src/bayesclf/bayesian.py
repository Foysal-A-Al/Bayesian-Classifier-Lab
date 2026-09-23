from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import t

def bayesian_correlated_ttest(differences,rho,rope=0.01,credible_mass=0.95,decision_threshold=0.95):
    d=np.asarray(differences,dtype=float); d=d[np.isfinite(d)]
    if d.size<2: raise ValueError("At least two paired finite differences are required.")
    if not 0<=rho<1: raise ValueError("rho must satisfy 0 <= rho < 1.")
    if rope<0: raise ValueError("rope must be non-negative.")
    n=d.size; df=n-1; mean=float(d.mean()); var=float(d.var(ddof=1))
    scale=float(np.sqrt(var*(1/n+rho/(1-rho))))
    if scale==0:
        pl=float(mean<-rope); pe=float(-rope<=mean<=rope); pr=float(mean>rope); lo=hi=mean
    else:
        dist=t(df=df,loc=mean,scale=scale); pl=float(dist.cdf(-rope)); pe=float(dist.cdf(rope)-dist.cdf(-rope)); pr=float(1-dist.cdf(rope))
        a=1-credible_mass; lo=float(dist.ppf(a/2)); hi=float(dist.ppf(1-a/2))
    decision="A_practically_better" if pr>=decision_threshold else "B_practically_better" if pl>=decision_threshold else "practically_equivalent" if pe>=decision_threshold else "no_decision"
    return {"n":int(n),"df":int(df),"mean_difference_A_minus_B":mean,"sample_variance":var,"rho":float(rho),
      "posterior_scale":scale,"rope":float(rope),"p_B_better":pl,"p_equivalent":pe,"p_A_better":pr,
      "credible_mass":float(credible_mass),"credible_interval_low":lo,"credible_interval_high":hi,
      "decision_threshold":float(decision_threshold),"decision":decision}

def compare_pair(results,model_a,model_b,metric="accuracy",rope=0.01,credible_mass=0.95,decision_threshold=0.95):
    a=results.loc[results.model==model_a,["dataset","repeat","fold",metric,"test_fraction"]].rename(columns={metric:"score_a"})
    b=results.loc[results.model==model_b,["dataset","repeat","fold",metric]].rename(columns={metric:"score_b"})
    p=a.merge(b,on=["dataset","repeat","fold"],validate="one_to_one").dropna(subset=["score_a","score_b"])
    if p.empty: raise ValueError(f"No paired scores for {model_a} vs {model_b}.")
    if p.dataset.nunique()!=1: raise ValueError("Filter to one dataset.")
    out=bayesian_correlated_ttest(p.score_a.to_numpy()-p.score_b.to_numpy(),float(np.median(p.test_fraction)),rope,credible_mass,decision_threshold)
    out.update({"dataset":p.dataset.iloc[0],"model_A":model_a,"model_B":model_b,"metric":metric}); return out

def compare_all_pairs(results,metric="accuracy",rope=0.01,credible_mass=0.95,decision_threshold=0.95):
    rows=[]
    for _,ds in results.groupby("dataset",sort=False):
        for a,b in combinations(list(ds.model.drop_duplicates()),2):
            rows.append(compare_pair(ds,a,b,metric,rope,credible_mass,decision_threshold))
    return pd.DataFrame(rows)

def decision_matrix(pairwise,dataset=None):
    df=pairwise if dataset is None else pairwise[pairwise.dataset==dataset]
    if df.dataset.nunique()!=1: raise ValueError("Decision matrix requires exactly one dataset.")
    models=sorted(set(df.model_A).union(df.model_B)); mat=pd.DataFrame("",index=models,columns=models)
    for m in models: mat.loc[m,m]="—"
    for _,r in df.iterrows():
        label=r.model_A+" better" if r.decision=="A_practically_better" else r.model_B+" better" if r.decision=="B_practically_better" else "Equivalent" if r.decision=="practically_equivalent" else "No decision"
        mat.loc[r.model_A,r.model_B]=label; mat.loc[r.model_B,r.model_A]=label
    return mat
