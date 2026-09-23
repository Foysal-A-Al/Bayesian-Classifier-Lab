from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import t

def summarize_performance(results):
    m=["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro","roc_auc","log_loss","train_seconds","predict_seconds"]
    return results.groupby(["dataset","model"])[m].agg(["mean","std"]).reset_index()

def save_pairwise_posterior_plots(pairwise,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    for _,r in pairwise.iterrows():
        mean,scale,df,rope=float(r.mean_difference_A_minus_B),float(r.posterior_scale),int(r.df),float(r.rope)
        if scale>0:
            dist=t(df=df,loc=mean,scale=scale); x=np.linspace(dist.ppf(.001),dist.ppf(.999),1000); y=dist.pdf(x)
        else: x,y=np.array([mean]),np.array([1.0])
        fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(x,y); ax.axvline(-rope,ls="--"); ax.axvline(rope,ls="--"); ax.axvline(0,ls=":")
        ax.set_xlabel(f"{r.model_A} - {r.model_B} ({r.metric})"); ax.set_ylabel("Posterior density")
        ax.set_title(f"{r.dataset}: {r.model_A} vs {r.model_B}\nP(B)={r.p_B_better:.3f}  P(eq)={r.p_equivalent:.3f}  P(A)={r.p_A_better:.3f}")
        fig.tight_layout(); fig.savefig(out/f"{r.model_A}_vs_{r.model_B}_{r.metric}.png",dpi=160); plt.close(fig)
