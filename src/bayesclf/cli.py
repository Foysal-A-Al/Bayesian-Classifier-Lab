from pathlib import Path
import argparse,yaml
from .classifiers import get_models
from .evaluation import evaluate_models
from .bayesian import compare_all_pairs,decision_matrix
from .reporting import summarize_performance,save_pairwise_posterior_plots
from .io import load_csv_numeric

def _run(cfg):
    d,e,b,o=cfg["data"],cfg["evaluation"],cfg["bayesian"],cfg["output"]
    X,y=load_csv_numeric(d["path"],d["target"])
    results=evaluate_models(X.to_numpy(),y.to_numpy(),get_models(e["random_state"]),d["dataset_name"],e["n_splits"],e["n_repeats"],e["random_state"])
    out=Path(o["directory"]); out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/"fold_scores.csv",index=False)
    summarize_performance(results).to_csv(out/"performance_summary.csv",index=False)
    pair=compare_all_pairs(results,e["metric"],b["rope"],b["credible_mass"],b["decision_threshold"])
    pair.to_csv(out/f"bayesian_{e['metric']}.csv",index=False)
    decision_matrix(pair,d["dataset_name"]).to_csv(out/f"bayesian_{e['metric']}_matrix.csv")
    save_pairwise_posterior_plots(pair,out/f"posterior_plots_{e['metric']}")
    print(f"Completed. Results: {out.resolve()}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.example.yml"); args=p.parse_args()
    with open(args.config,encoding="utf-8") as f: _run(yaml.safe_load(f))

if __name__=="__main__": main()
