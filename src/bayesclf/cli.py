import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

from .bayesian import bayesian_correlated_ttest, compare_all_pairs, decision_matrix
from .classifiers import get_models
from .evaluation import evaluate_models
from .io import load_csv_numeric
from .reporting import save_pairwise_posterior_plots, summarize_performance


def _run(cfg, base_dir=None):
    if not isinstance(cfg, dict) or any(
        section not in cfg for section in ["data", "evaluation", "bayesian", "output"]
    ):
        raise ValueError("Configuration needs data, evaluation, bayesian, and output sections.")
    d, e, b, o = cfg["data"], cfg["evaluation"], cfg["bayesian"], cfg["output"]
    supported = {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "log_loss",
        "train_seconds",
        "predict_seconds",
    }
    if e["metric"] not in supported:
        raise ValueError(f"Unsupported metric: {e['metric']}. Choose one of {sorted(supported)}")
    bayesian_correlated_ttest([0, 0], 0.1, b["rope"], b["credible_mass"], b["decision_threshold"])
    base_dir = Path(base_dir or Path.cwd())
    X, y = load_csv_numeric(base_dir / d["path"], d["target"])
    results = evaluate_models(
        X.to_numpy(),
        y.to_numpy(),
        get_models(e["random_state"]),
        d["dataset_name"],
        e["n_splits"],
        e["n_repeats"],
        e["random_state"],
    )
    out = base_dir / o["directory"]
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "fold_scores.csv", index=False)
    summarize_performance(results).to_csv(out / "performance_summary.csv", index=False)
    pair = compare_all_pairs(
        results, e["metric"], b["rope"], b["credible_mass"], b["decision_threshold"]
    )
    pair.to_csv(out / f"bayesian_{e['metric']}.csv", index=False)
    decision_matrix(pair, d["dataset_name"]).to_csv(out / f"bayesian_{e['metric']}_matrix.csv")
    save_pairwise_posterior_plots(pair, out / f"posterior_plots_{e['metric']}")
    manifest = {
        "config": cfg,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
        },
        "rows": len(X),
        "predictors": X.columns.tolist(),
        "target_classes": sorted(map(str, y.unique())),
        "score_rows": len(results),
        "pairwise_comparisons": len(pair),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed. Results: {out.resolve()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.example.yml")
    args = p.parse_args()
    try:
        config_path = Path(args.config).resolve()
        with config_path.open(encoding="utf-8") as f:
            _run(yaml.safe_load(f), config_path.parent)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
