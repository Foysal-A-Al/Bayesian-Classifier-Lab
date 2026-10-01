# Bayesian Classifier Lab

[![CI](https://github.com/Foysal-A-Al/Bayesian-Classifier-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Foysal-A-Al/Bayesian-Classifier-Lab/actions/workflows/ci.yml)

**Is one classifier meaningfully better, practically equivalent, or is the evidence inconclusive?** Compare five classifier families on identical repeated cross-validation splits, then inspect Bayesian posterior probabilities with a Region of Practical Equivalence (ROPE).

For researchers and students who need more than a leaderboard of mean accuracy scores. The included dataset is synthetic: it demonstrates the workflow, not clinical validation or a reproduction of the full thesis benchmark.

## Quick start

Python 3.10+ is required. Clone the project and create an isolated environment:

```bash
git clone https://github.com/Foysal-A-Al/Bayesian-Classifier-Lab.git
cd Bayesian-Classifier-Lab
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -e ".[dev]"
python scripts/generate_example_data.py
bayesclf --config config.demo.yml
```

The demo uses **3 folds × 2 repeats**: 30 model fits and 10 pairwise comparisons. Runtime depends on your hardware. Outputs go to `results/demo/`. Run from the repository root because configuration paths are relative to your working directory.

Prefer Conda?

```bash
conda env create -f environment.yml
conda activate bayesian-classifier-lab
python scripts/generate_example_data.py
bayesclf --config config.demo.yml
```

For **10 folds × 10 repeats** (500 model fits), use `bayesclf --config config.example.yml`. See [Docker instructions](docs/docker.md) for containers.

## Models

| Model | Family |
|---|---|
| RandomForest | Random Forest |
| SVM | RBF Support Vector Machine |
| DecisionTree | Decision Tree |
| BernoulliNB | Bernoulli Naive Bayes |
| GaussianNB | Gaussian Naive Bayes |

Every model receives identical split indices. Model-specific preprocessing lives in the [classifier pipelines](src/bayesclf/classifiers).

## Bring your own CSV

Copy `config.demo.yml`, set `data.path`, `data.target`, and `data.dataset_name`, then run `bayesclf --config your-config.yml`. Supply numeric predictors and a classification target. Choose your metric, fold/repeat counts, output directory, and a scientifically justified ROPE.

Supported metrics: `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `roc_auc`, `log_loss`, `train_seconds`, and `predict_seconds`. Accuracy/F1/AUC-style metrics are higher-is-better; log loss and timings are lower-is-better. Runtime comparisons also depend on hardware and system load.

The loader rejects missing values and nonnumeric predictors. Imputation, encoding, feature selection, and tuning belong inside CV-safe pipelines. Each class should have at least as many examples as the fold count. Grouped patients/subjects and longitudinal measurements need appropriate group/time-aware splits instead of the default stratified folds.

## Read the results

| Output | Purpose |
|---|---|
| `fold_scores.csv` | Scores, timings, and split identifiers for each model/fold |
| `performance_summary.csv` | Per-model means and standard deviations |
| `bayesian_<metric>.csv` | Probabilities, raw mean differences, intervals, and decisions |
| `bayesian_<metric>_matrix.csv` | All model-pair decisions |
| `posterior_plots_<metric>/` | Ten posterior-density figures |

Inspect `p_A_better`, `p_equivalent`, and `p_B_better`. At the default threshold, a category needs at least 0.95 posterior probability to trigger a decision; otherwise the result is `no_decision`. Inconclusive evidence does **not** establish equivalence.

`mean_difference_A_minus_B` and its interval retain raw score units: for log loss, a negative difference favors A. A ROPE of 0.01 is one percentage point for accuracy; other metrics need their own justification. The posterior depends on the correlated-t model and its assumptions. See [methodology](docs/methodology.md) for formulas and direction details.

## Development

```bash
python -m pytest -q
```

CI runs tests on pull requests. Reproducible bug reports and improvements to validation, statistical tests, and documentation are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
