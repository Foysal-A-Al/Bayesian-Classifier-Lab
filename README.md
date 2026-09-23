# Bayesian Classifier Lab

Conda-ready research framework for the five classifier families:

1. Random Forest
2. RBF Support Vector Machine
3. Decision Tree
4. Bernoulli Naive Bayes
5. Gaussian Naive Bayes

All models use identical repeated stratified cross-validation splits. Fold-level scores are retained and compared using a Bayesian correlated t-test and a Region of Practical Equivalence (ROPE).

## Setup

```bash
conda env create -f environment.yml
conda activate bayesian-classifier-lab
```

## Demo

```bash
python scripts/generate_example_data.py
bayesclf --config config.example.yml
```

## Outputs

- `fold_scores.csv`
- `performance_summary.csv`
- `bayesian_<metric>.csv`
- `bayesian_<metric>_matrix.csv`
- posterior-density figures for all 10 pairwise comparisons

Five classifiers produce `C(5,2)=10` pairwise comparisons.

## Tests

```bash
pytest -q
```

CI runs the same test suite on pull requests.

## Design notes

The base package deliberately expects clean numeric predictors. For real laboratory data, missing-value imputation, categorical encoding, batch handling, feature engineering, selection, and tuning should live inside CV-safe pipelines. When tuning hyperparameters, use nested CV and feed only outer-fold results to Bayesian comparison.

See `docs/methodology.md` for the statistical logic.
