# Methodology

The framework starts with the five thesis classifiers: Random Forest, RBF-SVM, Decision Tree, Bernoulli Naive Bayes, and Gaussian Naive Bayes.

## Paired repeated cross-validation

Every classifier receives the same split indices. With 10 folds and 10 repeats, each model yields 100 paired out-of-sample scores.

For models A and B:

```text
d_i = score_A,i - score_B,i
```

## Bayesian correlated t-test

```text
mu | d ~ Student-t(
  df=n-1,
  loc=mean(d),
  scale=sqrt(s^2 * (1/n + rho/(1-rho)))
)
```

The implementation uses `rho = n_test/n_total`. For equal 10-fold CV this is approximately 0.1.

## ROPE

For ROPE `[-r,+r]`:
- `P(mu < -r)`: B practically better
- `P(-r <= mu <= r)`: practical equivalence
- `P(mu > r)`: A practically better

The default `r=0.01` reproduces the thesis-style one-percentage-point ROPE for accuracy. New metrics should use scientifically justified practical thresholds.

## Metric direction

Accuracy, balanced accuracy, macro precision/recall/F1, and ROC AUC are higher-is-better metrics. Log loss and train/prediction times are lower-is-better metrics. Pairwise comparison infers these directions for the built-in metrics and records `higher_is_better` in its output. For custom error metrics, pass `higher_is_better=False` to the Python comparison API.

The posterior mean and credible interval always describe the raw difference `score_A - score_B`. For lower-is-better metrics, a negative difference favors A, and the superiority probabilities swap tails. ROPE equivalence is unchanged. The raw `bayesian_correlated_ttest` API defaults to higher-is-better; specify the direction explicitly when passing loss differences.

## Lab extension

If hyperparameters are tuned, use nested CV. Tune only inside each outer training fold; only outer-fold scores should enter Bayesian comparison.
