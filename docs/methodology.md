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

## Lab extension

If hyperparameters are tuned, use nested CV. Tune only inside each outer training fold; only outer-fold scores should enter Bayesian comparison.
