from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t


def bayesian_correlated_ttest(
    differences, rho, rope=0.01, credible_mass=0.95, decision_threshold=0.95, higher_is_better=True
):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Differences must be a one-dimensional finite array.")
    if d.size < 2:
        raise ValueError("At least two paired finite differences are required.")
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1.")
    if not np.isfinite(rope) or rope < 0:
        raise ValueError("rope must be finite and non-negative.")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must satisfy 0 < credible_mass < 1.")
    if not 0.5 < decision_threshold <= 1:
        raise ValueError("decision_threshold must satisfy 0.5 < decision_threshold <= 1.")
    n = d.size
    df = n - 1
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    scale = float(np.sqrt(var * (1 / n + rho / (1 - rho))))
    if scale == 0:
        pl = float(mean < -rope)
        pe = float(-rope <= mean <= rope)
        pr = float(mean > rope)
        lo = hi = mean
    else:
        dist = t(df=df, loc=mean, scale=scale)
        pl = float(dist.cdf(-rope))
        pe = float(dist.cdf(rope) - dist.cdf(-rope))
        pr = float(dist.sf(rope))
        a = 1 - credible_mass
        lo = float(dist.ppf(a / 2))
        hi = float(dist.ppf(1 - a / 2))
    # Keep the posterior on raw A-minus-B scores; swap superiority tails for losses.
    if not higher_is_better:
        pl, pr = pr, pl
    decision = (
        "A_practically_better"
        if pr >= decision_threshold
        else "B_practically_better"
        if pl >= decision_threshold
        else "practically_equivalent"
        if pe >= decision_threshold
        else "no_decision"
    )
    return {
        "n": int(n),
        "df": int(df),
        "mean_difference_A_minus_B": mean,
        "sample_variance": var,
        "rho": float(rho),
        "posterior_scale": scale,
        "rope": float(rope),
        "p_B_better": pl,
        "p_equivalent": pe,
        "p_A_better": pr,
        "credible_mass": float(credible_mass),
        "credible_interval_low": lo,
        "credible_interval_high": hi,
        "decision_threshold": float(decision_threshold),
        "decision": decision,
        "higher_is_better": bool(higher_is_better),
    }


def compare_pair(
    results,
    model_a,
    model_b,
    metric="accuracy",
    rope=0.01,
    credible_mass=0.95,
    decision_threshold=0.95,
    higher_is_better=None,
):
    a = results.loc[
        results.model == model_a, ["dataset", "repeat", "fold", metric, "test_fraction"]
    ].rename(columns={metric: "score_a"})
    b = results.loc[
        results.model == model_b, ["dataset", "repeat", "fold", metric, "test_fraction"]
    ].rename(columns={metric: "score_b", "test_fraction": "test_fraction_b"})
    p = a.merge(
        b, on=["dataset", "repeat", "fold"], validate="one_to_one", how="outer", indicator=True
    )
    if not p["_merge"].eq("both").all():
        raise ValueError("Models must have identical dataset/repeat/fold keys.")
    if not np.isfinite(p[["score_a", "score_b"]].to_numpy()).all():
        raise ValueError("Paired scores must be finite; investigate undefined metrics.")
    fractions = p[["test_fraction", "test_fraction_b"]].to_numpy()
    if not np.isfinite(fractions).all() or not ((fractions > 0) & (fractions < 1)).all():
        raise ValueError("test_fraction must be finite and strictly between 0 and 1.")
    if not np.allclose(p.test_fraction, p.test_fraction_b):
        raise ValueError("Models must use matching test fractions.")
    if p.empty:
        raise ValueError(f"No paired scores for {model_a} vs {model_b}.")
    if p.dataset.nunique() != 1:
        raise ValueError("Filter to one dataset.")
    if higher_is_better is None:
        higher_is_better = metric not in {"log_loss", "train_seconds", "predict_seconds"}
    out = bayesian_correlated_ttest(
        p.score_a.to_numpy() - p.score_b.to_numpy(),
        float(np.median(p.test_fraction)),
        rope,
        credible_mass,
        decision_threshold,
        higher_is_better,
    )
    out.update(
        {"dataset": p.dataset.iloc[0], "model_A": model_a, "model_B": model_b, "metric": metric}
    )
    return out


def compare_all_pairs(
    results,
    metric="accuracy",
    rope=0.01,
    credible_mass=0.95,
    decision_threshold=0.95,
    higher_is_better=None,
):
    rows = []
    for _, ds in results.groupby("dataset", sort=False):
        for a, b in combinations(list(ds.model.drop_duplicates()), 2):
            rows.append(
                compare_pair(
                    ds, a, b, metric, rope, credible_mass, decision_threshold, higher_is_better
                )
            )
    return pd.DataFrame(rows)


def decision_matrix(pairwise, dataset=None):
    df = pairwise if dataset is None else pairwise[pairwise.dataset == dataset]
    if df.dataset.nunique() != 1:
        raise ValueError("Decision matrix requires exactly one dataset.")
    models = sorted(set(df.model_A).union(df.model_B))
    mat = pd.DataFrame("", index=models, columns=models)
    for m in models:
        mat.loc[m, m] = "—"
    for _, r in df.iterrows():
        label = (
            r.model_A + " better"
            if r.decision == "A_practically_better"
            else r.model_B + " better"
            if r.decision == "B_practically_better"
            else "Equivalent"
            if r.decision == "practically_equivalent"
            else "No decision"
        )
        mat.loc[r.model_A, r.model_B] = label
        mat.loc[r.model_B, r.model_A] = label
    return mat
