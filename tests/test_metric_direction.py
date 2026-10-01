import numpy as np
import pandas as pd
import pytest

from bayesclf.bayesian import bayesian_correlated_ttest, compare_pair, decision_matrix


def scores(metric, a, b):
    return pd.DataFrame(
        [
            {
                "dataset": "demo",
                "model": model,
                "repeat": 1,
                "fold": fold,
                metric: value,
                "test_fraction": 0.2,
            }
            for model, values in (("A", a), ("B", b))
            for fold, value in enumerate(values, 1)
        ]
    )


@pytest.mark.parametrize("metric", ["log_loss", "train_seconds", "predict_seconds"])
def test_lower_scores_are_better_for_losses_and_times(metric):
    result = compare_pair(scores(metric, [0.1] * 3, [0.5] * 3), "A", "B", metric)
    assert result["decision"] == "A_practically_better"
    assert result["p_A_better"] == 1.0
    assert result["mean_difference_A_minus_B"] == pytest.approx(-0.4)
    matrix = decision_matrix(pd.DataFrame([result]))
    assert matrix.loc["A", "B"] == "A better"


def test_accuracy_direction_is_unchanged():
    result = compare_pair(scores("accuracy", [0.9] * 3, [0.5] * 3), "A", "B")
    assert result["decision"] == "A_practically_better"


def test_custom_metric_can_override_direction():
    result = compare_pair(
        scores("error", [0.1] * 3, [0.5] * 3), "A", "B", metric="error", higher_is_better=False
    )
    assert result["decision"] == "A_practically_better"


def test_nonzero_variance_swaps_only_superiority_probabilities():
    differences = [-0.2, -0.1, 0.03, -0.04]
    high = bayesian_correlated_ttest(differences, 0.2)
    low = bayesian_correlated_ttest(differences, 0.2, higher_is_better=False)
    assert low["p_A_better"] == high["p_B_better"]
    assert low["p_equivalent"] == high["p_equivalent"]
    assert low["credible_interval_low"] == high["credible_interval_low"]
    assert np.isclose(sum(low[k] for k in ("p_A_better", "p_B_better", "p_equivalent")), 1)


@pytest.mark.parametrize(
    "option,value",
    [
        ("rope", float("nan")),
        ("rope", float("inf")),
        ("credible_mass", 0),
        ("credible_mass", 1),
        ("decision_threshold", 0.5),
        ("decision_threshold", 1.1),
    ],
)
def test_invalid_posterior_settings_fail_clearly(option, value):
    with pytest.raises(ValueError, match=option):
        bayesian_correlated_ttest([0.1, 0.2], 0.2, **{option: value})
