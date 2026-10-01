import numpy as np
import pandas as pd
import pytest

from bayesclf.bayesian import bayesian_correlated_ttest, compare_pair
from bayesclf.evaluation import make_splits
from bayesclf.io import load_csv_numeric


@pytest.mark.parametrize(
    "csv,match",
    [
        ("x,x,target\n1,2,A\n2,3,B\n", "Duplicate"),
        ("target\nA\nB\n", "predictor"),
        ("x,target\ninf,A\n2,B\n", "infinite"),
        ("x,target\n1,A\n2,A\n", "two target"),
        ("x,target\n,A\n2,B\n", "Missing"),
    ],
)
def test_invalid_csv_is_rejected(tmp_path, csv, match):
    path = tmp_path / "input.csv"
    path.write_text(csv)
    with pytest.raises(ValueError, match=match):
        load_csv_numeric(path, "target")


def test_valid_csv_accepts_string_labels(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("x,target\n1,A\n2,B\n")
    X, y = load_csv_numeric(path, "target")
    assert X.shape == (2, 1) and y.tolist() == ["A", "B"]


@pytest.mark.parametrize("splits,repeats", [(1, 1), (3, 0), (2.5, 1), (True, 1)])
def test_invalid_cv_settings(splits, repeats):
    with pytest.raises(ValueError):
        make_splits(np.ones((6, 2)), [0, 0, 0, 1, 1, 1], splits, repeats)


def test_rare_class_fails_before_fitting():
    with pytest.raises(ValueError, match="Each target class"):
        make_splits(np.ones((6, 2)), [0, 0, 0, 0, 1, 1], 3, 1)


def test_incomplete_fold_pairs_are_not_silently_dropped():
    rows = pd.DataFrame(
        [
            {
                "dataset": "d",
                "model": model,
                "repeat": 1,
                "fold": fold,
                "accuracy": 0.5,
                "test_fraction": 0.2,
            }
            for model, fold in [("A", 1), ("A", 2), ("B", 1)]
        ]
    )
    with pytest.raises(ValueError, match="identical"):
        compare_pair(rows, "A", "B")


@pytest.mark.parametrize("d", [[0.1, float("nan")], [0.1, float("inf")], [[0.1], [0.2]]])
def test_invalid_paired_differences_are_not_discarded(d):
    with pytest.raises(ValueError, match="one-dimensional finite"):
        bayesian_correlated_ttest(d, 0.2)
