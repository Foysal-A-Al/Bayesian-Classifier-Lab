from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bayesclf.cli import _run


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "config.demo.yml").read_text())


def test_unknown_metric_is_rejected_before_loading_data(tmp_path):
    cfg = config()
    cfg["evaluation"]["metric"] = "made_up"
    with pytest.raises(ValueError, match="Unsupported metric"):
        _run(cfg, tmp_path)


def test_invalid_configuration_has_readable_error():
    with pytest.raises(ValueError, match="Configuration needs"):
        _run(None)


def test_invalid_rope_is_rejected_before_loading_data(tmp_path):
    cfg = deepcopy(config())
    cfg["bayesian"]["rope"] = -0.1
    with pytest.raises(ValueError, match="rope"):
        _run(cfg, tmp_path)
