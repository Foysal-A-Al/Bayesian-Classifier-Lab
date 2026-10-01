# Contributing

Useful contributions include input validation, statistical regression tests, accessible result visualizations, and group/time-aware evaluation.

Create an isolated Python 3.10+ environment and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/generate_example_data.py
bayesclf --config config.demo.yml
```

Run setup commands from the repository root. Data/output paths resolve relative to the configuration file. The synthetic demo produces 30 fold-score rows, 10 pairwise rows, 10 posterior figures, and a run manifest. Keep generated datasets and experiment results out of commits.

Bug reports should include the command/configuration, Python and package versions, expected/actual behavior, and a minimal synthetic reproducer. Do not upload private patient or laboratory records.

Keep PRs focused. Explain the problem and resulting behavior, include regression tests for evaluation/statistical changes, and state exactly what was tested. Preprocessing and tuning must preserve separation between training and evaluation data. Discuss major architectural/statistical changes first.

Credit actual collaborators accurately. Coauthor attribution should describe real contributions. Maintainer acceptance is not guaranteed.
