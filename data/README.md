# Data

Store local CSV classification datasets here. The target column is selected in the YAML configuration. Predictors must be numeric, finite, and nonmissing; duplicate column names and single-class targets are rejected. Each target class must contain at least as many rows as the configured fold count.

Do not commit sensitive or confidential laboratory data. Generated demo datasets are excluded from source control.

Generate the local demo CSV with:

```bash
python scripts/generate_example_data.py
```
