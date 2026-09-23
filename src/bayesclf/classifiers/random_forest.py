from sklearn.ensemble import RandomForestClassifier

def build(random_state: int = 42, **kwargs):
    """Random Forest. Gini: G(t)=1-sum_k p(k|t)^2."""
    params = {"n_estimators":500,"criterion":"gini","n_jobs":-1,"random_state":random_state}
    params.update(kwargs)
    return RandomForestClassifier(**params)
