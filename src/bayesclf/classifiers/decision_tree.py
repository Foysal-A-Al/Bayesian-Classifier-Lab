from sklearn.tree import DecisionTreeClassifier

def build(random_state: int = 42, **kwargs):
    params = {"criterion":"gini","random_state":random_state}
    params.update(kwargs)
    return DecisionTreeClassifier(**params)
