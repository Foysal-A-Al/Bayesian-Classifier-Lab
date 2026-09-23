from sklearn.naive_bayes import GaussianNB

def build(random_state: int = 42, **kwargs):
    params = {"var_smoothing":1e-9}
    params.update(kwargs)
    return GaussianNB(**params)
