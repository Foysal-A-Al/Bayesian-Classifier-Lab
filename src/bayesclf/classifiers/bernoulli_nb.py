from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.naive_bayes import BernoulliNB

def build(random_state: int = 42, **kwargs):
    """Generic numeric baseline: scale then binarize inside the CV fold."""
    params = {"alpha":1.0,"fit_prior":True}
    params.update(kwargs)
    return Pipeline([("scaler",StandardScaler()),("binarizer",Binarizer(threshold=0.0)),("bernoulli_nb",BernoulliNB(**params))])
