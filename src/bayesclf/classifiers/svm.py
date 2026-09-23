from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build(random_state: int = 42, **kwargs):
    """RBF-SVM; scaling is fitted inside each training fold."""
    params = {"C":1.0,"kernel":"rbf","gamma":"scale","probability":True,"random_state":random_state}
    params.update(kwargs)
    return Pipeline([("scaler",StandardScaler()),("svm",SVC(**params))])
