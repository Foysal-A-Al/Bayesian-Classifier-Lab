from .random_forest import build as random_forest
from .svm import build as svm
from .decision_tree import build as decision_tree
from .bernoulli_nb import build as bernoulli_nb
from .gaussian_nb import build as gaussian_nb

def get_models(random_state: int = 42):
    return {
        "RandomForest": random_forest(random_state=random_state),
        "SVM": svm(random_state=random_state),
        "DecisionTree": decision_tree(random_state=random_state),
        "BernoulliNB": bernoulli_nb(random_state=random_state),
        "GaussianNB": gaussian_nb(random_state=random_state),
    }
