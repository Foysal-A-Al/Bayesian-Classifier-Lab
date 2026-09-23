import numpy as np
from sklearn.datasets import make_classification
from bayesclf.classifiers import get_models
from bayesclf.evaluation import evaluate_models
from bayesclf.bayesian import compare_all_pairs

def test_complete_pipeline():
    X,y=make_classification(n_samples=180,n_features=8,n_informative=5,random_state=1)
    results=evaluate_models(X,y,get_models(1),"test",3,2,1)
    assert len(results)==5*3*2
    assert results.model.nunique()==5
    pair=compare_all_pairs(results,metric="accuracy",rope=0.01)
    assert len(pair)==10
    assert np.allclose(pair[["p_B_better","p_equivalent","p_A_better"]].sum(axis=1),1.0)
