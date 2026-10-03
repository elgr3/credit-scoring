import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring import config
from credit_scoring.train import train_logistic, train_xgboost


def _data(n=400):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.random(n) * 10 for c in config.RAW_FEATURES})
    X["age"] = rng.integers(20, 80, n).astype(float)
    y = pd.Series((X["RevolvingUtilizationOfUnsecuredLines"] + rng.normal(0, 1, n) > 6).astype(int))
    return X, y


def test_logistic_pipeline_outputs_probabilities():
    X, y = _data()
    proba = train_logistic(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_search_returns_fitted_pipeline_and_params():
    X, y = _data()
    pipe, params = train_xgboost(X, y, n_iter=2)
    assert "model__max_depth" in params
    assert pipe.predict_proba(X).shape == (len(X), 2)


def test_xgboost_learns_signal():
    X, y = _data(800)
    pipe, _ = train_xgboost(X, y, n_iter=2)
    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) > 0.7
