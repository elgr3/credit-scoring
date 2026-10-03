import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_scoring.config import SEED
from credit_scoring.features import CreditCleaner


def train_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("clean", CreditCleaner()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )
    return pipe.fit(X, y)


def train_xgboost(X: pd.DataFrame, y: pd.Series, n_iter: int = 20) -> tuple[Pipeline, dict]:
    imbalance = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipe = Pipeline(
        [
            ("clean", CreditCleaner()),
            (
                "model",
                XGBClassifier(
                    tree_method="hist",
                    eval_metric="auc",
                    scale_pos_weight=imbalance,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "model__n_estimators": randint(200, 800),
            "model__max_depth": randint(3, 7),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 20),
        },
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
