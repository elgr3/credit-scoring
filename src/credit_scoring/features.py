from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_scoring.config import LATE_COLUMNS, RAW_FEATURES

LATE_SPECIAL_CODES = (96, 98)
MIN_ADULT_AGE = 18

OUTPUT_FEATURES = RAW_FEATURES + [
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
    "late_special_code",
    "total_late",
    "debt_ratio_per_dependent",
]


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Imputes, repairs and enriches raw Give Me Some Credit features.

    Medians are learned on the training set only, then reused for any later data
    (test set, API requests), so no information leaks from evaluation data.
    """

    def fit(self, X: pd.DataFrame, y=None) -> CreditCleaner:
        X = X[RAW_FEATURES]
        adult_age = X.loc[X["age"] >= MIN_ADULT_AGE, "age"]
        self.medians_ = {
            "MonthlyIncome": float(X["MonthlyIncome"].median()),
            "age": float(adult_age.median()),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[RAW_FEATURES].astype(float).copy()
        out["MonthlyIncome_missing"] = out["MonthlyIncome"].isna().astype(int)
        out["NumberOfDependents_missing"] = out["NumberOfDependents"].isna().astype(int)
        out["MonthlyIncome"] = out["MonthlyIncome"].fillna(self.medians_["MonthlyIncome"])
        out["NumberOfDependents"] = out["NumberOfDependents"].fillna(0.0)
        out.loc[out["age"].isna() | (out["age"] < MIN_ADULT_AGE), "age"] = self.medians_["age"]

        special = out[LATE_COLUMNS].isin(LATE_SPECIAL_CODES)
        out["late_special_code"] = special.any(axis=1).astype(int)
        out[LATE_COLUMNS] = out[LATE_COLUMNS].mask(special, 0.0).fillna(0.0)
        out["total_late"] = out[LATE_COLUMNS].sum(axis=1)
        out["debt_ratio_per_dependent"] = out["DebtRatio"] / (out["NumberOfDependents"] + 1.0)
        return out[OUTPUT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def to_dict(self) -> dict:
        return {"medians": dict(self.medians_), "output_features": OUTPUT_FEATURES}

    @classmethod
    def from_dict(cls, d: dict) -> CreditCleaner:
        cleaner = cls()
        cleaner.medians_ = {k: float(v) for k, v in d["medians"].items()}
        return cleaner
