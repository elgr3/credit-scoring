import numpy as np
import pandas as pd
import pytest

from credit_scoring import config
from credit_scoring.features import OUTPUT_FEATURES, CreditCleaner


def _row(**overrides):
    base = {c: 1.0 for c in config.RAW_FEATURES}
    base.update(age=40.0, MonthlyIncome=5000.0, NumberOfDependents=1.0, DebtRatio=0.4)
    base.update(overrides)
    return base


@pytest.fixture
def train():
    return pd.DataFrame(
        [_row(age=30.0, MonthlyIncome=3000.0), _row(age=50.0, MonthlyIncome=7000.0)]
    )


def test_missing_income_imputed_with_train_median_and_flagged(train):
    cleaner = CreditCleaner().fit(train)
    out = cleaner.transform(pd.DataFrame([_row(MonthlyIncome=np.nan)]))
    assert out.loc[0, "MonthlyIncome"] == 5000.0
    assert out.loc[0, "MonthlyIncome_missing"] == 1


def test_missing_dependents_set_to_zero_and_flagged(train):
    out = CreditCleaner().fit(train).transform(pd.DataFrame([_row(NumberOfDependents=np.nan)]))
    assert out.loc[0, "NumberOfDependents"] == 0
    assert out.loc[0, "NumberOfDependents_missing"] == 1


@pytest.mark.parametrize("bad_age", [0.0, 12.0])
def test_implausible_age_replaced_by_train_median(train, bad_age):
    out = CreditCleaner().fit(train).transform(pd.DataFrame([_row(age=bad_age)]))
    assert out.loc[0, "age"] == 40.0


def test_late_codes_96_98_flagged_and_zeroed(train):
    row = _row(**{"NumberOfTimes90DaysLate": 98.0, "NumberOfTime30-59DaysPastDueNotWorse": 2.0})
    out = CreditCleaner().fit(train).transform(pd.DataFrame([row]))
    assert out.loc[0, "NumberOfTimes90DaysLate"] == 0
    assert out.loc[0, "late_special_code"] == 1
    assert out.loc[0, "total_late"] == 2 + 1  # 30-59 (2) + 60-89 (1) + 90 (zeroed)


def test_debt_ratio_per_dependent(train):
    row = _row(DebtRatio=0.9, NumberOfDependents=2.0)
    out = CreditCleaner().fit(train).transform(pd.DataFrame([row]))
    assert out.loc[0, "debt_ratio_per_dependent"] == pytest.approx(0.3)


def test_output_columns_and_order_independent_of_input_order(train):
    cleaner = CreditCleaner().fit(train)
    shuffled = pd.DataFrame([_row()])[list(reversed(config.RAW_FEATURES))]
    assert list(cleaner.transform(shuffled).columns) == OUTPUT_FEATURES


def test_fit_uses_train_only(train):
    cleaner = CreditCleaner().fit(train)
    cleaner.transform(pd.DataFrame([_row(MonthlyIncome=1_000_000.0)]))
    assert cleaner.medians_["MonthlyIncome"] == 5000.0


def test_dict_round_trip(train):
    cleaner = CreditCleaner().fit(train)
    clone = CreditCleaner.from_dict(cleaner.to_dict())
    row = pd.DataFrame([_row(MonthlyIncome=np.nan, age=0.0)])
    pd.testing.assert_frame_equal(cleaner.transform(row), clone.transform(row))
