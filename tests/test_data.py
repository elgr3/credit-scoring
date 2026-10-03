import numpy as np
import pandas as pd

from credit_scoring import config
from credit_scoring.data import load_dataset, split


def _fake_frame(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in config.RAW_FEATURES})
    df[config.TARGET] = pd.Categorical(np.where(rng.random(n) < 0.1, "1", "0"))
    return df


def test_load_dataset_reads_cache_and_casts_target(tmp_path):
    _fake_frame().to_parquet(tmp_path / "give_me_some_credit.parquet")
    X, y = load_dataset(cache_dir=tmp_path)
    assert list(X.columns) == config.RAW_FEATURES
    assert y.dtype == int and set(y.unique()) <= {0, 1}
    assert y.name == config.TARGET


def test_split_is_stratified_and_reproducible(tmp_path):
    _fake_frame(1000).to_parquet(tmp_path / "give_me_some_credit.parquet")
    X, y = load_dataset(cache_dir=tmp_path)
    a = split(X, y)
    b = split(X, y)
    assert len(a[0]) == 800 and len(a[1]) == 200
    assert abs(a[2].mean() - a[3].mean()) < 0.03
    assert a[0].index.equals(b[0].index)


def test_yes_no_target_is_mapped_to_one_zero(tmp_path):
    df = _fake_frame(50)
    df[config.TARGET] = pd.Categorical(np.where(np.arange(50) < 5, "Yes", "No"))
    df.to_parquet(tmp_path / "give_me_some_credit.parquet")
    _, y = load_dataset(cache_dir=tmp_path)
    assert y.sum() == 5 and set(y.unique()) == {0, 1}
