from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from credit_scoring.config import DATA_DIR, OPENML_ID, RAW_FEATURES, SEED, TARGET

CACHE_NAME = "give_me_some_credit.parquet"
# OpenML encodes the target as "Yes"/"No"; accept 0/1 too.
TARGET_MAP = {"Yes": 1, "No": 0, "1": 1, "0": 0}


def load_dataset(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and binary target, downloading from OpenML on first call."""
    cache = cache_dir / CACHE_NAME
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        df.to_parquet(cache)
    X = df[RAW_FEATURES].astype(float)
    y = df[TARGET].astype(str).map(TARGET_MAP).astype(int).rename(TARGET)
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
