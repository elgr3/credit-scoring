import numpy as np
import pandas as pd

from credit_scoring import config
from credit_scoring.export import export, load
from credit_scoring.train import train_xgboost


def test_export_load_round_trip_gives_identical_probabilities(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame({c: rng.random(300) * 10 for c in config.RAW_FEATURES})
    X["age"] = rng.integers(20, 80, 300).astype(float)
    y = pd.Series((X["DebtRatio"] > 5).astype(int))
    pipe, _ = train_xgboost(X, y, n_iter=1)

    export(pipe, tmp_path)
    cleaner, model = load(tmp_path)

    expected = pipe.predict_proba(X)[:, 1]
    got = model.predict_proba(cleaner.transform(X))[:, 1]
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert (tmp_path / "features.json").exists() and (tmp_path / "model.json").exists()
