import json

import numpy as np
import pytest

from credit_scoring.evaluate import plot_calibration, plot_roc, score, write_metrics


def test_perfect_ranking():
    m = score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc_roc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = score(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc_roc"] - 1, abs=1e-3)


def test_figures_and_json_are_written(tmp_path):
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.4, 0.9, 0.6, 0.1])
    plot_roc({"model": (y, p)}, tmp_path / "roc.png")
    plot_calibration(y, p, tmp_path / "cal.png")
    write_metrics({"auc": 0.5}, tmp_path / "m.json")
    assert (tmp_path / "roc.png").stat().st_size > 0
    assert (tmp_path / "cal.png").stat().st_size > 0
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8")) == {"auc": 0.5}
