from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def explain(pipe: Pipeline, X_sample: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Global SHAP summary + one local waterfall; returns features by mean |SHAP|."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    Xc = pipe.named_steps["clean"].transform(X_sample)
    explanation = shap.TreeExplainer(pipe.named_steps["model"])(Xc)

    shap.summary_plot(explanation, Xc, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=120)
    plt.close()

    riskiest = int(np.argmax(pipe.predict_proba(X_sample)[:, 1]))
    shap.plots.waterfall(explanation[riskiest], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_waterfall.png", dpi=120)
    plt.close()

    importance = np.abs(explanation.values).mean(axis=0)
    return [Xc.columns[i] for i in np.argsort(importance)[::-1]]
