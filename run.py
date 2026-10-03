"""Full pipeline: download -> split -> train -> evaluate -> explain -> export."""

from credit_scoring import config
from credit_scoring.data import load_dataset, split
from credit_scoring.evaluate import plot_calibration, plot_roc, score, write_metrics
from credit_scoring.explain import explain
from credit_scoring.export import export
from credit_scoring.train import train_logistic, train_xgboost


def main() -> None:
    print("[1/5] loading data", flush=True)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split(X, y)

    print("[2/5] training logistic baseline", flush=True)
    logistic = train_logistic(X_train, y_train)
    print("[3/5] tuning XGBoost (20 x 5-fold CV)", flush=True)
    xgb, best_params = train_xgboost(X_train, y_train)

    p_log = logistic.predict_proba(X_test)[:, 1]
    p_xgb = xgb.predict_proba(X_test)[:, 1]

    print("[4/5] evaluating and explaining", flush=True)
    plot_roc(
        {"Logistic regression": (y_test, p_log), "XGBoost": (y_test, p_xgb)},
        config.FIGURES_DIR / "roc.png",
    )
    plot_calibration(y_test, p_xgb, config.FIGURES_DIR / "calibration.png")
    ranked = explain(xgb, X_test.sample(2000, random_state=config.SEED), config.FIGURES_DIR)
    export(xgb, config.MODELS_DIR)

    print("[5/5] writing metrics", flush=True)
    write_metrics(
        {
            "dataset": {"n_rows": int(len(X)), "default_rate": round(float(y.mean()), 4)},
            "logistic_regression": score(y_test, p_log),
            "xgboost": score(y_test, p_xgb)
            | {"best_params": {k: round(float(v), 4) for k, v in best_params.items()}},
            "top_features": ranked[:5],
        },
        config.REPORTS_DIR / "metrics.json",
    )
    print((config.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
