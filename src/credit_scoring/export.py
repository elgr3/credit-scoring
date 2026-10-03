import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_scoring import __version__
from credit_scoring.features import CreditCleaner


def export(pipe: Pipeline, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.named_steps["model"].save_model(out_dir / "model.json")
    meta = pipe.named_steps["clean"].to_dict() | {"version": __version__}
    (out_dir / "features.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(out_dir: Path) -> tuple[CreditCleaner, XGBClassifier]:
    meta = json.loads((out_dir / "features.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(out_dir / "model.json")
    return CreditCleaner.from_dict(meta), model
