from .baselines import GridBaseline, StandingsBaseline
from .xgboost_model import XGBoostModel

BASELINES = ("grid", "standings")


def get_models() -> dict:
    """name -> zero-argument factory returning a fresh model. Register new algorithms here."""
    return {
        "xgboost": XGBoostModel.from_config,
        "grid": GridBaseline,
        "standings": StandingsBaseline,
    }
