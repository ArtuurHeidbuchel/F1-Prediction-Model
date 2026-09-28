import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .. import config
from ..features import FEATURE_SETS
from .base import BaseModel

DEFAULT_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
)
CONFIG_PATH = config.CONFIG_DIR / "xgboost_best.json"


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, feature_set: str = "v2", target: str = "position", params: dict | None = None):
        self.feature_set, self.target = feature_set, target
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.features = FEATURE_SETS[feature_set]
        self._model: XGBRegressor | None = None

    @classmethod
    def from_config(cls) -> "XGBoostModel":
        """Use the settings saved by `python -m f1model tune`, or defaults if none exist."""
        if CONFIG_PATH.exists():
            cfg = json.loads(CONFIG_PATH.read_text())
            return cls(cfg["feature_set"], cfg["target"], cfg["params"])
        return cls()

    def fit(self, train: pd.DataFrame) -> None:
        y = train["position"] - train["grid_clean"] if self.target == "delta" else train["position"]
        self._model = XGBRegressor(random_state=42, **self.params).fit(train[self.features], y)

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        pred = self._model.predict(race[self.features])
        return pred + race["grid_clean"].to_numpy() if self.target == "delta" else pred

    def feature_importances(self) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=self.features)