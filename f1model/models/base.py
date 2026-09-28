import numpy as np
import pandas as pd


class BaseModel:
    """Interface every algorithm implements. To add one: subclass this, then register it in
    models/__init__.py. `predict` returns one score per driver; LOWER = predicted to finish higher."""

    name = "base"

    def fit(self, train: pd.DataFrame) -> None:
        """Train on past races (rows have all features plus the real 'position')."""

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def feature_importances(self) -> pd.Series | None:
        return None