import numpy as np

from .base import BaseModel


class GridBaseline(BaseModel):
    """Finishing order = starting order."""
    name = "grid"

    def predict(self, race) -> np.ndarray:
        return race["grid_clean"].to_numpy(dtype=float)


class StandingsBaseline(BaseModel):
    """Finishing order = pre-race championship order."""
    name = "standings"

    def predict(self, race) -> np.ndarray:
        return race["standing_before"].to_numpy(dtype=float)