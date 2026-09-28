"""Predict the upcoming race (needs qualifying to have happened)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, data
from .features import add_features

PRED_COLS = ["model", "season", "round", "race_name", "driver_id", "driver_name",
             "constructor_id", "grid", "predicted_score", "predicted_position"]
NEXT_COLS = ["season", "round", "race_name", "circuit_id", "circuit_name", "locality",
             "country", "race_date", "race_time_utc", "status", "generated_at"]


def find_next_race(schedule: list[dict], now: datetime) -> dict | None:
    for r in schedule:
        if r["start"] + timedelta(hours=config.RACE_DURATION_HOURS) >= now:
            return r
    return None


def _info(race: dict | None, status: str) -> dict:
    if race is None:
        return {c: "" for c in NEXT_COLS} | {"status": status}
    return {
        "season": race["season"], "round": race["round"], "race_name": race["race_name"],
        "circuit_id": race["circuit_id"], "circuit_name": race["circuit_name"],
        "locality": race["locality"], "country": race["country"],
        "race_date": race["date"], "race_time_utc": race["time"], "status": status,
        "generated_at": "",
    }


def predict_next_race(hist: pd.DataFrame, factories: dict, now: datetime | None = None):
    """Returns (next_race_info dict, predictions DataFrame).
    status: 'predicted' | 'awaiting_qualifying' | 'no_upcoming_race'."""
    now = now or datetime.now(timezone.utc)
    race = find_next_race(data.fetch_schedule(now.year), now)
    if race is None:
        return _info(None, "no_upcoming_race"), pd.DataFrame(columns=PRED_COLS)

    quali = data.fetch_round_qualifying(race["season"], race["round"])
    if quali.empty:
        return _info(race, "awaiting_qualifying"), pd.DataFrame(columns=PRED_COLS)

    upcoming = pd.DataFrame(
        {
            "season": race["season"], "round": race["round"], "race_name": race["race_name"],
            "date": race["date"], "circuit_id": race["circuit_id"],
            "driver_id": quali["driver_id"], "driver_name": quali["driver_name"],
            "constructor_id": quali["constructor_id"], "grid": quali["quali_position"],
            "position": np.nan, "status": "Unknown", "points": 0.0, "q_best": quali["q_best"],
        }
    )
    done = hist[~((hist["season"] == race["season"]) & (hist["round"] == race["round"]))]
    full = add_features(pd.concat([done, upcoming], ignore_index=True))
    this_race = full[(full["season"] == race["season"]) & (full["round"] == race["round"])]
    train = full[full["race_idx"] < this_race["race_idx"].iloc[0]]

    rows = []
    for name, factory in factories.items():
        model = factory()
        model.fit(train)
        out = this_race[["season", "round", "race_name", "driver_id", "driver_name",
                         "constructor_id", "grid"]].copy()
        out.insert(0, "model", name)
        out["predicted_score"] = np.round(np.asarray(model.predict(this_race), dtype=float), 4)
        out["predicted_position"] = out["predicted_score"].rank(method="first").astype(int)
        rows.append(out.sort_values("predicted_position"))
    return _info(race, "predicted"), pd.concat(rows, ignore_index=True)[PRED_COLS]