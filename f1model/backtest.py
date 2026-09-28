from collections import defaultdict

import pandas as pd

from . import config
from .evaluate import score_race


def run_backtest(df: pd.DataFrame, factories: dict, seasons) -> tuple[pd.DataFrame, dict]:
    """Walk forward through every race in `seasons`: each model is trained ONLY on earlier
    races, then predicts that race. `df` must already have features (add_features)."""
    records, imps = [], defaultdict(list)
    for season in seasons:
        for rnd in sorted(df.loc[df["season"] == season, "round"].unique()):
            if rnd < config.MIN_ROUND:
                continue
            race = df[(df["season"] == season) & (df["round"] == rnd)]
            train = df[df["race_idx"] < race["race_idx"].iloc[0]]
            for name, factory in factories.items():
                model = factory()
                model.fit(train)
                scored = race.assign(score=model.predict(race))
                records.append(
                    {"model": name, "season": int(season), "round": int(rnd),
                     "race_name": race["race_name"].iloc[0], **score_race(scored, "score")}
                )
                imp = model.feature_importances()
                if imp is not None:
                    imps[name].append(imp)
    importance = {n: pd.concat(v, axis=1).mean(axis=1) for n, v in imps.items()}
    return pd.DataFrame(records), importance