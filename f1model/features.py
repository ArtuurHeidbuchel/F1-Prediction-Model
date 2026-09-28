"""Feature engineering. Every feature only uses information available BEFORE the race."""
import pandas as pd

FEATURES_V1 = [
    "grid_clean", "drv_form5", "drv_dnf10", "drv_circuit_avg",
    "team_form5", "pts_before", "standing_before",
]
FEATURES_V2 = FEATURES_V1 + [
    "qgap_pct", "qgap_vs_team", "grid_vs_team", "drv_qgap5",
    "drv_vs_team5", "team_qgap5", "team_pts5",
]
FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
    df["race_idx"] = df.groupby(["season", "round"]).ngroup()
    df = df.sort_values(["race_idx", "position"]).reset_index(drop=True)

    df["grid_clean"] = df["grid"].where(df["grid"] > 0, 20)
    finished = df["status"].str.match(r"^(Finished|\+\d+ Laps?|Lapped)")
    df["dnf"] = (~finished).astype(float)

    by_driver = df.groupby("driver_id")
    df["drv_form5"] = by_driver["position"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_dnf10"] = by_driver["dnf"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["drv_circuit_avg"] = df.groupby(["driver_id", "circuit_id"])["position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Qualifying pace and teammate comparisons
    pole = df.groupby("race_idx")["q_best"].transform("min")
    df["qgap_pct"] = (df["q_best"] / pole - 1) * 100
    by_race_team = df.groupby(["race_idx", "constructor_id"])
    df["qgap_vs_team"] = df["qgap_pct"] - by_race_team["qgap_pct"].transform("mean")
    df["grid_vs_team"] = df["grid_clean"] - by_race_team["grid_clean"].transform("mean")
    df["pos_vs_team"] = df["position"] - by_race_team["position"].transform("mean")
    by_driver = df.groupby("driver_id")  # re-create so new columns are visible
    df["drv_qgap5"] = by_driver["qgap_pct"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_vs_team5"] = by_driver["pos_vs_team"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Championship state going into each race
    df["pts_before"] = (
        df.groupby(["season", "driver_id"])["points"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
    )
    df["standing_before"] = df.groupby("race_idx")["pts_before"].rank(ascending=False, method="min")

    # Constructor form: mean of both cars per race, rolling over past races
    team = (
        df.groupby(["constructor_id", "race_idx"], as_index=False)
        .agg(team_pos=("position", "mean"), team_qgap=("qgap_pct", "mean"), team_pts=("points", "sum"))
        .sort_values(["constructor_id", "race_idx"])
    )
    for src, dst in [("team_pos", "team_form5"), ("team_qgap", "team_qgap5"), ("team_pts", "team_pts5")]:
        team[dst] = team.groupby("constructor_id")[src].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
    return df.merge(
        team[["constructor_id", "race_idx", "team_form5", "team_qgap5", "team_pts5"]],
        on=["constructor_id", "race_idx"], how="left",
    )