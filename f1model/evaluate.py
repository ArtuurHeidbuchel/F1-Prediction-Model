import numpy as np
import pandas as pd

from .models import BASELINES


def score_race(race: pd.DataFrame, pred_col: str) -> dict:
    """pred_col: lower value = predicted to finish higher."""
    predicted = race.sort_values(pred_col, kind="stable")["driver_id"].tolist()
    actual = race.sort_values("position")["driver_id"].tolist()
    pred_rank = race[pred_col].rank(method="first")
    nm = dict(zip(race["driver_id"], race["driver_name"]))
    return {
        "predicted_winner": nm[predicted[0]],
        "actual_winner": nm[actual[0]],
        "predicted_podium": "|".join(nm[d] for d in predicted[:3]),
        "actual_podium": "|".join(nm[d] for d in actual[:3]),
        "winner_hit": predicted[0] == actual[0],
        "podium_hit": set(predicted[:3]) == set(actual[:3]),
        "podium_exact": predicted[:3] == actual[:3],
        "spearman": pred_rank.corr(race["position"].rank()),
        "mae_positions": (pred_rank - race["position"]).abs().mean(),
    }


def summarize(races: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model, pooled over all races and per season."""
    scopes = [("all", races)] + [(str(s), races[races["season"] == s]) for s in sorted(races["season"].unique())]
    rows = []
    for scope, sub in scopes:
        for model, g in sub.groupby("model", sort=False):
            rows.append(
                {
                    "model": model, "scope": scope, "races": len(g),
                    "winner_hit_rate": round(g["winner_hit"].mean() * 100, 1),
                    "podium_hit_rate": round(g["podium_hit"].mean() * 100, 1),
                    "podium_exact_rate": round(g["podium_exact"].mean() * 100, 1),
                    "spearman": round(g["spearman"].mean(), 3),
                    "mae_positions": round(g["mae_positions"].mean(), 2),
                }
            )
    return pd.DataFrame(rows)


def compare(races: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """Each non-baseline model minus each baseline, per race, with a 95% bootstrap CI.
    If the interval includes 0, the difference could just be noise."""
    metrics = {"winner_hit": "winner_hit_rate", "podium_hit": "podium_hit_rate",
               "spearman": "spearman", "mae_positions": "mae_positions"}
    models = [m for m in races["model"].unique() if m not in BASELINES]
    rows = []
    for col, label in metrics.items():
        wide = races.assign(v=races[col].astype(float)).pivot_table(
            index=["season", "round"], columns="model", values="v"
        )
        is_rate = col in ("winner_hit", "podium_hit")
        for m in models:
            for b in sorted(BASELINES):
                if b not in wide or m not in wide:
                    continue
                diff = (wide[m] - wide[b]).to_numpy(dtype=float)
                rng = np.random.default_rng(0)
                means = rng.choice(diff, size=(n_boot, len(diff))).mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                k = 100 if is_rate else 1
                rows.append(
                    {"model": m, "baseline": b, "metric": label, "unit": "pp" if is_rate else "",
                     "diff": round(diff.mean() * k, 3), "ci_low": round(lo * k, 3),
                     "ci_high": round(hi * k, 3), "races": len(diff)}
                )
    return pd.DataFrame(rows)