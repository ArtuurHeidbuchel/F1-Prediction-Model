"""One full run: fetch -> backtest -> predict next race -> write CSVs to output/."""
from datetime import datetime, timezone

import pandas as pd

from . import config, data
from .backtest import run_backtest
from .evaluate import compare, summarize
from .features import add_features
from .io_utils import write_csv
from .models import get_models
from .predict import NEXT_COLS, predict_next_race


def run_pipeline(no_fetch: bool = False, skip_predict: bool = False) -> None:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = config.OUTPUT_DIR

    print("Loading data...")
    raw = data.load_dataset(refresh=not no_fetch, now=now)
    factories = get_models()
    df = add_features(raw)
    seasons = sorted(int(s) for s in df["season"].unique() if s >= config.BACKTEST_START_SEASON)

    print(f"Backtesting {list(factories)} on seasons {seasons}...")
    races, importance = run_backtest(df, factories, seasons)

    races_out = races.copy()
    for c in ("winner_hit", "podium_hit", "podium_exact"):
        races_out[c] = races_out[c].astype(int)
    races_out["spearman"] = races_out["spearman"].round(4)
    races_out["mae_positions"] = races_out["mae_positions"].round(3)
    write_csv(races_out.sort_values(["season", "round", "model"]), out / "backtest_races.csv")

    summary = summarize(races)
    summary["generated_at"] = stamp
    write_csv(summary, out / "backtest_summary.csv", timestamp_col="generated_at")
    write_csv(compare(races), out / "backtest_vs_baselines.csv")

    imp_rows = [
        {"model": m, "feature": f, "importance": round(float(v), 4)}
        for m, s in importance.items()
        for f, v in (s / s.sum()).sort_values(ascending=False).items()
    ]
    write_csv(pd.DataFrame(imp_rows, columns=["model", "feature", "importance"]), out / "feature_importance.csv")

    if skip_predict:
        return
    print("Predicting next race...")
    info, preds = predict_next_race(raw, factories, now)
    info["generated_at"] = stamp
    write_csv(pd.DataFrame([info])[NEXT_COLS], out / "next_race.csv", timestamp_col="generated_at")
    write_csv(preds, out / "predictions.csv")
    print(f"Next race status: {info['status']}")