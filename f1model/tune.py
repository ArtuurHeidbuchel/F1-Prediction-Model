"""Pick XGBoost settings using ONLY config.TUNE_SEASONS, and save them to config/.
Run manually (takes a few minutes); the weekly workflow just reads the saved file."""
import json
from functools import partial

from . import config, data
from .backtest import run_backtest
from .features import FEATURE_SETS, add_features
from .models.xgboost_model import CONFIG_PATH, DEFAULT_PARAMS, XGBoostModel

PARAM_GRID = [
    dict(DEFAULT_PARAMS, n_estimators=n, max_depth=d, min_child_weight=w)
    for d in (2, 3) for w in (5, 20) for n in (100, 300)
]


def tune(no_fetch: bool = False) -> None:
    df = add_features(data.load_dataset(refresh=not no_fetch))
    best = None
    for fs in FEATURE_SETS:
        for target in ("position", "delta"):
            for params in PARAM_GRID:
                races, _ = run_backtest(df, {"x": partial(XGBoostModel, fs, target, params)}, config.TUNE_SEASONS)
                score = float(races["spearman"].mean())
                if best is None or score > best["tune_spearman"]:
                    best = {"feature_set": fs, "target": target, "params": params,
                            "tune_spearman": round(score, 4), "tuned_on": config.TUNE_SEASONS}
            print(f"  done {fs}/{target}  best so far: {best['feature_set']}/{best['target']} {best['tune_spearman']}")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(best, indent=2))
    print(f"Saved {CONFIG_PATH}\n{json.dumps(best, indent=2)}")