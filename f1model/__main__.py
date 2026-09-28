import argparse

from .pipeline import run_pipeline
from .tune import tune


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m f1model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch data, backtest, predict, write output/*.csv")
    r.add_argument("--no-fetch", action="store_true", help="use cached data/*.csv")
    r.add_argument("--skip-predict", action="store_true", help="only backtest (works offline with --no-fetch)")
    t = sub.add_parser("tune", help="choose XGBoost settings on the tuning seasons")
    t.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        run_pipeline(args.no_fetch, args.skip_predict)
    else:
        tune(args.no_fetch)


if __name__ == "__main__":
    main()