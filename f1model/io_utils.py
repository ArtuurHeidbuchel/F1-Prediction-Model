from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path, timestamp_col: str | None = None) -> bool:
    """Write a CSV. If nothing but the timestamp would change, leave the file alone
    (so the workflow doesn't create a commit on every run). Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if timestamp_col and timestamp_col in df.columns and path.exists():
        try:
            old = pd.read_csv(path)
            if timestamp_col in old.columns and len(old):
                probe = df.copy()
                probe[timestamp_col] = old[timestamp_col].iloc[0]
                if probe.to_csv(index=False, lineterminator="\n") == path.read_text():
                    return False
        except Exception:
            pass
    df.to_csv(path, index=False, lineterminator="\n")
    return True