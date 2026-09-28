"""Fetching data from the Jolpica-F1 API (the Ergast successor)."""
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config

PAUSE = 0.3
RESULTS_CSV = config.DATA_DIR / "results.csv"
QUALI_CSV = config.DATA_DIR / "qualifying.csv"


def _get(path: str, params: dict | None = None, retries: int = 4) -> dict:
    url = f"{config.API_BASE}/{path}"
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(3 * 2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()["MRData"]
    resp.raise_for_status()
    raise RuntimeError(f"Request kept failing: {url}")


def _paged(path: str, extract) -> list[dict]:
    rows, offset, limit = [], 0, 100
    while True:
        data = _get(path, {"limit": limit, "offset": offset})
        for race in data["RaceTable"]["Races"]:
            rows.extend(extract(race))
        offset += limit
        if offset >= int(data["total"]):
            break
        time.sleep(PAUSE)
    return rows


def _to_seconds(t: str | None) -> float | None:
    if not t:
        return None
    try:
        if ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        return None


def _result_rows(race: dict) -> list[dict]:
    return [
        {
            "season": int(race["season"]),
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "date": race["date"],
            "circuit_id": race["Circuit"]["circuitId"],
            "driver_id": r["Driver"]["driverId"],
            "driver_name": f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}',
            "constructor_id": r["Constructor"]["constructorId"],
            "grid": int(r["grid"]),  # 0 = pit lane start
            "position": int(r["position"]),
            "status": r["status"],
            "points": float(r["points"]),
        }
        for r in race["Results"]
    ]


def _quali_rows(race: dict) -> list[dict]:
    rows = []
    for q in race["QualifyingResults"]:
        times = [_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        valid = [t for t in times if t is not None]
        rows.append(
            {
                "season": int(race["season"]),
                "round": int(race["round"]),
                "driver_id": q["Driver"]["driverId"],
                "driver_name": f'{q["Driver"]["givenName"]} {q["Driver"]["familyName"]}',
                "constructor_id": q["Constructor"]["constructorId"],
                "quali_position": int(q["position"]),
                "q_best": min(valid) if valid else None,
            }
        )
    return rows


def _fetch_seasons(kind: str, extract, seasons) -> pd.DataFrame:
    rows = []
    for year in seasons:
        print(f"  fetching {kind} {year}...")
        rows.extend(_paged(f"{year}/{kind}.json", extract))
        time.sleep(PAUSE)
    return pd.DataFrame(rows)


def fetch_schedule(year: int) -> list[dict]:
    data = _get(f"{year}/races.json", {"limit": 100})
    out = []
    for r in data["RaceTable"]["Races"]:
        t = r.get("time", "12:00:00Z")
        out.append(
            {
                "season": int(r["season"]),
                "round": int(r["round"]),
                "race_name": r["raceName"],
                "date": r["date"],
                "time": t,
                "start": datetime.fromisoformat(f'{r["date"]}T{t}'.replace("Z", "+00:00")),
                "circuit_id": r["Circuit"]["circuitId"],
                "circuit_name": r["Circuit"]["circuitName"],
                "locality": r["Circuit"]["Location"]["locality"],
                "country": r["Circuit"]["Location"]["country"],
            }
        )
    return sorted(out, key=lambda x: x["round"])


def fetch_round_qualifying(year: int, rnd: int) -> pd.DataFrame:
    """Qualifying for one round. Empty DataFrame if it hasn't happened yet."""
    data = _get(f"{year}/{rnd}/qualifying.json", {"limit": 100})
    rows = []
    for race in data["RaceTable"]["Races"]:
        rows.extend(_quali_rows(race))
    return pd.DataFrame(rows)


def load_dataset(refresh: bool = True, now: datetime | None = None) -> pd.DataFrame:
    """Results + qualifying for FIRST_SEASON..current season, one row per driver per race."""
    now = now or datetime.now(timezone.utc)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if refresh:
        seasons = range(config.FIRST_SEASON, now.year + 1)
        results = _fetch_seasons("results", _result_rows, seasons)
        quali = _fetch_seasons("qualifying", _quali_rows, seasons)
        results.to_csv(RESULTS_CSV, index=False)
        quali.to_csv(QUALI_CSV, index=False)
    else:
        results, quali = pd.read_csv(RESULTS_CSV), pd.read_csv(QUALI_CSV)
    if "driver_name" not in results:
        results["driver_name"] = results["driver_id"].str.replace("_", " ").str.title()
    quali = quali[["season", "round", "driver_id", "q_best"]]
    return results.merge(quali, on=["season", "round", "driver_id"], how="left")