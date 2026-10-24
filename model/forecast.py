"""Forecast service (owner: S1). One call: queue state in, quantile forecasts out.

    from model.forecast import forecast
    forecast(state)  # -> [{"horizon_min": 15, "p10": 2.1, "p50": 4.0, "p90": 7.9}, ...]

`state` is data/live/queue_state.json (docs/data_contract.md §4). Models are loaded from
model/artifacts/h{H}/ (built by model/export.py); every exported horizon is used by default.

CLI:  python model/forecast.py data/sample/queue_state_sample.json
"""
import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
DEFAULT_DAYS_TO_HOLIDAY = 14


def available_horizons() -> list[int]:
    return sorted(int(p.name[1:]) for p in ART.glob("h*") if (p / "meta.json").exists())


@lru_cache(maxsize=None)
def load(horizon: int):
    d = ART / f"h{horizon}"
    meta = json.loads((d / "meta.json").read_text())
    boosters = [lgb.Booster(model_file=str(d / f"q{int(q * 100)}.txt")) for q in meta["quantiles"]]
    return meta, boosters


def _log_ratio(a, b) -> float:
    """log1p(a) - log1p(b), NaN if either value is missing."""
    if a is None or b is None:
        return math.nan
    return math.log1p(a) - math.log1p(b)


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def build_features(state: dict, features: list[str]) -> pd.DataFrame:
    """Map a live queue state onto the model's features (names and order from meta.json)."""
    wait = float(state["wait_now_min"])
    hist = state.get("wait_history_min") or {}
    ts = datetime.fromisoformat(state["timestamp"])
    open_min = _hhmm(state["venue_open"])
    close_min = _hhmm(state["venue_close"])
    if close_min <= open_min:  # closes after midnight
        close_min += 24 * 60
    tod = ts.hour * 60 + ts.minute
    if tod < open_min and close_min > 24 * 60:
        tod += 24 * 60
    row = {
        "r_lag15": _log_ratio(hist.get("t-15"), wait),
        "r_lag30": _log_ratio(hist.get("t-30"), wait),
        "r_lag60": _log_ratio(hist.get("t-60"), wait),
        "rel_today": _log_ratio(wait, state.get("mean_wait_today_min")),
        "rel_week": _log_ratio(wait, state.get("median_wait_last_7d_min")),  # None -> cold start
        "mins_since_change": float(state.get("mins_since_change", 0)),
        "hour": ts.hour + ts.minute / 60,
        "dow": ts.weekday(),
        "mins_since_open": tod - open_min,
        "mins_to_close": close_min - tod,
        "frac_of_day": (tod - open_min) / (close_min - open_min),
        "HOLIDAY": int(state.get("is_holiday", 0)),
        "HOLIDAYPX": float(state.get("days_to_holiday", DEFAULT_DAYS_TO_HOLIDAY)),
    }
    return pd.DataFrame([[row[f] for f in features]], columns=features)


def forecast(state: dict, horizons: list[int] | None = None) -> list[dict]:
    """P10/P50/P90 wait forecasts (minutes) for each horizon."""
    wait = float(state["wait_now_min"])
    out = []
    for h in horizons or available_horizons():
        meta, boosters = load(h)
        x = build_features(state, meta["features"])
        z = sorted(float(b.predict(x)[0]) for b in boosters)
        p10, p50, p90 = (max(0.0, math.expm1(math.log1p(wait) + v)) for v in z)
        out.append({"horizon_min": h, "p10": round(p10, 1), "p50": round(p50, 1), "p90": round(p90, 1)})
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample/queue_state_sample.json"
    print(json.dumps(forecast(json.loads(Path(path).read_text())), indent=2))
