"""Live queue state (owner: S1 / H1).

Reads data/live/run.csv and writes the latest state to data/live/queue_state.json
(docs/data_contract.md §4).

  arrivals_last_5m = in_total(now) - in_total(now - 5 min)
  served_last_5m   = out_total(now) - out_total(now - 5 min)
  queue_length     = in_total - out_total (clipped at 0)
  service_rate     = served_last_5m / 5          (people per minute)
  wait_now_min     = queue_length / service_rate (Little's law)

When nobody was served in the last 5 min the rate falls back to the last 15 min, then to
DEFAULT_SERVICE_RATE. The wait is computed on a 1-minute grid ending at `now`, which gives the
wait history (t-15, t-30, t-60), today's mean wait and mins_since_change for model/forecast.py.

Usage:
  python ingest/state.py --open 18:00 --close 19:30              # once
  python ingest/state.py --open 18:00 --close 19:30 --watch 10   # every 10 s
  python ingest/state.py --run data/sample/fake_rush_run.csv --now 2026-10-24T18:40:00
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.live.kalman import run_model  # noqa: E402

RUN_CSV = ROOT / "data" / "live" / "run.csv"
STATE_JSON = ROOT / "data" / "live" / "queue_state.json"

WINDOW_MIN = 5
FALLBACK_WINDOW_MIN = 15
DEFAULT_SERVICE_RATE = 1.0  # people per minute, used before anyone has been served
DEFAULT_RUSH_MIN = 90       # venue_close = venue_open + this when --close is not given


def load_run(path: Path = RUN_CSV) -> pd.DataFrame:
    """run.csv as a DataFrame indexed by wall_time (sorted, duplicate times dropped)."""
    df = pd.read_csv(path, parse_dates=["wall_time"])
    df = df.dropna(subset=["wall_time"]).sort_values("wall_time")
    return df.drop_duplicates("wall_time", keep="last").set_index("wall_time")


def _asof(df: pd.DataFrame, times: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Counter value at each time (last row at or before it; first row for earlier times)."""
    idx = np.clip(df.index.searchsorted(times, side="right") - 1, 0, len(df) - 1)
    return df[col].to_numpy(dtype=float)[idx]


def _rate(df, times, served_col, window_min):
    """Served per minute over the window ending at each time; NaN when nobody was served."""
    start = df.index[0]
    from_t = (times - pd.Timedelta(minutes=window_min)).where(
        times - pd.Timedelta(minutes=window_min) > start, start)
    minutes = (times - from_t).total_seconds().to_numpy() / 60
    served = _asof(df, times, served_col) - _asof(df, from_t, served_col)
    ok = (minutes >= 1) & (served > 0)
    return np.where(ok, served / np.where(minutes > 0, minutes, 1), np.nan), served


def wait_series(df: pd.DataFrame, now: datetime | None = None, served_col: str = "out_total",
                freq_min: int = 1) -> pd.DataFrame:
    """Queue length, service rate and wait on a grid ending at `now` (default: last row)."""
    now = pd.Timestamp(now).floor("s") if now is not None else df.index[-1]
    df = df[df.index <= now]
    n = int((now - df.index[0]).total_seconds() // (60 * freq_min))
    times = pd.DatetimeIndex([now - pd.Timedelta(minutes=freq_min * k) for k in range(n, -1, -1)])
    queue = np.maximum(0, _asof(df, times, "in_total") - _asof(df, times, served_col))
    rate, _ = _rate(df, times, served_col, WINDOW_MIN)
    rate_long, _ = _rate(df, times, served_col, FALLBACK_WINDOW_MIN)
    rate = np.where(np.isnan(rate), rate_long, rate)
    rate = np.where(np.isnan(rate), DEFAULT_SERVICE_RATE, rate)
    wait = np.where(queue > 0, queue / rate, 0.0)
    return pd.DataFrame({"queue_length": queue, "service_rate_per_min": rate, "wait_min": wait},
                        index=times)


def _mins_since_change(waits: pd.Series) -> int:
    """Minutes since the wait (rounded to the nearest 5 min, like Disney's posted wait) changed."""
    rounded = (waits / 5).round().to_numpy()
    changed = np.nonzero(rounded != rounded[-1])[0]
    first_same = changed[-1] + 1 if len(changed) else 0
    return int((waits.index[-1] - waits.index[first_same]).total_seconds() // 60)


def compute_state(run_rows: pd.DataFrame, now: datetime | None = None,
                  venue_open: str | None = None, venue_close: str | None = None,
                  served_col: str = "out_total", is_holiday: int = 0,
                  days_to_holiday: int = 14, live: dict | None = None) -> dict:
    """Queue state at `now` (default: the last row) in the data contract §4 format.

    live: None = plain counts. A dict (e.g. {} or {"us_empty_cm": 400, "us_cm_per_person": 60})
    smooths queue length, service rate and wait with model/live/kalman.py and adds 80% intervals."""
    now = pd.Timestamp(now).floor("s") if now is not None else run_rows.index[-1]
    df = run_rows[run_rows.index <= now]
    if df.empty:
        raise ValueError(f"no rows at or before {now}")
    series = wait_series(df, now, served_col)
    extra = {}
    if live is not None:
        model, track = run_model(df, now, served_col, grid=series.index, **live)
        series = track
        summary = model.summary()
        series.iloc[-1] = [summary["queue_length"], summary["service_rate_per_min"], summary["wait_now_min"]]
        extra = {k: summary[k] for k in ("queue_length_sd", "arrival_rate_per_min", "arrival_rate_ci80",
                                         "service_rate_ci80", "wait_now_ci80_min")}
        extra["smoothing"] = "kalman" + ("+ultrasonic" if live.get("us_empty_cm") else "")
    last = series.iloc[-1]
    times = pd.DatetimeIndex([now, now - pd.Timedelta(minutes=WINDOW_MIN)])
    arrivals = _asof(df, times, "in_total")
    served = _asof(df, times, served_col)

    def wait_at(k):
        t = now - pd.Timedelta(minutes=k)
        return round(float(series["wait_min"].asof(t)), 1) if t >= series.index[0] else None

    if venue_open is None:
        venue_open = df.index[0].strftime("%H:%M")
    if venue_close is None:
        venue_close = (datetime.strptime(venue_open, "%H:%M")
                       + timedelta(minutes=DEFAULT_RUSH_MIN)).strftime("%H:%M")
    return {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "queue_length": int(round(last["queue_length"])),
        "arrivals_last_5m": int(arrivals[0] - arrivals[1]),
        "served_last_5m": int(served[0] - served[1]),
        "service_rate_per_min": round(float(last["service_rate_per_min"]), 2),
        "wait_now_min": round(float(last["wait_min"]), 1),
        "wait_history_min": {"t-15": wait_at(15), "t-30": wait_at(30), "t-60": wait_at(60)},
        "mean_wait_today_min": round(float(series["wait_min"].mean()), 1),
        "median_wait_last_7d_min": None,  # no history for a new venue: cold start
        "mins_since_change": _mins_since_change(series["wait_min"]),
        "venue_open": venue_open,
        "venue_close": venue_close,
        "is_holiday": is_holiday,
        "days_to_holiday": days_to_holiday,
        **extra,
    }


def write_json(obj: dict, path: Path) -> None:
    """Write via a temp file so readers (agent, dashboard) never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN_CSV)
    ap.add_argument("--out", type=Path, default=STATE_JSON)
    ap.add_argument("--open", dest="venue_open", help="HH:MM (default: first row of run.csv)")
    ap.add_argument("--close", dest="venue_close", help=f"HH:MM (default: open + {DEFAULT_RUSH_MIN} min)")
    ap.add_argument("--served", default="out_total", choices=["out_total", "btn_total"],
                    help="counter used as 'served' (btn_total = hand count)")
    ap.add_argument("--now", help="ISO time to compute the state at (replay); default: last row")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="recompute every SECONDS")
    ap.add_argument("--smooth", action="store_true", help="Kalman/Bayesian smoothing with error bars")
    ap.add_argument("--us-empty", type=float, help="ultrasonic cm with an empty queue (enables fusion)")
    ap.add_argument("--us-cm-per-person", type=float, help="ultrasonic cm closer per extra person")
    args = ap.parse_args()
    live = None
    if args.smooth or args.us_empty:
        live = {"us_empty_cm": args.us_empty, "us_cm_per_person": args.us_cm_per_person}

    while True:
        try:
            state = compute_state(load_run(args.run), args.now, args.venue_open, args.venue_close,
                                  args.served, live=live)
            write_json(state, args.out)
            print(f"{state['timestamp']}  queue {state['queue_length']:>3}  "
                  f"wait {state['wait_now_min']:>5} min  -> {args.out}")
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as e:
            print(f"waiting for data: {e}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
