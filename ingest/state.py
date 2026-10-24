"""Live queue state (owner: S1 / H1).

Reads data/live/run.csv and writes the latest state to data/live/queue_state.json
(docs/data_contract.md §4).

  arrivals_last_5m = in_total(now) - in_total(now - 5 min)
  served_last_5m   = out_total(now) - out_total(now - 5 min)
  queue_length     = in_total - out_total, minus the beam drift (see below)
  service_rate     = people served per *busy* minute over the last 5 min
  wait_instant_min = queue_length / service_rate (Little's law)
  wait_now_min     = trailing 5-min average of wait_instant_min (the "posted" wait the model uses)

Robustness for a whole night of logging:
- Board resets: a classic Uno reboots whenever a program opens its serial port, and the
  cumulative counters restart at 0. When `ms` goes backwards (or a counter drops) the totals
  before the reset are added back on, so counters stay cumulative.
- Beam drift: the exit beam misses people, so in - out creeps up. A "queue" of DRIFT_MAX people
  or fewer with no arrivals or departures for IDLE_RESET_MIN is drift: reset it to 0.
- Idle server: minutes with nobody queueing don't count toward the service rate, so a quiet
  server doesn't look slow.
- Rushes: each rush (dinner, midnight, breakfast, lunch) has its own opening and closing time in
  data/rushes.json. The rush around `now` is picked automatically and only its data is used
  (from PRE_OPEN_MIN before opening), so "today so far" means "this rush so far".

When nobody was served in the last 5 busy minutes the rate falls back to the last 15, then to
DEFAULT_SERVICE_RATE. The wait is computed on a 1-minute grid ending at `now`, which gives the
wait history (t-15, t-30, t-60), the rush's mean wait and mins_since_change for model/forecast.py.

Usage:
  python ingest/state.py --watch 10                              # times from data/rushes.json
  python ingest/state.py --open 18:00 --close 19:30 --watch 10   # override the schedule
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
from model.live.kalman import DRIFT_MAX, IDLE_RESET_MIN, run_model  # noqa: E402

RUN_CSV = ROOT / "data" / "live" / "run.csv"
STATE_JSON = ROOT / "data" / "live" / "queue_state.json"
SCHEDULE_JSON = ROOT / "data" / "rushes.json"
ULTRASONIC_JSON = ROOT / "data" / "live" / "ultrasonic.json"  # ingest/calibrate_ultrasonic.py

WINDOW_MIN = 5
FALLBACK_WINDOW_MIN = 15
DEFAULT_SERVICE_RATE = 1.0  # people per minute, used before anyone has been served
DEFAULT_RUSH_MIN = 90       # venue_close = venue_open + this when no close time is known
PRE_OPEN_MIN = 30           # a rush's data starts this long before it opens
POST_CLOSE_MIN = 60         # ... and `now` still belongs to that rush this long after it closes
QUEUE_STEP_S = 30           # resolution of the drift-corrected queue series
POSTED_SMOOTH_MIN = 5       # wait_now_min = trailing 5-min average of the instant wait
COUNTERS = ("in_total", "out_total", "btn_total")


def stitch_resets(df: pd.DataFrame) -> pd.DataFrame:
    """Undo board resets: after `ms` goes backwards or a counter drops, add the totals reached
    before the reset to every later row, so the counters stay cumulative."""
    cols = [c for c in COUNTERS if c in df]
    reset = np.zeros(len(df), dtype=bool)
    if "ms" in df:
        ms = df["ms"].to_numpy(dtype=float)
        reset[1:] |= ms[1:] < ms[:-1]
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        reset[1:] |= v[1:] < v[:-1]
    df.attrs["resets"] = int(reset.sum())
    if not reset.any():
        return df
    df = df.copy()
    seg = np.cumsum(reset)
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        carried = pd.Series(v).groupby(seg).last().cumsum().shift(1, fill_value=0)
        df[c] = v + carried.reindex(seg).to_numpy()
    df.attrs["resets"] = int(reset.sum())
    return df


def load_run(path: Path = RUN_CSV) -> pd.DataFrame:
    """run.csv as a DataFrame indexed by wall_time: sorted, duplicates and half-written rows
    dropped, board resets stitched (df.attrs["resets"] = how many)."""
    df = pd.read_csv(path, parse_dates=["wall_time"])
    df = df.dropna(subset=["wall_time", *[c for c in COUNTERS if c in df]]).sort_values("wall_time")
    df = df.drop_duplicates("wall_time", keep="last").set_index("wall_time")
    return stitch_resets(df)


def load_schedule(path: Path = SCHEDULE_JSON) -> list[dict]:
    try:
        return json.loads(path.read_text())["rushes"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return []


def _span(now: datetime, venue_open: str, venue_close: str):
    """(distance in s from now to the rush window, opening, closing) for the nearest occurrence
    of this rush; the window runs PRE_OPEN_MIN before opening to POST_CLOSE_MIN after closing."""
    best = None
    for d in (-1, 0, 1):
        o = datetime.combine(now.date() + timedelta(days=d), datetime.strptime(venue_open, "%H:%M").time())
        c = datetime.combine(o.date(), datetime.strptime(venue_close, "%H:%M").time())
        if c <= o:  # closes after midnight
            c += timedelta(days=1)
        lo, hi = o - timedelta(minutes=PRE_OPEN_MIN), c + timedelta(minutes=POST_CLOSE_MIN)
        dist = 0.0 if lo <= now <= hi else min(abs((now - lo).total_seconds()), abs((now - hi).total_seconds()))
        if best is None or dist < best[0]:
            best = (dist, o, c)
    return best


def rush_window(now: datetime, venue_open: str, venue_close: str) -> tuple[datetime, datetime]:
    """Opening and closing datetimes of the occurrence of this rush nearest to `now`."""
    return _span(now, venue_open, venue_close)[1:]


def resolve_rush(now: datetime, schedule: list[dict]) -> dict | None:
    """The scheduled rush whose window contains `now`, else the nearest one."""
    return min(schedule, key=lambda r: _span(now, r["open"], r["close"])[0], default=None)


def _asof(df: pd.DataFrame, times: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Counter value at each time (last row at or before it; first row for earlier times)."""
    idx = np.clip(df.index.searchsorted(times, side="right") - 1, 0, len(df) - 1)
    return df[col].to_numpy(dtype=float)[idx]


def queue_series(df: pd.DataFrame, served_col: str = "out_total") -> pd.Series:
    """in - served every QUEUE_STEP_S with beam drift removed: the queue is assumed empty when the
    data starts, and a queue of DRIFT_MAX or fewer with no movement for IDLE_RESET_MIN (or a
    negative one) is reset to 0."""
    r = df[["in_total", served_col]].resample(f"{QUEUE_STEP_S}s", label="right", closed="right").last().ffill()
    raw = (r["in_total"] - r[served_col]).to_numpy(dtype=float)
    moved = np.r_[True, (np.diff(r["in_total"].to_numpy()) != 0) | (np.diff(r[served_col].to_numpy()) != 0)]
    q = np.empty(len(raw))
    off, idle_s = (raw[0] if len(raw) else 0.0), 0.0
    for i, x in enumerate(raw):
        idle_s = 0.0 if moved[i] else idle_s + QUEUE_STEP_S
        cur = x - off
        if cur < 0 or (cur <= DRIFT_MAX and idle_s >= IDLE_RESET_MIN * 60):
            off, cur = x, 0.0
        q[i] = cur
    return pd.Series(q, index=r.index)


def _trailing_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of the last n values (fewer at the start) at each position."""
    c = np.cumsum(np.r_[0.0, x])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(0, i - n)]


def wait_series(df: pd.DataFrame, now: datetime | None = None, served_col: str = "out_total",
                freq_min: int = 1) -> pd.DataFrame:
    """Queue length, service rate and wait on a grid ending at `now` (default: last row)."""
    now = pd.Timestamp(now).floor("s") if now is not None else df.index[-1]
    df = df[df.index <= now]
    n = int((now - df.index[0]).total_seconds() // (60 * freq_min))
    times = pd.DatetimeIndex([now - pd.Timedelta(minutes=freq_min * k) for k in range(n, -1, -1)])
    qs = queue_series(df, served_col)
    queue = qs.reindex(qs.index.union(times)).ffill().reindex(times).fillna(0).to_numpy()
    step = np.r_[0.0, np.diff(_asof(df, times, served_col))]
    busy = ((np.r_[0.0, queue[:-1]] > 0) | (step > 0)).astype(float) * freq_min
    rates = []
    for w in (WINDOW_MIN // freq_min, FALLBACK_WINDOW_MIN // freq_min):
        s, b = _trailing_sum(step, w), _trailing_sum(busy, w)
        rates.append(np.where((s > 0) & (b > 0), s / np.maximum(b, 1e-9), np.nan))
    rate = np.where(np.isnan(rates[0]), rates[1], rates[0])
    rate = np.where(np.isnan(rate), DEFAULT_SERVICE_RATE, rate)
    wait = np.where(queue > 0, queue / rate, 0.0)
    out = pd.DataFrame({"queue_length": queue, "service_rate_per_min": rate, "wait_instant_min": wait},
                       index=times)
    out["wait_min"] = posted(out["wait_instant_min"], freq_min)
    return out


def posted(instant: pd.Series, freq_min: int = 1) -> pd.Series:
    """The 'posted' wait: trailing POSTED_SMOOTH_MIN average of the instant wait. Disney's posted
    waits (what the model was trained on) are smoothed by staff; the raw per-minute wait is far
    noisier, which made the forecast bands too narrow (model/backtest.py)."""
    return instant.rolling(max(1, POSTED_SMOOTH_MIN // freq_min), min_periods=1).mean()


def _mins_since_change(waits: pd.Series) -> int:
    """Minutes since the wait (rounded to the nearest 5 min, like Disney's posted wait) changed."""
    rounded = (waits / 5).round().to_numpy()
    changed = np.nonzero(rounded != rounded[-1])[0]
    first_same = changed[-1] + 1 if len(changed) else 0
    return int((waits.index[-1] - waits.index[first_same]).total_seconds() // 60)


def compute_state(run_rows: pd.DataFrame, now: datetime | None = None,
                  venue_open: str | None = None, venue_close: str | None = None,
                  served_col: str = "out_total", is_holiday: int = 0,
                  days_to_holiday: int = 14, live: dict | None = None, rush: str | None = None) -> dict:
    """Queue state at `now` (default: the last row) in the data contract §4 format.

    With venue_open, only this rush's rows are used (from PRE_OPEN_MIN before opening).
    live: None = plain counts. A dict (e.g. {} or {"us_empty_cm": 400, "us_cm_per_person": 60})
    smooths queue length, service rate and wait with model/live/kalman.py and adds 80% intervals."""
    now = pd.Timestamp(now).floor("s") if now is not None else run_rows.index[-1]
    df = run_rows[run_rows.index <= now]
    if venue_open is not None:
        if venue_close is None:
            venue_close = (datetime.strptime(venue_open, "%H:%M")
                           + timedelta(minutes=DEFAULT_RUSH_MIN)).strftime("%H:%M")
        opened, _ = rush_window(now.to_pydatetime(), venue_open, venue_close)
        df = df[df.index >= opened - timedelta(minutes=PRE_OPEN_MIN)]
    if df.empty:
        raise ValueError(f"no rows for this rush at or before {now}")
    series = wait_series(df, now, served_col)
    extra = {}
    if live is not None:
        model, track = run_model(df, now, served_col, grid=series.index, **live)
        summary = model.summary()
        series = track.rename(columns={"wait_min": "wait_instant_min"})
        series.loc[series.index[-1], ["queue_length", "service_rate_per_min", "wait_instant_min"]] = [
            summary["queue_length"], summary["service_rate_per_min"], summary["wait_now_min"]]
        series["wait_min"] = posted(series["wait_instant_min"])
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
    if rush:
        extra["rush"] = rush
    if run_rows.attrs.get("resets"):
        extra["board_resets"] = run_rows.attrs["resets"]
    return {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "queue_length": int(round(last["queue_length"])),
        "arrivals_last_5m": max(0, int(arrivals[0] - arrivals[1])),
        "served_last_5m": max(0, int(served[0] - served[1])),
        "service_rate_per_min": round(float(last["service_rate_per_min"]), 2),
        "wait_now_min": round(float(last["wait_min"]), 1),
        "wait_instant_min": round(float(last["wait_instant_min"]), 1),
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


def write_json(obj: dict, path: Path, retries: int = 20) -> None:
    """Write via a temp file so readers never see half a file. On Windows the swap fails while
    another program has the file open, so retry for up to ~1 s before giving up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(0.05)


def read_json(path: Path) -> dict | None:
    """The JSON in `path`, or None if it is missing, half-written or locked (Windows)."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN_CSV)
    ap.add_argument("--out", type=Path, default=STATE_JSON)
    ap.add_argument("--schedule", type=Path, default=SCHEDULE_JSON, help="rush opening/closing times")
    ap.add_argument("--open", dest="venue_open", help="HH:MM, overrides the schedule")
    ap.add_argument("--close", dest="venue_close", help="HH:MM, overrides the schedule")
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
        cal = read_json(ULTRASONIC_JSON) if not args.us_empty else None
        if cal and cal.get("us_cm_per_person"):
            live = {"us_empty_cm": cal["us_empty_cm"], "us_cm_per_person": cal["us_cm_per_person"]}
            print(f"ultrasonic calibration from {ULTRASONIC_JSON}: {live}")
        elif not args.us_empty:
            print("smoothing with beams only: run ingest/calibrate_ultrasonic.py to correct drift mid-rush")
    schedule = [] if args.venue_open else load_schedule(args.schedule)
    if not args.venue_open and not schedule:
        print(f"WARNING: no --open/--close and no {args.schedule}: using the whole of run.csv as one rush")

    last_rush = None
    while True:
        try:
            if not args.venue_open:  # re-read every cycle, so edited rush times apply without a restart
                schedule = load_schedule(args.schedule) or schedule
            df = load_run(args.run)
            now = pd.Timestamp(args.now) if args.now else df.index[-1]
            r = resolve_rush(now.to_pydatetime(), schedule) if schedule else \
                {"name": None, "open": args.venue_open, "close": args.venue_close}
            if r["name"] != last_rush and r["name"]:
                print(f"rush: {r['name']} ({r['open']}-{r['close']})")
                last_rush = r["name"]
            state = compute_state(df, now, r["open"], r["close"], args.served, live=live, rush=r["name"])
            write_json(state, args.out)
            print(f"{state['timestamp']}  queue {state['queue_length']:>3}  "
                  f"wait {state['wait_now_min']:>5} min  -> {args.out}")
        except (OSError, ValueError, pd.errors.EmptyDataError) as e:
            print(f"waiting for data: {e}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
