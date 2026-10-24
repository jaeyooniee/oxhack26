"""Python tools Gemini can call (owner: S2). All maths lives here, never in the LLM.

  get_queue_state() -> dict          latest data/live/queue_state.json
  get_forecast(horizons) -> list     model.forecast.forecast on that state
  compare_slots(slots, max_wait_min) -> list
      for each slot: expected wait (P50) and P(wait > max_wait_min), interpolated from P10/P50/P90

The state file is QUEUETIME_STATE (env) or data/live/queue_state.json; replay sets STATE directly.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest.state import rush_window  # noqa: E402
from model.forecast import available_horizons, forecast  # noqa: E402

STATE_PATH = Path(os.environ.get("QUEUETIME_STATE", ROOT / "data" / "live" / "queue_state.json"))
STATE: dict | None = None  # set by replay / tests to bypass the file


def get_queue_state() -> dict:
    """Latest live queue state: length, arrival and service rates, current wait estimate."""
    if STATE is not None:
        return STATE
    return json.loads(STATE_PATH.read_text())


def get_forecast(horizons: list[int]) -> list[dict]:
    """Quantile forecasts (P10/P50/P90, minutes) of the wait for each horizon in minutes."""
    have = available_horizons()
    wanted = sorted({int(h) for h in horizons if int(h) in have}) or have
    return forecast(get_queue_state(), wanted)


def prob_exceeds(x: float, p10: float, p50: float, p90: float) -> float:
    """P(wait > x) from a piecewise-linear CDF through the three quantiles (tails extended linearly)."""
    if p90 - p10 < 1e-6:
        return float(x < p50)
    if x <= p50:
        lo = max(p50 - p10, 1e-6)
        cdf = 0.5 - 0.4 * (p50 - x) / lo
    else:
        hi = max(p90 - p50, 1e-6)
        cdf = 0.5 + 0.4 * (x - p50) / hi
    return round(1 - float(np.clip(cdf, 0, 1)), 2)


def slot_forecasts(state: dict, forecasts: list[dict], slots: list[str], max_wait_min: float) -> list[dict]:
    """compare_slots without reading files: interpolate each quantile linearly in time.

    Minute 0 is the wait now (no uncertainty); slots after the longest horizon reuse it."""
    now = datetime.fromisoformat(state["timestamp"])
    fc = sorted(forecasts, key=lambda f: f["horizon_min"])
    w = float(state["wait_now_min"])
    xs = [0] + [f["horizon_min"] for f in fc]
    qs = {q: [w] + [f[q] for f in fc] for q in ("p10", "p50", "p90")}
    out = []
    for slot in slots:
        t = datetime.combine(now.date(), datetime.strptime(slot, "%H:%M").time())
        if now - t > timedelta(hours=12):  # e.g. 00:10 asked at 23:55
            t += timedelta(days=1)
        ahead = (t - now).total_seconds() / 60
        if ahead < -1:
            out.append({"slot": slot, "error": "slot is in the past"})
            continue
        ahead = max(0.0, ahead)
        p = {q: round(float(np.interp(ahead, xs, qs[q])), 1) for q in qs}
        out.append({
            "slot": slot,
            "minutes_ahead": round(ahead),
            "expected_wait_min": p["p50"],
            "p10": p["p10"],
            "p90": p["p90"],
            "prob_wait_over_max": prob_exceeds(max_wait_min, p["p10"], p["p50"], p["p90"]),
            "beyond_forecast_horizon": ahead > xs[-1],
        })
    return out


def compare_slots(slots: list[str], max_wait_min: float) -> list[dict]:
    """For each clock time in `slots` (HH:MM), expected wait and probability it exceeds max_wait_min."""
    state = get_queue_state()
    return slot_forecasts(state, forecast(state), slots, max_wait_min)


def candidate_slots(state: dict, step_min: int = 5, until_min: int | None = None, last_call_min: int = 0) -> list[str]:
    """Clock times from now to the longest horizon, every step_min, stopping last_call_min before
    closing (if sooner). "Now" is always the first slot."""
    now = datetime.fromisoformat(state["timestamp"])
    until = until_min if until_min is not None else max(available_horizons())
    _, close = rush_window(now, state["venue_open"], state["venue_close"])  # handles 23:45-00:45
    until = max(0, min(until, int((close - now).total_seconds() // 60) - last_call_min))
    return [(now + timedelta(minutes=m)).strftime("%H:%M") for m in range(0, until + 1, step_min)]
