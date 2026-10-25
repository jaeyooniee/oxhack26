"""Backtest the forecast on a recorded rush and calibrate its bands for this queue (owner: S1).

Replays run.csv minute by minute: at each minute t, state -> forecast, then compares with the wait
actually computed at t + h. Reports MAE against "no change" and how often the P10-P90 band
contained the truth.

The Disney bands are right 80% of the time on Disney, not necessarily on a food queue. --save
fits one widening factor k per horizon (split-conformal: the 80th percentile of how far outside
the band the truth fell, in band-widths) and writes model/artifacts/food_calibration.json;
model/forecast.py then widens every band:  p10 = p50 - k (p50 - p10),  p90 = p50 + k (p90 - p50).
Fit it on one rush and check it on another (--check) before trusting it.

Usage:
  python model/backtest.py data/replay/dinner_run.csv                  # report only
  python model/backtest.py data/replay/dinner_run.csv data/replay/midnight_run.csv --save
                                                                       # fit k on both rushes
  python model/backtest.py data/replay/breakfast_run.csv --check       # coverage with saved k

<rush>_camera.csv next to each <rush>_run.csv is picked up: its tracker counts are used in place
of the beams, as ingest/state.py does live (--source beams to ignore the camera).
"""
import argparse
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest.state import compute_state, load_counts, load_schedule, resolve_rush, state_and_history  # noqa: E402
from model import forecast as fc  # noqa: E402

TARGET = 0.8


def run(df, venue_open, venue_close, live=None, start_after_min=10, calibrated=False):
    """One row per (minute, horizon): forecast quantiles, wait now and the actual wait later."""
    end = df.index[-1]
    _, history = state_and_history(df, end, venue_open, venue_close, live=live)
    actual = history["wait_min"]  # same rush window and drift handling as the live state
    horizons = fc.available_horizons()
    rows = []
    t = history.index[0].ceil("min") + timedelta(minutes=start_after_min)
    while t + timedelta(minutes=min(horizons)) <= end:
        state = compute_state(df, t, venue_open, venue_close, live=live)
        if state["status"] == "closed":
            t += timedelta(minutes=1)
            continue
        for f in fc.forecast(state, calibrated=calibrated):
            target = t + timedelta(minutes=f["horizon_min"])
            if target <= end:
                rows.append({"t": t, **f, "wait_now": state["wait_now_min"],
                             "actual": float(actual.asof(target))})
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


def conformal_k(g: pd.DataFrame) -> float:
    """Smallest k such that the widened band covers TARGET of the points (finite-sample quantile)."""
    up = (g["actual"] - g["p50"]) / (g["p90"] - g["p50"]).clip(lower=0.1)
    down = (g["p50"] - g["actual"]) / (g["p50"] - g["p10"]).clip(lower=0.1)
    score = np.maximum(up, down).to_numpy()
    n = len(score)
    q = min(1.0, math.ceil((n + 1) * TARGET) / n)
    return round(max(1.0, float(np.quantile(score, q))), 2)


def report(res: pd.DataFrame) -> dict:
    out = {}
    for h, g in res.groupby("horizon_min"):
        cover = ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()
        out[int(h)] = {"n": len(g), "MAE_model": round((g["p50"] - g["actual"]).abs().mean(), 2),
                       "MAE_no_change": round((g["wait_now"] - g["actual"]).abs().mean(), 2),
                       "coverage_P10_P90": round(float(cover), 2),
                       "mean_band_width": round(float((g["p90"] - g["p10"]).mean()), 1),
                       "k_for_80pct": conformal_k(g)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="one or more recorded rushes (pooled)")
    ap.add_argument("--open", dest="venue_open")
    ap.add_argument("--close", dest="venue_close")
    ap.add_argument("--smooth", action="store_true", help="use the Kalman state, as state.py --smooth")
    ap.add_argument("--save", action="store_true", help=f"fit k and write {fc.CALIBRATION_JSON.name}")
    ap.add_argument("--check", action="store_true", help="report with the saved calibration applied")
    ap.add_argument("--source", default="auto", choices=["auto", "camera", "beams"],
                    help="arrivals / served from the camera tracker (<rush>_camera.csv) or the beams")
    args = ap.parse_args()

    parts, rushes = [], []
    for path in args.runs:
        cam = path.with_name(path.name.replace("_run.csv", "_camera.csv"))
        df = load_counts(path, cam if cam != path else None, args.source)
        if args.venue_open:
            venue_open, venue_close = args.venue_open, args.venue_close
        else:
            r = resolve_rush(df.index[-1].to_pydatetime(), load_schedule())
            venue_open, venue_close = r["open"], r["close"]
            print(f"{path.name}: rush {r['name']} ({venue_open}-{venue_close}) from data/rushes.json, "
                  f"counts from {df.attrs['source']}")
        parts.append(run(df, venue_open, venue_close, {} if args.smooth else None, calibrated=args.check))
        rushes.append(f"{path.name} {venue_open}-{venue_close}")
    rep = report(pd.concat(parts))
    print(f"{'h':>4} {'n':>4} {'MAE model':>10} {'no change':>10} {'coverage':>9} {'width':>6} {'k(80%)':>7}")
    for h, r in rep.items():
        print(f"{h:>4} {r['n']:>4} {r['MAE_model']:>10} {r['MAE_no_change']:>10} "
              f"{r['coverage_P10_P90']:>9} {r['mean_band_width']:>6} {r['k_for_80pct']:>7}")
    if args.save:
        cal = {"source": rushes, "target_coverage": TARGET,
               "k": {str(h): r["k_for_80pct"] for h, r in rep.items()},
               "before": {str(h): r["coverage_P10_P90"] for h, r in rep.items()}}
        fc.CALIBRATION_JSON.write_text(json.dumps(cal, indent=2))
        print(f"saved -> {fc.CALIBRATION_JSON}  (check it on another rush with --check)")


if __name__ == "__main__":
    main()
