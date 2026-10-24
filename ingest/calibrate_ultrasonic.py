"""Calibrate the ultrasonic at the food point (owner: H1). Run while logger.py is logging.

Step 1, nobody in the queue:            python ingest/calibrate_ultrasonic.py --empty
Step 2, N people standing in the queue: python ingest/calibrate_ultrasonic.py --people 4

Each step takes the median distance over the last --seconds of data/live/run.csv. Step 2 writes
data/live/ultrasonic.json ({"us_empty_cm", "us_cm_per_person"}), which `ingest/state.py --smooth`
picks up automatically; the Kalman filter then uses the ultrasonic to correct beam drift.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from ingest.state import RUN_CSV, ULTRASONIC_JSON as CALIBRATION, load_run  # noqa: E402


def median_distance(run: Path, seconds: int) -> float:
    try:
        df = load_run(run)
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot read {run} ({e}): start ingest/logger.py first")
    d = df[df.index >= df.index[-1] - pd.Timedelta(seconds=seconds)]["dist_cm"]
    d = d[d > 0]  # -1 = no echo
    if len(d) < seconds // 3:
        raise SystemExit(f"only {len(d)} echoes in the last {seconds} s: is the sensor pointing down the queue?")
    return float(d.median())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--empty", action="store_true", help="nobody is queueing right now")
    g.add_argument("--people", type=int, help="this many people are queueing right now")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--run", type=Path, default=RUN_CSV)
    args = ap.parse_args()

    cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    d = median_distance(args.run, args.seconds)
    if args.empty:
        cal = {"us_empty_cm": round(d, 1)}
        print(f"empty queue: {d:.1f} cm. Now put N people in the queue and run --people N.")
    else:
        if "us_empty_cm" not in cal:
            raise SystemExit("run --empty first")
        per = (cal["us_empty_cm"] - d) / args.people
        if per < 5:
            raise SystemExit(f"distance only dropped {cal['us_empty_cm'] - d:.0f} cm for {args.people} people: "
                             "the sensor isn't seeing the queue. Re-aim it and start again.")
        cal["us_cm_per_person"] = round(per, 1)
        print(f"{args.people} people: {d:.1f} cm -> {per:.1f} cm per person")
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(cal, indent=2))
    print(f"saved -> {CALIBRATION}")


if __name__ == "__main__":
    main()
