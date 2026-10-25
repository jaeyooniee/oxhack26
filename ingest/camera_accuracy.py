"""How well does the camera count people? Measure it, and tune the tracker (owner: H2 / S1).

Ground truth (docs/data_contract.md §6, §9):
  served   the hand-count button (btn_total in run.csv), or data/ground_truth/hand_count_<rush>.csv
           (`tally` below: press Enter for every person served, if the button is not there)
  queue    data/ground_truth/queue_count_<rush>.csv: someone counts the people in the queue now
           and then (`note` below)

The served count is the test for distinct people: if the tracker loses someone behind another
person and gives them a new ID, that person counts twice (one extra arrival, one extra served),
so camera served > hand served. Missed people show as camera served < hand served.

Usage:
  python ingest/camera_accuracy.py report --rush dinner          # camera vs hand count for a rush
  python ingest/camera_accuracy.py report --camera data/replay/dinner_camera.csv --run data/replay/dinner_run.csv
  python ingest/camera_accuracy.py note 7                        # "7 people in the queue now"
  python ingest/camera_accuracy.py tally                         # Enter = one person served
  python ingest/camera_accuracy.py tune --rush dinner            # re-run the tracker on the recorded
                                                                 # detections with other settings
  python ingest/camera_accuracy.py tune --rush dinner --save     # ... and use the best ones from now on
"""
import argparse
import gzip
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest import state as S  # noqa: E402
from ingest import tracker as T  # noqa: E402
from ingest.tracker import Tracker, decode_hist, feet_in_zone  # noqa: E402

LIVE = ROOT / "data" / "live"
GROUND_TRUTH = ROOT / "data" / "ground_truth"
DETECTIONS_LOG = LIVE / "detections.jsonl"
TRACKER_PARAMS_JSON = LIVE / "tracker_params.json"  # read by ingest/camera_count.py at start
BIN_MIN = 5
ROW_S = 3.0           # tracker replays write one row per this many seconds, like camera_count.py
LABEL_MATCH_S = 15    # a queue count is compared with the camera row nearest to it, within this
GRID = {"confirm_hits": [2, 3, 5], "exit_s": [1.5, 3.0, 5.0], "reid_s": [10.0, 30.0, 60.0],
        "reid_sim": [0.5, 0.6, 0.7]}


# ---------------------------------------------------------------------- ground truth
def rush_for(name: str | None, at: datetime | None = None) -> dict | None:
    sched = S.load_schedule()
    if name:
        r = next((r for r in sched if r["name"] == name), None)
        if r is None:
            raise SystemExit(f"no rush called {name!r} in data/rushes.json")
        return r
    return S.resolve_rush(at or datetime.now(), sched) if sched else None


def window(rush: dict | None, ref: pd.Timestamp, t_from=None, t_to=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Time range to score: --from/--to, else the rush's window nearest `ref` (from PRE_OPEN_MIN
    before opening to POST_CLOSE_MIN after closing), else everything."""
    if t_from or t_to:
        return pd.Timestamp(t_from or "1970-01-01"), pd.Timestamp(t_to or "2100-01-01")
    if rush:
        o, c = S.rush_window(ref.to_pydatetime(), rush["open"], rush["close"])
        return (pd.Timestamp(o - timedelta(minutes=S.PRE_OPEN_MIN)),
                pd.Timestamp(c + timedelta(minutes=S.POST_CLOSE_MIN)))
    return pd.Timestamp("1970-01-01"), pd.Timestamp("2100-01-01")


def hand_served(run_path: Path | None, hand_paths: list[Path]) -> pd.Series | None:
    """Cumulative people served by hand, indexed by time: the hand-count CSV(s) if given,
    else the button counter in run.csv. None if neither has any."""
    events = []
    for p in hand_paths:
        try:
            ev = pd.read_csv(p, parse_dates=["wall_time"]).dropna()
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        events.append((ev[ev["event"] == "served"] if "event" in ev else ev)["wall_time"])
    if events:
        t = pd.concat(events).sort_values()
        return pd.Series(np.arange(1, len(t) + 1, dtype=float), index=pd.DatetimeIndex(t))
    if run_path is not None:
        try:
            run = S.load_run(run_path)
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return None
        if "btn_total" in run and run["btn_total"].iloc[-1] > run["btn_total"].iloc[0]:
            return run["btn_total"]
    return None


def load_labels(path: Path | None) -> pd.DataFrame | None:
    try:
        lab = pd.read_csv(path, parse_dates=["wall_time"]).dropna()
    except (OSError, ValueError, TypeError, pd.errors.EmptyDataError):
        return None
    return lab.sort_values("wall_time").set_index("wall_time") if "people" in lab and not lab.empty else None


# ---------------------------------------------------------------------- scoring
def _value_at(series: pd.Series, t: pd.Timestamp) -> float:
    """Cumulative counter at time t (0 before its first row)."""
    v = series[series.index <= t]
    return float(v.iloc[-1]) if len(v) else 0.0


def score(camera: pd.DataFrame, hand: pd.Series | None, labels: pd.DataFrame | None,
          lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Camera vs ground truth inside [lo, hi]. Served is compared over the hand-count window
    (first to last served by hand), so the 20 minutes of button presses are a fair test."""
    cam = camera[(camera.index >= lo) & (camera.index <= hi)]
    out: dict = {"from": str(lo), "to": str(hi), "camera_rows": len(cam)}
    if cam.empty:
        return out
    if S.has_tracker_counts(cam):
        cols = ["in_total", "out_total"] + [c for c in ("distinct_total",) if c in camera]
        cnt = S.stitch_resets(camera[cols].astype(float))  # the whole file: counts before lo are subtracted
        before = lo - pd.Timedelta(seconds=1)
        out["camera_arrivals"] = int(_value_at(cnt["in_total"], hi) - _value_at(cnt["in_total"], before))
        if "distinct_total" in cnt:
            out["camera_distinct"] = int(_value_at(cnt["distinct_total"], hi) - _value_at(cnt["distinct_total"], before))
        if hand is not None:
            h = hand[(hand.index >= lo) & (hand.index <= hi)]
            prev = np.r_[_value_at(hand, lo - pd.Timedelta(seconds=1)), h.to_numpy()[:-1]]
            presses = h.index[h.to_numpy() > prev]
            if len(presses) >= 2:
                a, b = presses[0], presses[-1]
                # first press at a: the served counted from just before it, to the last press
                start = a - pd.Timedelta(seconds=1)
                n_hand = _value_at(hand, b) - _value_at(hand, start)
                n_cam = _value_at(cnt["out_total"], b + pd.Timedelta(seconds=ROW_S)) - _value_at(cnt["out_total"], start)
                out["hand_window"] = [a.strftime("%H:%M:%S"), b.strftime("%H:%M:%S")]
                out["served_hand"], out["served_camera"] = int(n_hand), int(n_cam)
                out["served_error_pct"] = round(100 * (n_cam - n_hand) / n_hand, 1) if n_hand else None
                edges = pd.date_range(start, b + pd.Timedelta(minutes=BIN_MIN), freq=f"{BIN_MIN}min")
                per_bin = [(_value_at(cnt["out_total"], e2) - _value_at(cnt["out_total"], e1),
                            _value_at(hand, e2) - _value_at(hand, e1)) for e1, e2 in zip(edges[:-1], edges[1:])]
                out[f"served_per_{BIN_MIN}min_mae"] = round(float(np.mean([abs(c - t) for c, t in per_bin])), 2)
    if labels is not None:
        lab = labels[(labels.index >= lo) & (labels.index <= hi)]
        rows = []
        for t, r in lab.iterrows():
            i = cam.index.get_indexer([t], method="nearest")[0]
            if abs((cam.index[i] - t).total_seconds()) <= LABEL_MATCH_S:
                rows.append({"true": r["people"], "heads": cam["people_in_zone"].iloc[i],
                             **({"tracked": cam["tracked_in_zone"].iloc[i]} if "tracked_in_zone" in cam else {})})
        if rows:
            d = pd.DataFrame(rows)
            out["queue_counts"] = len(d)
            for col in ("heads", "tracked"):
                if col in d:
                    out[f"queue_mae_{col}"] = round(float((d[col] - d["true"]).abs().mean()), 2)
                    out[f"queue_bias_{col}"] = round(float((d[col] - d["true"]).mean()), 2)
    return out


def explain(rep: dict) -> list[str]:
    lines = [f"window {rep['from'][11:19]} - {rep['to'][11:19]}, {rep['camera_rows']} camera rows"]
    if "camera_arrivals" in rep:
        lines.append(f"camera: {rep['camera_arrivals']} arrivals, {rep.get('camera_distinct', '?')} distinct IDs")
    if "served_hand" in rep:
        e = rep["served_error_pct"]
        verdict = ("" if e is None else " -> about right" if abs(e) <= 10 else
                   " -> counting some people twice (lost behind others, new ID): raise reid_s / lower reid_sim"
                   if e > 0 else " -> missing people: lower confirm_hits, check the zone covers the exit")
        lines.append(f"served {rep['hand_window'][0]}-{rep['hand_window'][1]}: camera {rep['served_camera']} vs "
                     f"hand {rep['served_hand']} ({e:+.1f}%){verdict}")
        lines.append(f"served per {BIN_MIN} min: off by {rep[f'served_per_{BIN_MIN}min_mae']} people on average")
    else:
        lines.append("no hand count in this window: press the button (or run `tally`) for 20 min")
    if "queue_counts" in rep:
        lines.append(f"queue length vs {rep['queue_counts']} hand counts: heads off by {rep['queue_mae_heads']} "
                     f"(bias {rep['queue_bias_heads']:+})" + (f", tracker off by {rep['queue_mae_tracked']} "
                     f"(bias {rep['queue_bias_tracked']:+})" if "queue_mae_tracked" in rep else ""))
    else:
        lines.append("no queue counts in this window: run `note N` now and then")
    return lines


# ---------------------------------------------------------------------- replaying the tracker
def read_log(path: Path):
    """Yield ("start", header) and ("frame", (wall_time, [(box, hist), ...])) from detections.jsonl(.gz)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line
            if "zone" in d:
                yield "start", d
            elif "t" in d:
                yield "frame", (datetime.fromisoformat(d["t"]),
                                [(b[:4], decode_hist(b[4] if len(b) > 4 else "")) for b in d["b"]])


def track_log(path: Path, lo=None, hi=None, **params) -> pd.DataFrame:
    """Run the tracker over a detections log: camera.csv rows (every ROW_S) as a DataFrame.
    Each restart of camera_count.py (a header line) starts a new tracker; counters carry on."""
    rows, tracker, base, next_row, heads = [], None, {"in_total": 0, "out_total": 0, "distinct_total": 0}, None, []
    t0 = None
    for kind, item in read_log(path):
        if kind == "start":
            if tracker is not None:
                c = tracker.counts()
                base = {k: base[k] + c[k] for k in base}
            tracker = Tracker(item["zone"], **params)
            t0, next_row = None, None
            continue
        wall, dets = item
        if tracker is None or (lo is not None and wall < lo) or (hi is not None and wall > hi):
            continue
        t0 = t0 or wall
        c = tracker.update(dets, (wall - t0).total_seconds())
        heads.append(sum(1 for b, _ in dets if feet_in_zone(b, tracker.zone)))
        next_row = next_row or wall + timedelta(seconds=ROW_S)
        if wall >= next_row:
            tot = {k: base[k] + c[k] for k in base}
            rows.append({"wall_time": wall.replace(microsecond=0), "people_in_zone": int(np.median(heads)), **tot,
                         "tracked_in_zone": tot["in_total"] - tot["out_total"]})
            heads, next_row = [], wall + timedelta(seconds=ROW_S)
    if not rows:
        return pd.DataFrame(columns=["people_in_zone", "in_total", "out_total", "distinct_total", "tracked_in_zone"])
    return pd.DataFrame(rows).drop_duplicates("wall_time", keep="last").set_index("wall_time")


def _evaluate(job):
    path, lo, hi, params, hand, labels = job
    cam = track_log(path, lo, hi, **params)
    return params, score(cam, hand, labels, lo, hi)


def _loss(rep: dict) -> float:
    """What tuning minimises: served error (people) + queue-length error, where known."""
    loss = abs(rep["served_camera"] - rep["served_hand"]) if "served_hand" in rep else 0.0
    if "queue_mae_tracked" in rep:
        loss += rep["queue_mae_tracked"] * 2
    return loss


# ---------------------------------------------------------------------- commands
def cmd_report(args) -> None:
    camera = S.load_camera(args.camera)
    if camera is None:
        raise SystemExit(f"no camera rows in {args.camera}")
    rush = rush_for(args.rush, camera.index[-1].to_pydatetime())
    lo, hi = window(rush, camera.index[-1], args.t_from, args.t_to)
    name = (rush or {}).get("name")
    hand = hand_served(args.run, args.hand_count or [p for p in [GROUND_TRUTH / f"hand_count_{name}.csv"] if name and p.exists()])
    labels = load_labels(args.labels or (GROUND_TRUTH / f"queue_count_{name}.csv" if name else None))
    rep = score(camera, hand, labels, lo, hi)
    rep["rush"] = name
    print(f"camera accuracy{' for ' + name if name else ''}:")
    for line in explain(rep):
        print("  " + line)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"-> {args.json}")


def cmd_tune(args) -> None:
    if not args.log.exists():
        raise SystemExit(f"no detections log at {args.log} (camera_count.py writes it unless --no-log)")
    first = next((item[0] for kind, item in read_log(args.log) if kind == "frame"), None)
    if first is None:
        raise SystemExit(f"{args.log} has no frames")
    rush = rush_for(args.rush, first)
    lo, hi = window(rush, pd.Timestamp(first), args.t_from, args.t_to)
    name = (rush or {}).get("name")
    hand = hand_served(args.run, args.hand_count or [p for p in [GROUND_TRUTH / f"hand_count_{name}.csv"] if name and p.exists()])
    labels = load_labels(args.labels or (GROUND_TRUTH / f"queue_count_{name}.csv" if name else None))
    if hand is None and labels is None:
        raise SystemExit("nothing to tune against: needs a hand count (button / tally) or queue counts (note)")
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(args.log, lo.to_pydatetime(), hi.to_pydatetime(), p, hand, labels) for p in combos]
    jobs.append((args.log, lo.to_pydatetime(), hi.to_pydatetime(), {"reid": False}, hand, labels))
    print(f"replaying {args.log.name} with {len(jobs)} tracker settings ...")
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_evaluate, jobs))
    results.sort(key=lambda r: _loss(r[1]))
    defaults = {k: getattr(T, k.upper()) for k in GRID}
    if defaults not in combos:
        results.append(_evaluate((args.log, lo.to_pydatetime(), hi.to_pydatetime(), defaults, hand, labels)))
    default = next(r for r in results if r[0] == defaults)
    no_reid = next(r for r in results if r[0] == {"reid": False})
    print(f"{'settings':<60} {'served cam/hand':>16} {'queue err':>9}")

    def show(label, rep):
        served = f"{rep.get('served_camera', '-')}/{rep.get('served_hand', '-')}"
        print(f"{label:<60} {served:>16} {rep.get('queue_mae_tracked', '-'):>9}")

    for p, rep in results[:args.top]:
        show(", ".join(f"{k}={v}" for k, v in p.items()), rep)
    show("(current defaults)", default[1])
    show("(no re-identification)", no_reid[1])
    if args.save:
        best = results[0][0]
        TRACKER_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
        TRACKER_PARAMS_JSON.write_text(json.dumps({**best, "_tuned_on": f"{args.log.name} {name or ''}".strip(),
                                                   "_score": results[0][1]}, indent=2, default=str))
        print(f"saved -> {TRACKER_PARAMS_JSON} (camera_count.py uses it from its next start; "
              "check it on another rush with `report`)")


def _append(path: Path, row: str, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if new:
            f.write(header + "\n")
        f.write(row + "\n")


def cmd_note(args) -> None:
    rush = rush_for(args.rush)
    path = args.out or GROUND_TRUTH / f"queue_count_{(rush or {}).get('name', 'unscheduled')}.csv"
    _append(path, f"{datetime.now():%Y-%m-%dT%H:%M:%S},{args.people}", "wall_time,people")
    print(f"{args.people} people at {datetime.now():%H:%M:%S} -> {path}")


def cmd_tally(args) -> None:
    rush = rush_for(args.rush)
    path = args.out or GROUND_TRUTH / f"hand_count_{(rush or {}).get('name', 'unscheduled')}.csv"
    print(f"Enter = one person served (q + Enter to stop) -> {path}")
    n = 0
    for line in sys.stdin:
        if line.strip().lower() == "q":
            break
        _append(path, f"{datetime.now():%Y-%m-%dT%H:%M:%S},served", "wall_time,event")
        n += 1
        print(f"  {n} served", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--rush", help="rush name from data/rushes.json (default: the one nearest the data)")
        p.add_argument("--from", dest="t_from", help="ISO time, instead of the rush window")
        p.add_argument("--to", dest="t_to")
        p.add_argument("--run", type=Path, default=LIVE / "run.csv", help="run.csv with the hand-count button")
        p.add_argument("--hand-count", type=Path, nargs="*", help="hand_count_<rush>.csv instead of the button")
        p.add_argument("--labels", type=Path, help="queue_count_<rush>.csv (default from --rush)")
    sub.choices["report"].add_argument("--camera", type=Path, default=LIVE / "camera.csv")
    sub.choices["report"].add_argument("--json", type=Path, help="also write the numbers here")
    sub.choices["tune"].add_argument("--log", type=Path, default=DETECTIONS_LOG, help="detections.jsonl(.gz)")
    sub.choices["tune"].add_argument("--top", type=int, default=8)
    sub.choices["tune"].add_argument("--save", action="store_true", help=f"write the best to {TRACKER_PARAMS_JSON.name}")
    for name in ("note", "tally"):
        p = sub.add_parser(name)
        p.add_argument("--rush", help="default: the rush happening now")
        p.add_argument("--out", type=Path)
    sub.choices["note"].add_argument("people", type=int, help="people in the queue right now")
    args = ap.parse_args()
    {"report": cmd_report, "tune": cmd_tune, "note": cmd_note, "tally": cmd_tally}[args.cmd](args)


if __name__ == "__main__":
    main()
