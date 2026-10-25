"""Keep all the sensor data: snapshot data/live/ into the repo, all night (owner: everyone).

data/live/ is not committed (it changes every second). This copies it into folders that are,
so nothing is lost if the laptop dies, and everyone can pull the data (docs/data_contract.md §10):

  data/archive/<date>/run.csv, camera.csv     everything the board and the camera logged
  data/archive/<date>/detections_<rush>.jsonl.gz
                                              camera boxes + fingerprints for each rush (no pixels),
                                              to re-run and tune the tracker later
  data/archive/<date>/predictions.jsonl       every new forecast + recommendation, timestamped
  data/archive/<date>/*.json                  camera zone, ultrasonic calibration, tracker settings
  data/replay/<rush>_run.csv, <rush>_camera.csv
                                              each rush cut out, ready for the demo replay,
                                              model/backtest.py and ingest/camera_accuracy.py
  data/ground_truth/                          hand counts and queue counts (already written there)

<date> is the day the logging started, so one night stays in one folder. Only complete lines are
copied: files being written right now are safe to snapshot.

Usage:
  python ingest/archive.py                         # snapshot once
  python ingest/archive.py --watch 10 --commit     # every 10 min, git commit each snapshot
  python ingest/archive.py --watch 10 --commit --push   # ... and push, so it is off the laptop
"""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest import state as S  # noqa: E402

LIVE = ROOT / "data" / "live"
ARCHIVE = ROOT / "data" / "archive"
REPLAY = ROOT / "data" / "replay"
GROUND_TRUTH = ROOT / "data" / "ground_truth"
CSVS = ("run.csv", "camera.csv")
SMALL_JSON = ("camera_zone.json", "ultrasonic.json", "tracker_params.json")
TS_LEN = 19  # "2026-10-24T18:05:12"


def complete_lines(path: Path) -> list[str]:
    """The file's lines, without a half-written last line."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    lines = text.split("\n")
    return [ln for ln in lines[:-1] if ln.strip()]  # the part after the last newline is unfinished


def first_time(lines: list[str]) -> str | None:
    for ln in lines[1:]:
        if len(ln) >= TS_LEN and ln[4] == "-" and ln[10] == "T":
            return ln[:TS_LEN]
    return None


def rush_windows(t_first: str, t_last: str, schedule: list[dict]) -> list[tuple[str, str, str]]:
    """(name, from, to) of every rush occurrence overlapping the data, as ISO strings (which
    compare in time order). Window: PRE_OPEN_MIN before opening to POST_CLOSE_MIN after closing.
    The latest occurrence of a rush is called <rush>; earlier ones <rush>_<date>."""
    lo, hi = datetime.fromisoformat(t_first), datetime.fromisoformat(t_last)
    found = []
    for r in schedule:
        day = lo.date() - timedelta(days=1)
        while day <= hi.date():
            o = datetime.combine(day, datetime.strptime(r["open"], "%H:%M").time())
            c = datetime.combine(day, datetime.strptime(r["close"], "%H:%M").time())
            c += timedelta(days=1) if c <= o else timedelta(0)  # closes after midnight
            a, b = o - timedelta(minutes=S.PRE_OPEN_MIN), c + timedelta(minutes=S.POST_CLOSE_MIN)
            if a <= hi and b >= lo:
                found.append([r["name"], o, a.isoformat(timespec="seconds"), b.isoformat(timespec="seconds")])
            day += timedelta(days=1)
    out = []
    for name, o, a, b in found:
        later = any(n == name and o2 > o for n, o2, _, _ in found)
        out.append((f"{name}_{o:%Y-%m-%d}" if later else name, a, b))
    return out


def write_if_changed(path: Path, text: str) -> bool:
    try:
        if path.read_text() == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def slice_detections(log: Path, windows: list[tuple[str, str, str]], folder: Path) -> list[str]:
    """detections_<rush>.jsonl.gz per rush window (header lines kept, so the zone is known)."""
    if not log.exists() or not windows:
        return []
    outs: dict[str, list[str]] = {name: [] for name, _, _ in windows}
    header, needs_header = None, {name: True for name in outs}
    with log.open(errors="replace") as f:
        for ln in f:
            if not ln.endswith("\n"):
                break  # half-written
            if '"zone"' in ln:  # camera_count.py (re)started: the next frames belong to this header
                header, needs_header = ln, {name: True for name in outs}
                continue
            t = ln.split('"t":', 1)[-1].lstrip(' "')[:TS_LEN]  # {"t":"2026-10-24T18:05:12.345",...
            for name, a, b in windows:
                if a <= t <= b:
                    if needs_header[name] and header:
                        outs[name].append(header)
                        needs_header[name] = False
                    outs[name].append(ln)
    written = []
    for name, lines in outs.items():
        if len(lines) > 1:
            path = folder / f"detections_{name}.jsonl.gz"
            data = "".join(lines).encode()
            old = None
            try:
                old = gzip.decompress(path.read_bytes())
            except (OSError, EOFError, gzip.BadGzipFile):
                pass
            if old != data:
                path.write_bytes(gzip.compress(data, mtime=0))
                written.append(path.name)
    return written


def log_prediction(folder: Path) -> bool:
    """Append data/live/prediction.json to predictions.jsonl when it is a new one."""
    pred = S.read_json(LIVE / "prediction.json")
    if not pred or not pred.get("generated_at"):
        return False
    path = folder / "predictions.jsonl"
    last = complete_lines(path)[-1:] if path.exists() else []
    if last and json.loads(last[0]).get("generated_at") == pred["generated_at"]:
        return False
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({k: v for k, v in pred.items() if k != "checked_at"}) + "\n")
    return True


def snapshot(live: Path = LIVE, archive: Path = ARCHIVE, replay: Path = REPLAY, schedule=None) -> dict:
    """One snapshot. Returns what changed."""
    files = {name: complete_lines(live / name) for name in CSVS}
    starts = [t for t in (first_time(v) for v in files.values()) if t]
    if not starts:
        return {"folder": None, "changed": []}
    folder = archive / min(starts)[:10]
    changed = []
    for name, lines in files.items():
        if len(lines) > 1 and write_if_changed(folder / name, "\n".join(lines) + "\n"):
            changed.append(name)
    for name in SMALL_JSON:
        if (live / name).exists() and S.read_json(live / name) is not None:
            text = (live / name).read_text()
            if write_if_changed(folder / name, text):
                changed.append(name)
    if log_prediction(folder):
        changed.append("predictions.jsonl")
    lasts = [v[-1][:TS_LEN] for v in files.values() if len(v) > 1]
    windows = rush_windows(min(starts), max(lasts), S.load_schedule() if schedule is None else schedule)
    for rush, a, b in windows:
        for name in CSVS:
            lines = files[name]
            if len(lines) < 2:
                continue
            rows = [ln for ln in lines[1:] if a <= ln[:TS_LEN] <= b]
            if len(rows) >= 2:
                out = replay / f"{rush}_{name.replace('.csv', '')}.csv"
                if write_if_changed(out, "\n".join([lines[0], *rows]) + "\n"):
                    changed.append(str(out.relative_to(replay.parent)))
    changed += slice_detections(live / "detections.jsonl", windows, folder)
    return {"folder": folder, "changed": changed}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def commit(push: bool) -> str:
    paths = [str(p.relative_to(ROOT)) for p in (ARCHIVE, REPLAY, GROUND_TRUTH) if p.exists()]
    git("add", "--", *paths)
    if git("diff", "--cached", "--quiet", "--", *paths).returncode == 0:
        return "nothing new to commit"
    r = git("commit", "-m", f"Sensor data up to {datetime.now():%H:%M}", "--", *paths)
    if r.returncode != 0:
        return f"commit failed: {(r.stderr or r.stdout).strip()}"
    msg = "committed"
    if push:
        p = git("push")
        msg += ", pushed" if p.returncode == 0 else f", push failed (will retry next time): {p.stderr.strip()[:200]}"
    return msg


def size_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6 if folder.exists() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, metavar="MINUTES", help="snapshot every MINUTES")
    ap.add_argument("--commit", action="store_true", help="git commit each snapshot (data folders only)")
    ap.add_argument("--push", action="store_true", help="also git push (needs --commit)")
    args = ap.parse_args()
    if not shutil.which("git") and args.commit:
        raise SystemExit("git not found: run without --commit")
    while True:
        try:
            snap = snapshot()
            if snap["folder"] is None:
                print(f"{datetime.now():%H:%M:%S}  nothing in {LIVE} yet")
            else:
                print(f"{datetime.now():%H:%M:%S}  {snap['folder'].relative_to(ROOT)} ({size_mb(snap['folder']):.1f} MB): "
                      f"{', '.join(snap['changed']) or 'no change'}"
                      + (f" | {commit(args.push)}" if args.commit else ""))
        except OSError as e:
            print(f"snapshot failed, trying again next time: {e}")
        if not args.watch:
            break
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
