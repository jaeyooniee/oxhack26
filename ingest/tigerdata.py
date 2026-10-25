"""Stream the queue's time series into Tiger Cloud (TimescaleDB hypertables) (owner: everyone).

Every few seconds, new rows from data/live/ go into one hypertable each:

  camera_counts   camera.csv: people in zone, arrivals / served from the tracker      (docs §3)
  board_readings  run.csv: ultrasonic, beams, hand-count button                       (docs §2)
  queue_states    queue_state.json: queue length, wait now, service rate              (docs §4)
  predictions     prediction.json: forecast P10/P50/P90, recommendation, explanation  (docs §5)
  hand_counts     data/ground_truth/hand_count_<rush>.csv                             (docs §6)
  queue_counts    data/ground_truth/queue_count_<rush>.csv                            (docs §9)

plus a continuous aggregate queue_1min (per-minute queue length and arrivals / served), which
Tiger keeps up to date by itself. Only rows newer than what is already stored are sent, so
restarts and re-runs never duplicate anything. The CSV files stay the source of truth: if the
wifi drops, the next round catches up.

Setup (once):
  1. Tiger Console -> New service -> download the config; copy its connection string into .env:
       TIGER_DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require
  2. pip install "psycopg[binary]"
  3. python ingest/tigerdata.py setup

Usage:
  python ingest/tigerdata.py sync --watch 10     # keep sending new rows (leave running)
  python ingest/tigerdata.py sync                # send once
  python ingest/tigerdata.py query               # last 30 min, per minute, from the database
  python ingest/tigerdata.py sync --live data/archive/2026-10-24 --site replay   # load a recorded night
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest import state as S  # noqa: E402

LIVE = ROOT / "data" / "live"
GROUND_TRUTH = ROOT / "data" / "ground_truth"
SITE = "oxhack-food-point"
HORIZONS = (15, 30, 60)

TABLES = {
    "camera_counts": """
        time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL,
        people_in_zone INT, in_total INT, out_total INT, distinct_total INT, tracked_in_zone INT""",
    "board_readings": """
        time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL,
        dist_cm DOUBLE PRECISION, in_total INT, out_total INT, btn_total INT""",
    "queue_states": """
        time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL, rush TEXT, status TEXT, counts_from TEXT,
        queue_length INT, wait_now_min DOUBLE PRECISION, service_rate_per_min DOUBLE PRECISION,
        arrivals_last_5m INT, served_last_5m INT, state JSONB""",
    "predictions": """
        time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL, state_time TIMESTAMPTZ,
        recommendation TEXT, best_time TEXT, minutes_saved DOUBLE PRECISION, wait_now_min DOUBLE PRECISION,
        """ + ", ".join(f"p{q}_{h} DOUBLE PRECISION" for h in HORIZONS for q in (10, 50, 90)) + """,
        source TEXT, explanation TEXT, prediction JSONB""",
    "hand_counts": "time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL, rush TEXT NOT NULL, event TEXT",
    "queue_counts": "time TIMESTAMPTZ NOT NULL, site TEXT NOT NULL, rush TEXT NOT NULL, people INT",
}
UNIQUE = ("camera_counts", "board_readings", "queue_states", "predictions")  # one row per (site, time)

CAGG = """
CREATE MATERIALIZED VIEW IF NOT EXISTS queue_1min WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', time) AS minute, site,
       avg(tracked_in_zone) AS queue_avg, max(tracked_in_zone) AS queue_max,
       max(people_in_zone) AS heads_max,
       max(in_total) - min(in_total) AS arrived, max(out_total) - min(out_total) AS served
FROM camera_counts GROUP BY minute, site WITH NO DATA"""
CAGG_POLICY = """
SELECT add_continuous_aggregate_policy('queue_1min', start_offset => INTERVAL '1 day',
       end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE)"""


def connect(url: str | None):
    try:
        import psycopg
    except ImportError:
        raise SystemExit('psycopg is not installed: pip install "psycopg[binary]"')
    url = url or os.environ.get("TIGER_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("no connection string: put TIGER_DATABASE_URL=postgres://... in .env "
                         "(from the config file Tiger Console gave you), or pass --url")
    return psycopg.connect(url, connect_timeout=10)


def local_tz(ts) -> datetime:
    """Laptop wall time (naive, local) -> aware datetime, so TIMESTAMPTZ stores the right instant."""
    t = pd.Timestamp(ts).to_pydatetime()
    return t if t.tzinfo else t.astimezone()


# ---------------------------------------------------------------------- setup
def setup(conn) -> list[str]:
    """Create the tables (hypertables on Tiger) and the per-minute continuous aggregate.
    Safe to run again. Returns notes about what was created."""
    notes = []
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
        timescale = cur.fetchone() is not None
    conn.commit()
    for name, cols in TABLES.items():
        with conn.cursor() as cur:
            if timescale:
                try:
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols}) WITH (tsdb.hypertable, "
                                "tsdb.partition_column = 'time', tsdb.segmentby = 'site', tsdb.orderby = 'time DESC')")
                    conn.commit()
                except Exception:  # older TimescaleDB without the WITH (tsdb...) syntax
                    conn.rollback()
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
                    cur.execute("SELECT create_hypertable(%s, 'time', if_not_exists => TRUE)", (name,))
                    conn.commit()
            else:
                cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
                conn.commit()
            if name in UNIQUE:
                cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {name}_site_time ON {name} (site, time)")
            else:
                cur.execute(f"CREATE INDEX IF NOT EXISTS {name}_site_rush_time ON {name} (site, rush, time)")
            conn.commit()
        notes.append(f"{name}: {'hypertable' if timescale else 'plain table (no TimescaleDB here)'}")
    if timescale:
        try:
            conn.autocommit = True  # continuous aggregates cannot be created inside a transaction
            with conn.cursor() as cur:
                cur.execute(CAGG)
                cur.execute(CAGG_POLICY)
            notes.append("queue_1min: continuous aggregate, refreshed every minute")
        except Exception as e:
            notes.append(f"queue_1min: not created ({type(e).__name__}: {str(e).splitlines()[0]})")
        finally:
            conn.autocommit = False
    return notes


# ---------------------------------------------------------------------- rows from the files
def _csv(path: Path, cols: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, parse_dates=["wall_time"])
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if not set(cols) <= set(df):
        return None
    return df.dropna(subset=["wall_time", *cols]).drop_duplicates("wall_time", keep="last")


def camera_rows(live: Path, site: str) -> list[tuple]:
    cols = ["people_in_zone", "in_total", "out_total", "distinct_total", "tracked_in_zone"]
    df = _csv(live / "camera.csv", cols)
    if df is None:
        return []
    return [(local_tz(r.wall_time), site, *(int(getattr(r, c)) for c in cols)) for r in df.itertuples()]


def board_rows(live: Path, site: str) -> list[tuple]:
    df = _csv(live / "run.csv", ["dist_cm", "in_total", "out_total", "btn_total"])
    if df is None:
        return []
    return [(local_tz(r.wall_time), site, float(r.dist_cm), int(r.in_total), int(r.out_total), int(r.btn_total))
            for r in df.itertuples()]


def state_row(live: Path, site: str) -> list[tuple]:
    st = S.read_json(live / "queue_state.json")
    if not isinstance(st, dict) or "timestamp" not in st:
        return []
    return [(local_tz(st["timestamp"]), site, st.get("rush"), st.get("status"), st.get("counts_from"),
             st.get("queue_length"), st.get("wait_now_min"), st.get("service_rate_per_min"),
             st.get("arrivals_last_5m"), st.get("served_last_5m"), json.dumps(st))]


def prediction_rows(live: Path, site: str) -> list[tuple]:
    """The live prediction.json, plus predictions.jsonl when loading an archived night."""
    preds = []
    p = S.read_json(live / "prediction.json")
    if isinstance(p, dict):
        preds.append(p)
    try:
        for line in (live / "predictions.jsonl").read_text().splitlines():
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    rows = {}
    for p in preds:
        when = p.get("generated_at") or p.get("timestamp")
        if not when:
            continue
        fc = {f["horizon_min"]: f for f in p.get("forecasts") or [] if isinstance(f, dict)}
        quantiles = [fc.get(h, {}).get(f"p{q}") for h in HORIZONS for q in (10, 50, 90)]
        rows[when] = (local_tz(when), site, local_tz(p["timestamp"]) if p.get("timestamp") else None,
                      p.get("recommendation"), p.get("best_time"), p.get("minutes_saved"), p.get("wait_now_min"),
                      *quantiles, p.get("source"), p.get("explanation"),
                      json.dumps({k: v for k, v in p.items() if k != "checked_at"}))
    return list(rows.values())


def ground_truth(folder: Path, site: str) -> dict[tuple[str, str], list[tuple]]:
    """{(table, rush): rows} from hand_count_<rush>.csv and queue_count_<rush>.csv."""
    out = {}
    for path in sorted(folder.glob("hand_count_*.csv")):
        df = _csv(path, ["event"]) if path.exists() else None
        rush = path.stem.removeprefix("hand_count_")
        if df is not None:
            out[("hand_counts", rush)] = [(local_tz(r.wall_time), site, rush, r.event) for r in df.itertuples()]
    for path in sorted(folder.glob("queue_count_*.csv")):
        df = _csv(path, ["people"])
        rush = path.stem.removeprefix("queue_count_")
        if df is not None:
            out[("queue_counts", rush)] = [(local_tz(r.wall_time), site, rush, int(r.people)) for r in df.itertuples()]
    return out


# ---------------------------------------------------------------------- sync
def _insert(cur, table: str, rows: list[tuple]) -> None:
    marks = ", ".join(["%s"] * len(rows[0]))
    conflict = " ON CONFLICT (site, time) DO NOTHING" if table in UNIQUE else ""
    cur.executemany(f"INSERT INTO {table} VALUES ({marks}){conflict}", rows)


def sync(conn, live: Path = LIVE, truth: Path = GROUND_TRUTH, site: str = SITE) -> dict:
    """Send what is new. Time-series tables: rows after the newest one stored for this site.
    Ground truth (small, and people may fix it by hand): each rush's rows are replaced."""
    sent = {}
    sources = {"camera_counts": camera_rows, "board_readings": board_rows,
               "queue_states": state_row, "predictions": prediction_rows}
    with conn.cursor() as cur:
        for table, rows_fn in sources.items():
            rows = rows_fn(live, site)
            if not rows:
                continue
            cur.execute(f"SELECT max(time) FROM {table} WHERE site = %s", (site,))
            last = cur.fetchone()[0]
            new = [r for r in rows if last is None or r[0] > last]
            if new:
                _insert(cur, table, new)
                sent[table] = len(new)
        for (table, rush), rows in ground_truth(truth, site).items():
            cur.execute(f"SELECT count(*) FROM {table} WHERE site = %s AND rush = %s", (site, rush))
            if cur.fetchone()[0] == len(rows):
                continue  # unchanged (the files only grow, or are fixed by hand)
            cur.execute(f"DELETE FROM {table} WHERE site = %s AND rush = %s", (site, rush))
            if rows:
                _insert(cur, table, rows)
            sent[f"{table}:{rush}"] = len(rows)
    conn.commit()
    return sent


def query(conn, site: str = SITE, minutes: int = 30) -> pd.DataFrame:
    """Per-minute queue from the database (the continuous aggregate when there is one)."""
    sql_cagg = ("SELECT minute, round(queue_avg::numeric, 1) AS queue, heads_max, arrived, served "
                "FROM queue_1min WHERE site = %s AND minute > (SELECT max(time) FROM camera_counts WHERE site = %s) "
                "- make_interval(mins => %s) ORDER BY minute")
    sql_plain = ("SELECT date_trunc('minute', time) AS minute, round(avg(tracked_in_zone)::numeric, 1) AS queue, "
                 "max(people_in_zone) AS heads_max, max(in_total) - min(in_total) AS arrived, "
                 "max(out_total) - min(out_total) AS served FROM camera_counts WHERE site = %s AND time > "
                 "(SELECT max(time) FROM camera_counts WHERE site = %s) - make_interval(mins => %s) "
                 "GROUP BY 1 ORDER BY 1")
    with conn.cursor() as cur:
        try:
            cur.execute(sql_cagg, (site, site, minutes))
        except Exception:
            conn.rollback()
            cur.execute(sql_plain, (site, site, minutes))
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "sync", "query"])
    ap.add_argument("--url", help="connection string (default: TIGER_DATABASE_URL in .env)")
    ap.add_argument("--site", default=SITE, help="label for this queue, so several can share the database")
    ap.add_argument("--live", type=Path, default=LIVE, help="folder with camera.csv, run.csv, ... (or an archive folder)")
    ap.add_argument("--truth", type=Path, default=GROUND_TRUTH)
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="sync every SECONDS")
    ap.add_argument("--minutes", type=int, default=30, help="query: how far back")
    args = ap.parse_args()

    conn = connect(args.url)
    if args.cmd == "setup":
        for note in setup(conn):
            print(note)
        return
    if args.cmd == "query":
        df = query(conn, args.site, args.minutes)
        print(df.to_string(index=False) if not df.empty else f"no camera rows for site {args.site!r} yet")
        return
    while True:
        try:
            sent = sync(conn, args.live, args.truth, args.site)
            print(f"{datetime.now():%H:%M:%S}  " + (", ".join(f"{k} +{v}" for k, v in sent.items()) or "up to date"))
        except Exception as e:  # wifi drop, server restart: reconnect and catch up next round
            print(f"{datetime.now():%H:%M:%S}  sync failed, retrying next round: {type(e).__name__}: {e}")
            try:
                conn.close()
                conn = connect(args.url)
            except SystemExit:
                raise
            except Exception:
                pass
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
