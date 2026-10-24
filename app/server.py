"""QueueTime web app + phone app server (owner: S2). Standard library only.

Live:    python app/server.py
         reads data/live/ (run.csv, queue_state.json, prediction.json) written by logger.py,
         state.py --watch and queue_agent.py --watch.
Replay:  python app/server.py --replay data/sample/fake_rush_run.csv --speed 30
         plays a recorded rush through state -> forecast -> rules (the demo backup).

Then open  http://<laptop-ip>:8000/        web dashboard
           http://<laptop-ip>:8000/phone   phone app (Add to Home Screen to install it)
           /terms and /privacy             the legal pages
The address is printed at start. Phones must be on the same wifi as the laptop.

API (JSON): GET /api/snapshot     everything both apps show, one call every 2 s
            POST /api/replay      {"action": "jump", "minute": 25} | {"action": "pause"} | {"action": "play"}
"""
import argparse
import json
import math
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.queue_agent import MAX_WAIT_MIN, STALE_S, run_once  # noqa: E402
from agent.tools import slot_forecasts  # noqa: E402
from ingest.logger import PREDICTION_MAX_AGE_S, prediction_age_s  # noqa: E402
from ingest.state import (load_camera, load_run, load_schedule, load_ultrasonic, next_rush,  # noqa: E402
                          read_json, resolve_rush, rush_window, state_and_history)

WEB = Path(__file__).resolve().parent / "web"
LIVE = ROOT / "data" / "live"
HISTORY_MIN = 90       # minutes of history sent to the apps
CACHE_S = 1.0          # snapshots are recomputed at most once a second, however many phones poll
STATIC = {             # the only files served: url -> (file under app/web, content type)
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/phone": ("phone.html", "text/html; charset=utf-8"),
    "/phone.html": ("phone.html", "text/html; charset=utf-8"),
    "/qt.js": ("qt.js", "text/javascript; charset=utf-8"),
    "/site.css": ("site.css", "text/css; charset=utf-8"),
    "/terms": ("terms.html", "text/html; charset=utf-8"),
    "/privacy": ("privacy.html", "text/html; charset=utf-8"),
    "/sw.js": ("sw.js", "text/javascript; charset=utf-8"),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/icons/icon.svg": ("icons/icon.svg", "image/svg+xml"),
    "/fonts/IBMPlexSans-Regular.woff2": ("fonts/IBMPlexSans-Regular.woff2", "font/woff2"),
    "/fonts/IBMPlexSans-SemiBold.woff2": ("fonts/IBMPlexSans-SemiBold.woff2", "font/woff2"),
    "/fonts/IBMPlexSans-Bold.woff2": ("fonts/IBMPlexSans-Bold.woff2", "font/woff2"),
    "/fonts/IBMPlexMono-Regular.woff2": ("fonts/IBMPlexMono-Regular.woff2", "font/woff2"),
    "/fonts/IBMPlexMono-Medium.woff2": ("fonts/IBMPlexMono-Medium.woff2", "font/woff2"),
    "/fonts/LICENSE.txt": ("fonts/LICENSE.txt", "text/plain; charset=utf-8"),
    "/icons/icon-192.png": ("icons/icon-192.png", "image/png"),
    "/icons/icon-512.png": ("icons/icon-512.png", "image/png"),
}


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy and pandas scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _schedule(now: datetime, current: str | None) -> list[dict]:
    return [{**r, "current": r.get("name") == current} for r in load_schedule()]


def _history(series: pd.DataFrame) -> list[dict]:
    s = series[series.index >= series.index[-1] - pd.Timedelta(minutes=HISTORY_MIN)]
    return [{"t": t.strftime("%H:%M"), "wait": round(float(r["wait_min"]), 2), "queue": round(float(r["queue_length"]), 1)}
            for t, r in s.iterrows()]


def _best_slot(state: dict | None, pred: dict | None) -> dict | None:
    """The agent's slot forecast for the recommended time: expected wait, P10/P90, P(wait > 5 min)."""
    if not state or not pred or pred.get("recommendation") != "wait" or not pred.get("forecasts"):
        return None
    try:
        slot = slot_forecasts(state, pred["forecasts"], [pred["best_time"]], MAX_WAIT_MIN)[0]
    except (KeyError, ValueError, TypeError):
        return None
    return None if "error" in slot else {**slot, "max_wait_min": MAX_WAIT_MIN}


def _closes_at(state: dict | None) -> str | None:
    """When the current rush closes (ISO, local), so the apps never show slots after it."""
    try:
        _, close = rush_window(datetime.fromisoformat(state["timestamp"]), state["venue_open"], state["venue_close"])
    except (TypeError, KeyError, ValueError):
        return None
    return close.isoformat(timespec="seconds")


def _sensors(df: pd.DataFrame, now: pd.Timestamp) -> dict:
    last = df[df.index <= now].iloc[-1]
    return {"dist_cm": float(last.get("dist_cm", float("nan"))), "in": int(last["in_total"]),
            "out": int(last["out_total"]), "btn": int(last["btn_total"]),
            "last_row": df[df.index <= now].index[-1].strftime("%H:%M:%S"), "board_resets": df.attrs.get("resets", 0)}


class Source:
    """Builds snapshots, live or replay. Thread-safe; cached for CACHE_S."""

    def __init__(self, replay: Path | None = None, speed: float = 30, camera: Path | None = None):
        self.replay, self.speed = replay, speed
        self.lock = threading.Lock()
        self.cached: tuple[float, dict] | None = None
        if replay:
            self.df = load_run(replay)
            guess = replay.with_name(replay.name.replace("_run.csv", "_camera.csv"))
            self.cam = load_camera(camera or guess) if (camera or guess != replay) else None
            self.t0, self.offset_s, self.paused_at = time.monotonic(), 0.0, None
            self.pred_cache: dict = {}

    # ---------------------------------------------------------------- replay clock
    def replay_now(self) -> pd.Timestamp:
        elapsed = (self.paused_at or time.monotonic()) - self.t0
        t = self.df.index[0] + pd.Timedelta(seconds=self.offset_s + elapsed * self.speed)
        return min(self.df.index[-1], t.floor("s"))

    def control(self, cmd: dict) -> dict:
        if not self.replay:
            raise ValueError("replay controls only work in --replay mode")
        with self.lock:
            action = cmd.get("action")
            if action == "jump":
                total = (self.df.index[-1] - self.df.index[0]).total_seconds() / 60
                minute = float(cmd.get("minute", 0))
                if not 0 <= minute <= total:
                    raise ValueError(f"minute must be between 0 and {total:.0f}")
                self.t0, self.offset_s = time.monotonic(), minute * 60
                if self.paused_at:
                    self.paused_at = self.t0
            elif action == "pause" and not self.paused_at:
                self.paused_at = time.monotonic()
            elif action == "play" and self.paused_at:
                self.t0 += time.monotonic() - self.paused_at
                self.paused_at = None
            elif action not in ("pause", "play"):
                raise ValueError(f"unknown action {action!r}")
            self.cached = None
        return {"ok": True, "paused": bool(self.paused_at)}

    # ---------------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        with self.lock:
            if self.cached and time.monotonic() - self.cached[0] < CACHE_S:
                return self.cached[1]
            try:
                snap = self._replay_snapshot() if self.replay else self._live_snapshot()
            except Exception as e:  # a bad file must never take the apps down: say what happened
                snap = {"status": "error", "message": f"{type(e).__name__}: {e}"}
            snap = _clean({"mode": "replay" if self.replay else "live", "server_time": datetime.now().isoformat(timespec="seconds"),
                           "status": "ok", "message": "", "state": None, "prediction": None, "history": [],
                           "schedule": [], "sensors": None, "heartbeat_age_s": None, "replay": None,
                           "best_slot": _best_slot(snap.get("state"), snap.get("prediction")),
                           "closes_at": _closes_at(snap.get("state")), **snap})
            self.cached = (time.monotonic(), snap)
            return snap

    def _replay_snapshot(self) -> dict:
        now = self.replay_now()
        df = self.df[self.df.index <= now]
        schedule = load_schedule()
        r = resolve_rush(now.to_pydatetime(), schedule) if schedule else {"name": None, "open": None, "close": None}
        cam = self.cam[self.cam.index <= now] if self.cam is not None else None
        state, series = state_and_history(df, now, r["open"], r["close"], rush=r["name"],
                                          ultrasonic=load_ultrasonic(), camera=cam)
        if state["status"] == "closed" and schedule:
            state["next_rush"] = next_rush(now.to_pydatetime(), schedule)
        key = state["timestamp"][:16]  # rules are cheap, but no need to redo them within a minute
        if key not in self.pred_cache:
            self.pred_cache = {key: run_once(state, use_gemini=False, out=None)}
        pred = {**self.pred_cache[key], "checked_at": datetime.now().isoformat(timespec="seconds")}
        total = (self.df.index[-1] - self.df.index[0]).total_seconds() / 60
        return {"state": state, "prediction": pred, "history": _history(series), "sensors": _sensors(self.df, now),
                "schedule": _schedule(now.to_pydatetime(), r["name"]), "heartbeat_age_s": 0,
                "replay": {"file": self.replay.name, "speed": self.speed, "paused": bool(self.paused_at),
                           "minute": round((now - self.df.index[0]).total_seconds() / 60, 1), "total_min": round(total)}}

    def _live_snapshot(self) -> dict:
        state = read_json(LIVE / "queue_state.json")
        pred = read_json(LIVE / "prediction.json")
        if state is None:
            return {"status": "no_data", "message": "No queue state yet: start ingest/logger.py, "
                    "ingest/state.py --watch and agent/queue_agent.py --watch.", "prediction": pred,
                    "schedule": _schedule(datetime.now(), None)}
        now = pd.Timestamp(state["timestamp"])
        out = {"state": state, "schedule": _schedule(now.to_pydatetime(), state.get("rush"))}
        age_state = (datetime.now() - now.to_pydatetime()).total_seconds()
        hb = prediction_age_s(pred)
        out["heartbeat_age_s"] = hb
        if age_state > STALE_S:  # same rules as the agent and the LEDs
            out.update(status="stale", message=f"No fresh sensor data for {age_state / 60:.0f} min: is logger.py running?")
            pred = {**(pred or {}), "recommendation": "unknown", "explanation": out["message"]}
        elif hb is None or hb > PREDICTION_MAX_AGE_S:
            out.update(status="stale", message="No update from the agent " + (f"for {hb / 60:.0f} min" if hb else "yet")
                       + ": is agent/queue_agent.py --watch running?")
            pred = {**(pred or {}), "recommendation": "unknown", "explanation": out["message"]}
        out["prediction"] = pred
        try:
            df = load_run(LIVE / "run.csv")
            df = df[df.index <= now]
            _, series = state_and_history(df, now, state["venue_open"], state["venue_close"],
                                          live={} if "smoothing" in state else None,
                                          ultrasonic=load_ultrasonic(), camera=load_camera(LIVE / "camera.csv"))
            out["history"], out["sensors"] = _history(series), _sensors(df, now)
        except (OSError, ValueError, KeyError) as e:
            out["message"] = out.get("message") or f"history unavailable: {e}"
        return out


def make_handler(source: Source):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QueueTime"

        def log_message(self, fmt, *args):  # quiet: the apps poll every 2 s
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/snapshot":
                return self._json(200, source.snapshot())
            if path in STATIC:
                name, ctype = STATIC[path]
                try:
                    body = (WEB / name).read_bytes()
                except OSError:
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, body, ctype, "no-cache")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/replay":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 1000:
                    raise ValueError("body too large")
                cmd = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(cmd, dict):
                    raise ValueError("expected a JSON object")
                self._json(200, source.control(cmd))
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"ok": False, "error": str(e)})

    return Handler


def lan_ip() -> str:
    """This laptop's address on the wifi (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = phones on the wifi can connect; 127.0.0.1 = this laptop only")
    ap.add_argument("--replay", type=Path, help="run.csv of a recorded rush")
    ap.add_argument("--speed", type=float, default=30, help="replay speed (x real time)")
    ap.add_argument("--camera", type=Path, help="camera.csv for the replay (default: <rush>_camera.csv)")
    args = ap.parse_args()
    source = Source(args.replay, args.speed, args.camera)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(source))
    host = lan_ip() if args.host == "0.0.0.0" else args.host
    print(f"QueueTime {'replay of ' + args.replay.name if args.replay else 'live'}")
    print(f"  web:   http://{host}:{args.port}/")
    print(f"  phone: http://{host}:{args.port}/phone   (same wifi; Add to Home Screen to install)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
