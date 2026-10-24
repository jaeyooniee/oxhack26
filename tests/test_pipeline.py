"""End-to-end checks for the live pipeline. Every scenario from the code reviews is here.

Run:  python tests/test_pipeline.py          (or: python -m pytest tests/, if pytest is installed)

Synthetic sensor logs are built second by second, so the true queue is always known.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import queue_agent as qa  # noqa: E402
from agent import tools  # noqa: E402
from app.alerts import phrase  # noqa: E402
from ingest import state as S  # noqa: E402
from ingest.camera_count import count_in_zone  # noqa: E402
from ingest.logger import csv_row, led_command, parse_line  # noqa: E402
from model import backtest as bt  # noqa: E402
from model import forecast as fc  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="queuetime_tests_"))
DAY = "2026-10-24"
SCHED = [{"name": "dinner", "open": "18:00", "close": "19:30"},
         {"name": "midnight", "open": "23:45", "close": "00:45"},
         {"name": "breakfast", "open": "08:00", "close": "09:00"},
         {"name": "lunch", "open": "12:00", "close": "13:00"}]
EMPTY_CM, CM_PER = 400.0, 25.0
US = {"us_empty_cm": EMPTY_CM, "us_cm_per_person": CM_PER}


def ts(hhmmss: str, day: str = DAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day}T{hhmmss}")


def build(start: str, end: str, events: list, name: str, beam_out_miss: set = frozenset(),
          dist=None) -> pd.DataFrame:
    """One row per second. events: (time, "in"|"serve") happenings; a served person crosses the
    exit beam (unless their index is in beam_out_miss) and the hand-count button.
    dist(t, true_queue) -> ultrasonic cm (default: calibrated geometry)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    ev = sorted((pd.Timestamp(t), k) for t, k in events)
    i_in = i_out = i_btn = 0
    served_n = 0
    rows = ["wall_time,ms,dist_cm,in_total,out_total,btn_total"]
    e = 0
    t = t0
    while t <= t1:
        while e < len(ev) and ev[e][0] <= t:
            if ev[e][1] == "in":
                i_in += 1
            else:
                if served_n not in beam_out_miss:
                    i_out += 1
                i_btn += 1
                served_n += 1
            e += 1
        q = i_in - i_btn
        d = dist(t, q) if dist else EMPTY_CM - CM_PER * q
        rows.append(f"{t:%Y-%m-%dT%H:%M:%S},{int((t - t0).total_seconds() * 1000) + 1000},{d:.1f},{i_in},{i_out},{i_btn}")
        t += pd.Timedelta(seconds=1)
    path = TMP / f"{name}.csv"
    path.write_text("\n".join(rows) + "\n")
    return S.load_run(path)


def state(df, at, live=None, **kw):
    now = pd.Timestamp(at)
    r = S.resolve_rush(now.to_pydatetime(), SCHED)
    st = S.compute_state(df, now, r["open"], r["close"], live=live, rush=r["name"], **kw)
    if st["status"] == "closed":
        st["next_rush"] = S.next_rush(now.to_pydatetime(), SCHED)
    return st


def both(df, at, **kw):
    """Plain and Kalman states at the same time."""
    return state(df, at, **kw), state(df, at, live={}, **kw)


# ---------------------------------------------------------------- review 2: drift reset bug
def test_stationary_pre_rush_queue_is_not_wiped():
    """2 people line up at 17:45 and stand still; first served at 18:05; the other keeps waiting."""
    events = [("2026-10-24T17:45:00", "in"), ("2026-10-24T17:45:20", "in"), ("2026-10-24T18:05:00", "serve")]
    df = build("2026-10-24T17:30:00", "2026-10-24T18:25:00", events, "stationary")
    for at, want in [("17:46:00", 2), ("17:49:00", 2), ("17:55:00", 2), ("18:04:30", 2),
                     ("18:06:00", 1), ("18:15:00", 1), ("18:25:00", 1)]:
        for st in both(df, ts(at)):
            assert st["queue_length"] == want, (at, st.get("smoothing", "plain"), st["queue_length"], want)
            assert st["status"] == "open", (at, st["status"])
    # and without the ultrasonic giving it away (dist reads "empty" the whole time: broken sensor)
    df2 = build("2026-10-24T17:30:00", "2026-10-24T18:25:00", events, "stationary_no_us", dist=lambda t, q: -1)
    for at, want in [("17:55:00", 2), ("18:15:00", 1)]:
        for st in both(df2, ts(at)):
            assert st["queue_length"] == want, (at, "no-echo", st["queue_length"])
    # the ultrasonic is calibrated and sees the 2 people: still kept
    for st in both(df, ts("17:55:00"), ultrasonic=US):
        assert st["queue_length"] == 2


def test_phantom_queue_reset_after_closing():
    """Exit beam missed 3 people: beams say 3 left although nobody is there."""
    events = []
    for k in range(20):
        events.append((f"{DAY}T18:{5 + k:02d}:00", "in"))
        events.append((f"{DAY}T18:{6 + k:02d}:30", "serve"))
    df = build(f"{DAY}T17:30:00", f"{DAY}T20:00:00", events, "phantom", beam_out_miss={2, 7, 12})
    for at in ("19:00:00", "19:29:00"):  # no other sensor: the beams are all we have until closing
        assert state(df, ts(at))["queue_length"] == 3, at
    for st in both(df, ts("19:31:00")):  # closed at 19:30, nobody moved since 18:25: reset
        assert st["queue_length"] == 0 and st["status"] == "open", st  # reset, still in grace period
    for st in both(df, ts("19:50:00")):
        assert st["queue_length"] == 0 and st["status"] == "closed", st


def test_phantom_queue_reset_by_ultrasonic_mid_rush():
    events = []
    for k in range(10):
        events.append((f"{DAY}T18:{5 + k:02d}:00", "in"))
        events.append((f"{DAY}T18:{6 + k:02d}:30", "serve"))
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:40:00", events, "phantom_us", beam_out_miss={1, 4})
    assert state(df, ts("18:30:00"))["queue_length"] == 2  # beams only: phantom stays (documented)
    for st in both(df, ts("18:30:00"), ultrasonic=US):
        assert st["queue_length"] == 0, st  # ultrasonic says empty for > 60 s


def test_phantom_queue_reset_by_camera():
    events = [(f"{DAY}T18:05:00", "in"), (f"{DAY}T18:05:10", "in"), (f"{DAY}T18:06:00", "serve"),
              (f"{DAY}T18:07:00", "serve")]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:30:00", events, "phantom_cam", beam_out_miss={0, 1})
    grid = pd.date_range(f"{DAY}T17:30:00", f"{DAY}T18:30:00", freq="3s")
    # a working camera: sees the 2 people while they are there (18:05-18:07), then nobody
    cam_empty = pd.DataFrame({"people_in_zone": ((grid >= ts("18:05:00")) & (grid < ts("18:07:00"))) * 2}, index=grid)
    cam_two = cam_empty.assign(people_in_zone=2)
    assert state(df, ts("18:20:00"))["queue_length"] == 2
    for st in both(df, ts("18:20:00"), camera=cam_empty):
        assert st["queue_length"] == 0, st
    for st in both(df, ts("18:20:00"), camera=cam_two):
        assert st["queue_length"] == 2, st
    # camera only just started seeing nobody (< 2 min): no reset yet
    cam_late = cam_two.copy()
    cam_late.loc[cam_late.index >= ts("18:19:00"), "people_in_zone"] = 0
    assert state(df, ts("18:20:00"), camera=cam_late)["queue_length"] == 2


def test_big_queue_never_reset_by_one_sensor():
    """A glitchy ultrasonic saying 'empty' must not wipe a real queue of 10."""
    events = [(f"{DAY}T18:05:{k:02d}", "in") for k in range(10)]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:20:00", events, "big", dist=lambda t, q: EMPTY_CM)
    for st in both(df, ts("18:20:00"), ultrasonic=US):
        assert st["queue_length"] == 10, st


def test_dead_or_misaimed_sensors_never_wipe_a_queue():
    """Ultrasonic stuck at the 'empty' distance and a camera that never sees anyone: a real
    stationary queue of 2 must survive, even with both 'saying' empty for a long time."""
    events = [(f"{DAY}T17:45:00", "in"), (f"{DAY}T17:45:20", "in")]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:20:00", events, "dead", dist=lambda t, q: EMPTY_CM)
    blind = pd.DataFrame({"people_in_zone": 0}, index=pd.date_range(f"{DAY}T17:30:00", f"{DAY}T18:20:00", freq="3s"))
    for at in ("17:50:00", "18:00:00", "18:20:00"):
        for st in both(df, ts(at), ultrasonic=US, camera=blind):
            assert st["queue_length"] == 2, (at, st)


def test_arrival_right_after_sensor_empty_is_kept():
    """Working ultrasonic, queue empty for a while, then 1 person walks in: not wiped as drift."""
    events = [(f"{DAY}T18:05:00", "in"), (f"{DAY}T18:06:00", "serve"), (f"{DAY}T18:20:00", "in")]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:21:30", events, "fresh")
    for at in ("18:20:10", "18:20:40", "18:21:30"):
        for st in both(df, ts(at), ultrasonic=US):
            assert st["queue_length"] == 1, (at, st)


# ---------------------------------------------------------------- review 2: smoothing lag
def test_wait_follows_a_clearing_queue_within_two_minutes():
    events = [(f"{DAY}T18:0{k // 10}:{(k % 10) * 6:02d}", "in") for k in range(20)]       # 20 by 18:02
    events += [(f"{DAY}T18:{10 + k // 2:02d}:{(k % 2) * 30:02d}", "serve") for k in range(20)]  # 2/min, done 18:19:30
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:30:00", events, "clearing")
    st = state(df, ts("18:12:00"))
    assert st["wait_now_min"] > 5, st
    for at in ("18:21:30", "18:25:00"):
        for st in both(df, ts(at)):
            assert st["wait_instant_min"] == 0 and st["wait_now_min"] == 0, (at, st)


def test_idle_server_does_not_look_slow():
    """2 served quickly at 18:10, nobody for 10 min, then 1 person arrives: no inflated wait."""
    events = [(f"{DAY}T18:09:50", "in"), (f"{DAY}T18:09:55", "in"), (f"{DAY}T18:10:00", "serve"),
              (f"{DAY}T18:10:20", "serve"), (f"{DAY}T18:20:00", "in")]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:21:00", events, "idle")
    st = state(df, ts("18:21:00"))
    # 2 served in 20 s: anything near a minute is right; counting the idle 10 min gave 7.5
    assert st["queue_length"] == 1 and st["wait_instant_min"] <= 2.0, st


# ---------------------------------------------------------------- review 1: resets, rushes, files
def test_board_reset_is_stitched():
    events = [(f"{DAY}T18:{5 + k // 3:02d}:{(k % 3) * 20:02d}", "in") for k in range(30)]
    events += [(f"{DAY}T18:{10 + k:02d}:15", "serve") for k in range(20)]
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:40:00", events, "noreset")
    raw = pd.read_csv(TMP / "noreset.csv", parse_dates=["wall_time"])
    for at in ("18:20:00", "18:30:00"):  # two resets
        m = raw.wall_time >= ts(at)
        base = raw[m].iloc[0]
        for c in ("ms", "in_total", "out_total", "btn_total"):
            raw.loc[m, c] = raw.loc[m, c] - base[c]
    raw.to_csv(TMP / "reset.csv", index=False)
    df2 = S.load_run(TMP / "reset.csv")
    assert df2.attrs["resets"] == 2
    for at in ("18:22:00", "18:35:00", "18:40:00"):
        a, b = state(df, ts(at)), state(df2, ts(at))
        for k in ("queue_length", "arrivals_last_5m", "served_last_5m", "wait_now_min"):
            assert a[k] == b[k], (at, k, a[k], b[k])
        assert b["arrivals_last_5m"] >= 0 and b["served_last_5m"] >= 0


def test_half_written_and_empty_files():
    p = TMP / "half.csv"
    p.write_text("wall_time,ms,dist_cm,in_total,out_total,btn_total\n"
                 "2026-10-24T18:00:00,1000,400.0,0,0,0\n2026-10-24T18:00:01,2000,400.0,1,0,0\n"
                 "2026-10-24T18:00:02,3000,39")
    df = S.load_run(p)
    assert len(df) == 2 and df["in_total"].iloc[-1] == 1
    (TMP / "header.csv").write_text("wall_time,ms,dist_cm,in_total,out_total,btn_total\n")
    try:
        S.load_run(TMP / "header.csv")
        raise AssertionError("expected ValueError for a header-only file")
    except ValueError:
        pass
    assert S.load_camera(TMP / "missing.csv") is None
    (TMP / "cam_header.csv").write_text("wall_time,people_in_zone\n")
    assert S.load_camera(TMP / "cam_header.csv") is None


def test_each_rush_uses_its_own_times_and_data():
    ev = [(f"{DAY}T18:{5 + k:02d}:00", "in") for k in range(10)] + \
         [(f"{DAY}T18:{6 + k:02d}:00", "serve") for k in range(10)]
    ev += [("2026-10-24T23:50:00", "in"), ("2026-10-24T23:50:30", "in"), ("2026-10-25T00:05:00", "in")]
    df = build(f"{DAY}T17:30:00", "2026-10-25T00:20:00", ev, "night", beam_out_miss={3})
    a = state(df, ts("18:40:00"))
    assert (a["rush"], a["venue_open"], a["venue_close"]) == ("dinner", "18:00", "19:30"), a
    b = state(df, pd.Timestamp("2026-10-25T00:10:00"))
    assert (b["rush"], b["venue_open"], b["venue_close"]) == ("midnight", "23:45", "00:45"), b
    assert b["queue_length"] == 3, b  # dinner's missed exit does not leak into midnight
    assert b["status"] == "open"


def test_closed_between_rushes():
    ev = [(f"{DAY}T18:05:00", "in"), (f"{DAY}T18:06:00", "serve")]
    df = build(f"{DAY}T17:30:00", f"{DAY}T23:30:00", ev, "evening")
    for at in ("20:10:00", "21:30:00", "23:00:00"):
        st = state(df, ts(at))
        assert st["status"] == "closed" and st["queue_length"] == 0, (at, st)
        assert st["next_rush"] == {"name": "midnight", "open": "23:45"}, st
        pred = qa.run_once(st, use_gemini=False, out=None)
        assert pred["recommendation"] == "closed" and pred["forecasts"] == [], pred
        assert "midnight at 23:45" in pred["explanation"], pred
        assert led_command(pred) == "O" and phrase(pred) is None
    for at in ("17:31:00", "18:00:00", "19:44:00"):  # pre-open window and closing grace
        assert state(df, ts(at))["status"] == "open", at


def test_people_still_queueing_after_close_keep_it_open():
    ev = [(f"{DAY}T19:2{k}:00", "in") for k in range(5)]
    df = build(f"{DAY}T17:30:00", f"{DAY}T19:55:00", ev, "late")
    st = state(df, ts("19:55:00"))
    assert st["queue_length"] == 5 and st["status"] == "open", st


def test_no_schedule_means_always_open():
    df = build(f"{DAY}T17:30:00", f"{DAY}T21:00:00", [], "nosched")
    st = S.compute_state(df, ts("21:00:00"))
    assert st["status"] == "open"


def test_slots_stop_at_closing_even_across_midnight():
    s = lambda t, o, c: {"timestamp": t, "venue_open": o, "venue_close": c}  # noqa: E731
    assert tools.candidate_slots(s("2026-10-24T23:55:00", "23:45", "00:30"))[-1] == "00:30"
    assert tools.candidate_slots(s("2026-10-25T00:10:00", "23:45", "00:45"))[-1] == "00:45"
    assert tools.candidate_slots(s("2026-10-24T18:05:00", "18:00", "19:30"))[-1] == "19:05"
    assert tools.candidate_slots(s("2026-10-24T19:45:00", "18:00", "19:30")) == ["19:45"]


def test_next_rush_and_resolve():
    assert S.next_rush(datetime(2026, 10, 24, 20, 0), SCHED) == {"name": "midnight", "open": "23:45"}
    assert S.next_rush(datetime(2026, 10, 25, 1, 0), SCHED) == {"name": "breakfast", "open": "08:00"}
    assert S.resolve_rush(datetime(2026, 10, 25, 0, 30), SCHED)["name"] == "midnight"
    assert S.resolve_rush(datetime(2026, 10, 24, 17, 35), SCHED)["name"] == "dinner"
    assert S.load_schedule(TMP / "nope.json") == []
    (TMP / "bad.json").write_text("{not json")
    assert S.load_schedule(TMP / "bad.json") == []


# ---------------------------------------------------------------- files, LEDs, heartbeat
def test_write_json_retries_when_locked_and_readers_survive():
    target = TMP / "locked.json"
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("locked")
        return real(a, b)
    S.os.replace = flaky
    try:
        S.write_json({"a": 1}, target)
    finally:
        S.os.replace = real
    assert json.loads(target.read_text()) == {"a": 1} and calls["n"] == 4
    S.os.replace = lambda a, b: (_ for _ in ()).throw(PermissionError("locked"))
    try:
        S.write_json({"a": 2}, target, retries=3)
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass
    finally:
        S.os.replace = real
    orig = Path.read_text
    Path.read_text = lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("locked"))
    try:
        assert S.read_json(target) is None
        from ingest.logger import read_prediction
        assert read_prediction(target) is None
    finally:
        Path.read_text = orig


def test_led_staleness():
    now = datetime(2026, 10, 24, 20, 0, 0)
    ago = lambda s: (now - timedelta(seconds=s)).isoformat(timespec="seconds")  # noqa: E731
    assert led_command({"recommendation": "go_now", "checked_at": ago(10)}, now) == "G"
    assert led_command({"recommendation": "wait", "generated_at": ago(30)}, now) == "A"
    assert led_command({"recommendation": "wait", "checked_at": ago(120)}, now) == "O"
    assert led_command({"recommendation": "closed", "checked_at": ago(1)}, now) == "O"
    assert led_command({"recommendation": "wait"}, now) == "O"
    assert led_command({"recommendation": "wait", "checked_at": "garbage"}, now) == "O"
    assert led_command(None, now) == "O"


def test_agent_heartbeat_and_stale_state():
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:10:00", [(f"{DAY}T18:05:00", "in")], "hb")
    st = state(df, ts("18:10:00"))
    st["timestamp"] = datetime.now().isoformat(timespec="seconds")  # pretend it's live
    sp, pp = TMP / "hb_state.json", TMP / "hb_pred.json"
    S.write_json(st, sp)
    proc = subprocess.Popen([sys.executable, str(ROOT / "agent/queue_agent.py"), "--state", str(sp),
                             "--out", str(pp), "--watch", "1", "--no-gemini"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            p = S.read_json(pp)
            if p and p["checked_at"] > p["generated_at"]:
                break
            time.sleep(0.5)
        else:
            raise AssertionError(f"no heartbeat: {S.read_json(pp)}")
        assert p["recommendation"] in ("go_now", "wait")
        assert led_command(p) in ("G", "A")
    finally:
        proc.terminate()
        proc.wait(10)
    old = dict(st, timestamp=(datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds"))
    p = qa.run_once(old, use_gemini=False, out=None, max_age_s=qa.STALE_S)
    assert p["recommendation"] == "unknown" and led_command(p) == "O"


# ---------------------------------------------------------------- agent and Gemini validation
class _FakeGenai:
    """Stands in for google.genai: replies with `text`, after optionally calling the tools."""
    def __init__(self, text, call_tools=True):
        self.text, self.call_tools = text, call_tools

    def install(self):
        import google.genai as genai
        fake = self

        class Chat:
            def __init__(self, tools_):
                self.tools = tools_

            def send_message(self, prompt):
                if fake.call_tools:
                    for f in self.tools:
                        f(**({"horizons": [15, 30, 60]} if f.__name__ == "get_forecast" else
                             {"slots": tools.candidate_slots(tools.get_queue_state()), "max_wait_min": 5}
                             if f.__name__ == "compare_slots" else {}))
                return type("R", (), {"text": fake.text})()

        class Client:
            def __init__(self, **kw):
                self.chats = type("C", (), {"create": staticmethod(lambda model, config: Chat(config.tools))})()
        self.orig = genai.Client
        genai.Client = Client
        return self

    def uninstall(self):
        import google.genai as genai
        genai.Client = self.orig


def test_gemini_answers_are_validated():
    df = build(f"{DAY}T17:30:00", f"{DAY}T18:20:00",
               [(f"{DAY}T18:0{k // 6}:{(k % 6) * 10:02d}", "in") for k in range(30)] +
               [(f"{DAY}T18:1{k}:00", "serve") for k in range(5)], "gem")
    st = state(df, ts("18:20:00"))
    fcs = fc.forecast(st)
    table = {s["slot"]: s for s in qa._slot_table(st, fcs)}
    os.environ["GEMINI_API_KEY"] = "test"
    try:
        good = json.dumps({"recommendation": "go_now", "best_time": st["timestamp"][11:16],
                           "explanation": f"Wait is about {st['wait_now_min']} min now: go."})
        g = _FakeGenai(good).install()
        try:
            p = qa.run_once(st, use_gemini=True, out=None)
        finally:
            g.uninstall()
        assert p["source"].endswith("gemini") and p["recommendation"] == "go_now", p

        small = min(table.values(), key=lambda s: abs(st["wait_now_min"] - s["expected_wait_min"] - 1))
        bad_cases = {
            "invented number": json.dumps({"recommendation": "go_now", "best_time": "18:20",
                                           "explanation": "Wait is about 123.4 min."}),
            "invented time": json.dumps({"recommendation": "go_now", "best_time": "18:20",
                                         "explanation": "It clears at 21:17."}),
            "bad json": "Sure! Go now.",
            "bad label": json.dumps({"recommendation": "maybe", "best_time": "18:20", "explanation": "x"}),
            "unknown slot": json.dumps({"recommendation": "wait", "best_time": "03:17", "explanation": "Later."}),
            "tiny saving": json.dumps({"recommendation": "wait", "best_time": small["slot"],
                                       "explanation": "Come back later."}),
        }
        for name, text in bad_cases.items():
            g = _FakeGenai(text).install()
            try:
                p = qa.run_once(st, use_gemini=True, out=None)
            finally:
                g.uninstall()
            if name == "tiny saving" and round(st["wait_now_min"] - small["expected_wait_min"]) >= qa.MIN_SAVING_MIN:
                continue
            assert p["source"].endswith("rules"), (name, p)
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_rule_decision_and_explanation_numbers():
    assert qa._explanation_ok("Wait 6.9 min now, 4.5 at 19:05", {6.9, 4.5}, {"19:05"})
    assert qa._explanation_ok("Wait about 7 min", {6.9}, set()) and qa._explanation_ok("about 6", {6.9}, set())
    assert not qa._explanation_ok("Wait 12 min", {6.9}, set())
    assert not qa._explanation_ok("Wait 6.8 min", {6.9}, set())
    assert not qa._explanation_ok("Wait 8 min", {6.9}, set())
    assert not qa._explanation_ok("Come at 19:07", {1.0}, {"19:05"})


def test_gemini_wait_with_a_real_saving_is_accepted():
    """A clear 'wait' answer quoting the saving (a subtraction no tool returns) must pass."""
    st = {"timestamp": "2026-10-24T18:20:00", "status": "open", "queue_length": 30, "wait_now_min": 30.0,
          "wait_instant_min": 30.0, "venue_open": "18:00", "venue_close": "19:30"}
    fcs = [{"horizon_min": 15, "p10": 10.0, "p50": 20.0, "p90": 30.0},
           {"horizon_min": 30, "p10": 5.0, "p50": 12.0, "p90": 20.0},
           {"horizon_min": 60, "p10": 2.0, "p50": 6.0, "p90": 12.0}]
    best = min(qa._slot_table(st, fcs), key=lambda s: s["expected_wait_min"])
    saved = round(30.0 - best["expected_wait_min"])
    text = json.dumps({"recommendation": "wait", "best_time": best["slot"],
                       "explanation": f"Come back at {best['slot']}: about {best['expected_wait_min']:g} min, saving about {saved} minutes."})
    os.environ["GEMINI_API_KEY"] = "test"
    real_fc, real_state = qa.forecast, tools.STATE
    g = _FakeGenai(text, call_tools=False).install()
    try:
        d = qa.gemini_decision(st, fcs)
    finally:
        g.uninstall()
        del os.environ["GEMINI_API_KEY"]
    assert d["recommendation"] == "wait" and d["minutes_saved"] == saved, d


# ---------------------------------------------------------------- model, tools, small parts
def test_prob_exceeds_is_a_sane_probability():
    for p10, p50, p90 in [(1, 3, 8), (0, 0, 0), (5, 5, 5), (2, 10, 11)]:
        xs = np.linspace(-5, 40, 50)
        ps = [tools.prob_exceeds(x, p10, p50, p90) for x in xs]
        assert all(0 <= p <= 1 for p in ps) and all(a >= b for a, b in zip(ps, ps[1:])), (p10, p50, p90)


def test_forecast_calibration_widens_bands():
    st = json.loads((ROOT / "data/sample/queue_state_sample.json").read_text())
    base = {f["horizon_min"]: f for f in fc.forecast(st, calibrated=False)}
    real = fc.CALIBRATION_JSON
    fc.CALIBRATION_JSON = TMP / "cal.json"
    try:
        fc.CALIBRATION_JSON.write_text(json.dumps({"source": ["x"], "k": {"15": 2.0, "30": 1.0, "60": 3.0}}))
        wide = {f["horizon_min"]: f for f in fc.forecast(st)}
    finally:
        fc.CALIBRATION_JSON = real
    for h, k in ((15, 2.0), (30, 1.0), (60, 3.0)):
        b, w = base[h], wide[h]
        assert w["p50"] == b["p50"]
        assert abs((w["p90"] - w["p50"]) - k * (b["p90"] - b["p50"])) <= 0.15, (h, b, w)
        assert w["p10"] >= 0


def test_forecast_monotone_quantiles_on_many_states():
    df = build(f"{DAY}T17:30:00", f"{DAY}T19:00:00",
               [(f"{DAY}T18:{k // 2:02d}:{(k % 2) * 30:02d}", "in") for k in range(60)] +
               [(f"{DAY}T18:{10 + k // 2:02d}:15", "serve") for k in range(60)], "mono")
    for m in range(0, 90, 7):
        st = state(df, ts("17:30:00") + pd.Timedelta(minutes=m))
        if st["status"] == "closed":
            continue
        for f in fc.forecast(st):
            assert 0 <= f["p10"] <= f["p50"] <= f["p90"], (m, f)


def test_logger_parsing():
    good = '{"ms": 123456, "dist_cm": 182.4, "in": 37, "out": 31, "btn": 12}\r\n'
    assert parse_line(good) == {"ms": 123456, "dist_cm": 182.4, "in": 37, "out": 31, "btn": 12}
    for bad in ['{"ms": 1, "dist_cm"', "garbage", "[1,2]", '{"ms":1}', '{"ms":"x","dist_cm":1,"in":1,"out":1,"btn":1}', ""]:
        assert parse_line(bad) is None, bad
    assert csv_row(parse_line(good), datetime(2026, 10, 24, 18, 0, 0)) == "2026-10-24T18:00:00,123456,182.4,37,31,12"


def test_camera_zone():
    poly = np.array([[0, 0], [320, 0], [320, 480], [0, 480]])
    assert count_in_zone([(10, 10, 100, 300), (250, 50, 330, 470), (400, 10, 500, 300), (300, 10, 340, 200)], poly) == 3
    assert count_in_zone([], poly) == 0


def test_alert_phrases():
    assert phrase({"recommendation": "go_now", "wait_now_min": 0.4}) == "The food queue is empty. Go now."
    assert "19:20" in phrase({"recommendation": "wait", "wait_now_min": 27.5, "best_time": "19:20", "minutes_saved": 11})
    for p in ({"recommendation": "unknown"}, {"recommendation": "closed"}, None):
        assert phrase(p) is None


def test_backtest_runs_and_skips_closed_minutes():
    ev = [(f"{DAY}T18:{k // 2:02d}:{(k % 2) * 30:02d}", "in") for k in range(40)] + \
         [(f"{DAY}T18:{5 + k // 2:02d}:15", "serve") for k in range(40)]
    df = build(f"{DAY}T17:30:00", f"{DAY}T20:10:00", ev, "bt")
    res = bt.run(df, "18:00", "19:30")
    assert not res.empty and (res["t"] <= ts("19:45:00")).all()
    rep = bt.report(res)
    assert set(rep) == {15, 30, 60} and all(r["k_for_80pct"] >= 1 for r in rep.values())


def test_sample_files_still_work():
    st = S.compute_state(S.load_run(ROOT / "data/sample/run_sample.csv"))
    assert st["queue_length"] == 11 and st["wait_instant_min"] == 6.9 and st["status"] == "open"
    assert qa.run_once(st, use_gemini=False, out=None)["recommendation"] in ("go_now", "wait")
    sample = json.loads((ROOT / "data/sample/queue_state_sample.json").read_text())
    assert len(fc.forecast(sample)) == 3


def test_fake_rush_replay_every_minute_plain_and_kalman():
    """The demo replay: every minute of the fake rush, both modes, through the agent."""
    df = S.load_run(ROOT / "data/sample/fake_rush_run.csv")
    cam = S.load_camera(ROOT / "data/sample/fake_rush_camera.csv")
    for m in range(0, 90):
        at = df.index[0] + pd.Timedelta(minutes=m)
        for live in (None, {}):
            st = state(df[df.index <= at], at, live=live, ultrasonic=US, camera=cam[cam.index <= at])
            p = qa.run_once(st, use_gemini=False, out=None)
            assert p["recommendation"] in ("go_now", "wait", "closed"), (m, p)
            assert st["queue_length"] >= 0 and st["wait_now_min"] >= 0, (m, st)


def test_dashboard_replay_renders_every_stage():
    """The replay dashboard, driven by Streamlit's test runner: before opening, peak, quiet end,
    closing grace and closed; plain and Kalman. No exception or error box allowed."""
    import logging
    from streamlit.testing.v1 import AppTest
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    df = pd.read_csv(ROOT / "data/sample/fake_rush_run.csv")
    last = df.iloc[-1]
    times = pd.date_range("2026-10-24T19:30:00", "2026-10-24T20:10:00", freq="5s")
    tail = pd.DataFrame({"wall_time": times.strftime("%Y-%m-%dT%H:%M:%S"), "dist_cm": 400.0,
                         "ms": last.ms + 1000 * (times - pd.Timestamp("2026-10-24T19:29:59")).total_seconds().astype(int),
                         "in_total": last.in_total, "out_total": last.out_total, "btn_total": last.btn_total})
    pd.concat([df, tail[df.columns]]).to_csv(TMP / "long_run.csv", index=False)
    expected = {0: ("Go now",), 30: ("Go now", "Go at"), 92: ("Go now",), 129: ("Closed",)}
    argv = sys.argv
    try:
        for minute, cards in expected.items():
            for smooth in (False, True):
                sys.argv = ["app.py", "--replay", str(TMP / "long_run.csv"),
                            "--camera", str(ROOT / "data/sample/fake_rush_camera.csv"),
                            "--us-empty", "400", "--us-cm-per-person", "25", "--speed", "30"]
                at = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60)
                at.session_state["t0"] = at.session_state["paused_at"] = time.monotonic()
                at.session_state["offset_s"], at.session_state["speed"] = minute * 60.0, 30
                at.run()
                if smooth:
                    next(c for c in at.checkbox if c.label.startswith("Kalman")).check()
                    at.run()
                errs = [e.value for e in at.exception] + [e.value for e in at.error]
                assert not errs, (minute, smooth, errs)
                card = next(m.value for m in at.markdown if "font-size:2.6rem" in m.value)
                assert any(c in card for c in cards), (minute, smooth, card[:200])
                if smooth and minute != 129:
                    assert any("kalman+ultrasonic" in c.value for c in at.caption), (minute, "no Kalman caption")
    finally:
        sys.argv = argv


def _decide(ts_, wait, fcs, close="19:30"):
    st = {"timestamp": ts_, "status": "open", "queue_length": 20, "wait_now_min": wait, "wait_instant_min": wait,
          "venue_open": "18:00", "venue_close": close}
    return qa.rule_decision(st, fcs)


def test_no_far_off_or_after_closing_advice():
    """Review 3: 'wait 55 minutes to save 3' and slots near closing must not be recommended."""
    flat = [{"horizon_min": 15, "p10": 6, "p50": 11.4, "p90": 17}, {"horizon_min": 30, "p10": 5, "p50": 10.9, "p90": 17},
            {"horizon_min": 60, "p10": 3, "p50": 8.5, "p90": 15}]
    d = _decide("2026-10-24T18:29:00", 11.4, flat)
    assert d["recommendation"] == "go_now", d  # the reviewer's case: 3 min saved for a 55-min delay
    drop = [{"horizon_min": 15, "p10": 2, "p50": 4.0, "p90": 8}, {"horizon_min": 30, "p10": 1, "p50": 3.0, "p90": 7},
            {"horizon_min": 60, "p10": 1, "p50": 2.5, "p90": 6}]
    d = _decide("2026-10-24T18:20:00", 14.0, drop)
    assert d["recommendation"] == "wait" and d["best_time"] <= "18:35", d  # a real, soon drop still says wait
    assert d["minutes_saved"] >= 3
    # 19:05, closes 19:30: big drop forecast, but nothing later than 19:15 may be suggested
    d = _decide("2026-10-24T19:05:00", 14.0, drop)
    assert d["recommendation"] == "go_now" or d["best_time"] <= "19:15", d
    d = _decide("2026-10-24T19:20:00", 14.0, drop)  # inside last call: only "now" is left
    assert d["recommendation"] == "go_now", d
    # midnight rush: 00:20 with close 00:45 -> last call 00:30
    d = _decide("2026-10-25T00:20:00", 14.0, drop, close="00:45")
    assert d["recommendation"] == "go_now" or d["best_time"] <= "00:30", d
    # every slot the agent can ever be offered respects last call
    st = {"timestamp": "2026-10-24T18:50:00", "venue_open": "18:00", "venue_close": "19:30"}
    assert tools.candidate_slots(st, last_call_min=qa.LAST_CALL_MIN)[-1] == "19:15"


def test_gemini_far_off_wait_is_rejected():
    st = {"timestamp": "2026-10-24T18:29:00", "status": "open", "queue_length": 20, "wait_now_min": 11.4,
          "wait_instant_min": 11.4, "venue_open": "18:00", "venue_close": "19:30"}
    flat = [{"horizon_min": 15, "p10": 6, "p50": 11.4, "p90": 17}, {"horizon_min": 30, "p10": 5, "p50": 10.9, "p90": 17},
            {"horizon_min": 60, "p10": 3, "p50": 8.5, "p90": 15}]
    os.environ["GEMINI_API_KEY"] = "test"
    try:
        for slot in ("19:14", "19:24"):  # within last call but not worth it / after last call
            g = _FakeGenai(json.dumps({"recommendation": "wait", "best_time": slot, "explanation": "Come back later."}),
                           call_tools=False).install()
            try:
                qa.gemini_decision(st, flat)
                raise AssertionError(f"accepted a pointless wait until {slot}")
            except ValueError:
                pass
            finally:
                g.uninstall()
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_fake_rush_never_suggests_a_slot_near_closing():
    df = S.load_run(ROOT / "data/sample/fake_rush_run.csv")
    for m in range(5, 90, 3):
        at = df.index[0] + pd.Timedelta(minutes=m)
        st = state(df[df.index <= at], at)
        p = qa.run_once(st, use_gemini=False, out=None)
        if p["recommendation"] == "wait":
            ahead = (pd.Timestamp(f"{at.date()}T{p['best_time']}") - at).total_seconds() / 60
            assert p["best_time"] <= "19:15", (m, p)
            assert p["minutes_saved"] - qa.DELAY_COST * ahead >= qa.MIN_SAVING_MIN - 0.6, (m, p, ahead)


def _serve(source):
    """Start app/server.py's handler on a free port in a thread; returns (base url, server)."""
    import threading
    from http.server import ThreadingHTTPServer
    from app import server as srv
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(source))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd


def _get(url, data=None):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_app_server_static_files_and_replay_api():
    from app import server as srv
    base, httpd = _serve(srv.Source(ROOT / "data/sample/fake_rush_run.csv", 30))
    try:
        for path, ctype in [("/", "text/html"), ("/phone", "text/html"), ("/qt.js", "javascript"), ("/site.css", "text/css"),
                            ("/terms", "text/html"), ("/privacy", "text/html"),
                            ("/sw.js", "javascript"), ("/manifest.webmanifest", "manifest+json"),
                            ("/icons/icon-192.png", "image/png"), ("/icons/icon-512.png", "image/png"), ("/icons/icon.svg", "svg")]:
            code, ct, body = _get(base + path)
            assert code == 200 and ctype in ct and body, (path, code, ct)
        manifest = json.loads(_get(base + "/manifest.webmanifest")[2])
        assert manifest["start_url"] == "/phone" and any(i["sizes"] == "512x512" for i in manifest["icons"])
        for bad in ("/../ingest/state.py", "/%2e%2e/README.md", "/app/server.py", "/web/qt.js", "/data/live/run.csv", "/api/nope"):
            assert _get(base + bad)[0] == 404, bad
        # the pages only reference files the server actually serves
        for page in ("/", "/phone", "/terms", "/privacy"):
            html = _get(base + page)[2].decode()
            import re
            for ref in re.findall(r'(?:href|src)="(/[^"]*)"', html):
                assert ref in srv.STATIC, (page, ref)
        # house rules for every page and the app code: no em dashes, no banned fonts, no rounded
        # corners, shadows, gradients or animation; every page links terms and privacy
        for path in ("/", "/phone", "/terms", "/privacy", "/site.css", "/qt.js"):
            text = _get(base + path)[2].decode()
            for banned in ("—", "Inter", "Geist", "Space Grotesk", "border-radius: 1", "box-shadow",
                           "linear-gradient", "radial-gradient", "transition", "@keyframes", "animation"):
                assert banned not in text, (path, banned)
        # nothing loads from outside the laptop (review 3: venue wifi may block Google Fonts)
        for path in ("/", "/phone", "/terms", "/privacy", "/site.css", "/qt.js", "/sw.js"):
            text = _get(base + path)[2].decode()
            assert "googleapis" not in text and "gstatic" not in text, path
        css = _get(base + "/site.css")[2].decode()
        for font in re.findall(r'url\("(/fonts/[^"]+)"\)', css):
            code, ct, body = _get(base + font)
            assert code == 200 and ct == "font/woff2" and body[:4] == b"wOF2", font
        assert _get(base + "/fonts/LICENSE.txt")[0] == 200
        for page in ("/", "/phone"):
            html = _get(base + page)[2].decode()
            assert 'href="/terms"' in html and 'href="/privacy"' in html, page
            assert 'class="web loading"' in html or 'class="phone loading"' in html, (page, "no skeleton state")
        assert _get(base + "/api/replay", json.dumps({"action": "jump", "minute": 25}).encode())[0] == 200
        assert _get(base + "/api/replay", json.dumps({"action": "pause"}).encode())[0] == 200
        time.sleep(1.1)
        code, ct, body = _get(base + "/api/snapshot")
        snap = json.loads(body)
        assert code == 200 and "application/json" in ct
        assert snap["mode"] == "replay" and snap["status"] == "ok" and snap["replay"]["paused"]
        assert snap["closes_at"] == "2026-10-24T19:30:00", snap["closes_at"]
        assert snap["state"]["timestamp"].startswith("2026-10-24T18:25"), snap["state"]["timestamp"]
        assert snap["prediction"]["recommendation"] in ("go_now", "wait") and snap["history"] and snap["sensors"]
        if snap["prediction"]["recommendation"] == "wait":
            assert snap["best_slot"]["slot"] == snap["prediction"]["best_time"]
            assert 0 <= snap["best_slot"]["prob_wait_over_max"] <= 1
        for bad in ({"action": "jump", "minute": 999}, {"action": "explode"}, [1, 2], "x" * 2000):
            assert _get(base + "/api/replay", json.dumps(bad).encode())[0] == 400, bad
        assert _get(base + "/api/replay", b"{not json")[0] == 400
    finally:
        httpd.shutdown()


def test_app_server_live_states_and_broken_files():
    from app import server as srv
    live = TMP / "live_app"
    live.mkdir(exist_ok=True)
    for f in live.iterdir():
        f.unlink()
    real_live = srv.LIVE
    srv.LIVE = live
    base, httpd = _serve(srv.Source())
    snap = lambda: (time.sleep(1.05), json.loads(_get(base + "/api/snapshot")[2]))[1]  # noqa: E731
    try:
        s = snap()
        assert s["status"] == "no_data" and s["mode"] == "live", s
        (live / "queue_state.json").write_text('{"timestamp": "2026-10-24T18:0')  # half-written
        assert snap()["status"] == "no_data"
        # a fresh state + prediction, re-stamped to now
        now = datetime.now().replace(microsecond=0)
        ev = [((now - timedelta(minutes=20) + timedelta(seconds=30 * k)).isoformat(), "in") for k in range(30)]
        ev += [((now - timedelta(minutes=15) + timedelta(seconds=40 * k)).isoformat(), "serve") for k in range(15)]
        df = build((now - timedelta(minutes=25)).isoformat(), now.isoformat(), ev, "live_app_run")
        (live / "run.csv").write_text((TMP / "live_app_run.csv").read_text())
        o, c = (now - timedelta(minutes=20)).strftime("%H:%M"), (now + timedelta(minutes=60)).strftime("%H:%M")
        st = S.compute_state(df, None, o, c, rush="dinner")
        S.write_json(st, live / "queue_state.json")
        S.write_json(qa.run_once(st, use_gemini=False, out=None), live / "prediction.json")
        s = snap()
        assert s["status"] == "ok" and s["prediction"]["recommendation"] in ("go_now", "wait"), s
        assert s["history"] and s["sensors"]["in"] == 30 and s["heartbeat_age_s"] < 60
        json.dumps(s, allow_nan=False)
        # agent stopped 3 min ago
        p = json.loads((live / "prediction.json").read_text())
        p["checked_at"] = p["generated_at"] = (now - timedelta(minutes=3)).isoformat(timespec="seconds")
        S.write_json(p, live / "prediction.json")
        s = snap()
        assert s["status"] == "stale" and s["prediction"]["recommendation"] == "unknown", s
        # sensor stopped 10 min ago
        S.write_json(qa.run_once(st, use_gemini=False, out=None), live / "prediction.json")
        S.write_json(dict(st, timestamp=(now - timedelta(minutes=10)).isoformat(timespec="seconds")), live / "queue_state.json")
        s = snap()
        assert s["status"] == "stale" and "sensor" in s["message"], s
        # run.csv garbage: still answers, with the state, just no history
        S.write_json(st, live / "queue_state.json")
        (live / "run.csv").write_text("this is not a csv\n\x00\x01")
        s = snap()
        assert s["state"] and s["history"] == [] and s["status"] in ("ok", "stale"), s
        # state file holding JSON of the wrong shape: an error message, never a crash
        (live / "queue_state.json").write_text("[1, 2, 3]")
        s = snap()
        assert s["status"] == "error" and s["message"], s
        assert _get(base + "/api/replay", json.dumps({"action": "pause"}).encode())[0] == 400  # not in replay mode
    finally:
        httpd.shutdown()
        srv.LIVE = real_live


def run_all():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        t = time.time()
        try:
            f()
            print(f"PASS  {name}  ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
