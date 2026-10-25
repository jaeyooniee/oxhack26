"""Streamlit dashboard (owner: S2).

Live:    streamlit run app/app.py
         (reads data/live/run.csv, queue_state.json, prediction.json written by
          ingest/logger.py, ingest/state.py --watch and agent/queue_agent.py --watch)
Replay:  streamlit run app/app.py -- --replay data/sample/fake_rush_run.csv --speed 20
         (plays a recorded rush through state -> forecast -> agent inside the dashboard;
          use it if the sensor fails. <rush>_camera.csv next to it is picked up automatically)

Panels: recommendation, live queue numbers (with 80% range when Kalman smoothing is on),
wait history + forecast band (P10-P90, P50), sensor accuracy (beams vs hand count vs camera).
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.queue_agent import run_once  # noqa: E402
from ingest.state import (load_camera, load_counts, load_schedule, load_ultrasonic, next_rush,  # noqa: E402
                          read_json, resolve_rush, state_and_history)
from ingest.logger import PREDICTION_MAX_AGE_S, prediction_age_s  # noqa: E402
from model.forecast import calibration  # noqa: E402

LIVE = ROOT / "data" / "live"
REFRESH_S = 2
CARD = {"go_now": ("#1b8a3a", "Go now"), "wait": ("#c77700", "Wait"), "closed": ("#4a5260", "Closed"),
        "unknown": ("#666666", "No recommendation")}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, help="run.csv of a recorded rush")
    ap.add_argument("--speed", type=float, default=20, help="replay speed (x real time)")
    ap.add_argument("--open", dest="venue_open")
    ap.add_argument("--close", dest="venue_close")
    ap.add_argument("--camera", type=Path, help="camera.csv for the replay (default: <rush>_camera.csv)")
    ap.add_argument("--us-empty", type=float, help="ultrasonic cm with an empty queue (Kalman fusion)")
    ap.add_argument("--us-cm-per-person", type=float, help="ultrasonic cm closer per extra person")
    args = ap.parse_known_args()[0]
    if args.replay and args.camera is None:
        guess = args.replay.with_name(args.replay.name.replace("_run.csv", "_camera.csv"))
        args.camera = guess if guess != args.replay else None
    return args


@st.cache_data(show_spinner=False)
def _load(path: str, camera: str | None, mtimes: tuple, source: str) -> pd.DataFrame:
    return load_counts(Path(path), Path(camera) if camera else None, source)


def load(path: Path, camera: Path | None = None, source: str = "auto") -> pd.DataFrame | None:
    """Counters as ingest/state.py sees them: the camera tracker's when it is running."""
    paths = [p for p in (path, camera) if p is not None and p.exists()]
    if not paths:
        return None
    try:
        return _load(str(path), str(camera) if camera else None, tuple(p.stat().st_mtime for p in paths), source)
    except (OSError, ValueError, KeyError):
        return None


def recommendation_card(pred: dict | None):
    rec = (pred or {}).get("recommendation", "unknown")
    color, title = CARD.get(rec, CARD["unknown"])
    if rec == "wait":
        title = f"Go at {pred['best_time']} · save ~{pred['minutes_saved']} min"
    expl = (pred or {}).get("explanation", "Waiting for the first prediction.")
    st.markdown(
        f"""<div style="background:{color};color:#fff;border-radius:14px;padding:22px 26px;margin-bottom:8px">
        <div style="font-size:2.6rem;font-weight:700;line-height:1.1">{title}</div>
        <div style="font-size:1.15rem;margin-top:8px;opacity:.95">{expl}</div></div>""",
        unsafe_allow_html=True)
    if pred:
        st.caption(f"source: {pred.get('source', '?')} · state at {pred['timestamp'][11:19]}")


def metrics(state: dict):
    c = st.columns(5)
    c[0].metric("Wait now (min)", f"{state['wait_now_min']:g}",
                help="Median of the last 3 minutes, like a posted wait. "
                     f"Right this second: {state.get('wait_instant_min', state['wait_now_min']):g} min.")
    c[1].metric("In queue", state["queue_length"])
    c[2].metric("Arrived (5 min)", state["arrivals_last_5m"])
    c[3].metric("Served (5 min)", state["served_last_5m"])
    c[4].metric("Service rate", f"{state['service_rate_per_min']:g}/min")
    if "wait_now_ci80_min" in state:
        lo, hi = state["wait_now_ci80_min"]
        st.caption(f"{state['smoothing']}: wait right now is 80% likely between {lo:g} and {hi:g} min · "
                   f"queue ±{state['queue_length_sd']:g} people · arrivals {state['arrival_rate_per_min']:g}/min")


def forecast_chart(history: pd.DataFrame, now: pd.Timestamp, pred: dict | None,
                   actual: pd.DataFrame | None = None):
    hist = history.reset_index(names="time")[["time", "wait_min"]].assign(kind="measured")
    layers = []
    if pred:
        w = float(pred["wait_now_min"])
        fc = [{"time": now, "p10": w, "p50": w, "p90": w}] + [
            {"time": now + pd.Timedelta(minutes=f["horizon_min"]), **{q: f[q] for q in ("p10", "p50", "p90")}}
            for f in pred["forecasts"]]
        fc = pd.DataFrame(fc)
        layers += [
            alt.Chart(fc).mark_area(opacity=0.25, color="#4c78a8").encode(
                x="time:T", y=alt.Y("p10:Q", title="wait (min)"), y2="p90:Q",
                tooltip=[alt.Tooltip("time:T", format="%H:%M"), "p10", "p50", "p90"]),
            alt.Chart(fc).mark_line(strokeDash=[6, 4], point=True, color="#4c78a8").encode(x="time:T", y="p50:Q"),
        ]
    if actual is not None and not actual.empty:
        act = actual.reset_index(names="time")
        layers.append(alt.Chart(act).mark_line(color="#999", strokeDash=[2, 2]).encode(
            x="time:T", y="wait_min:Q"))
    layers.append(alt.Chart(hist).mark_line(color="#e45756", strokeWidth=3).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")), y=alt.Y("wait_min:Q", title="wait (min)"),
        tooltip=[alt.Tooltip("time:T", format="%H:%M"), alt.Tooltip("wait_min:Q", format=".1f")]))
    layers.append(alt.Chart(pd.DataFrame({"time": [now]})).mark_rule(color="#888").encode(x="time:T"))
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    cal = calibration()
    note = ("Red: measured wait (median of the last 3 min). Blue: forecast P50 (dashed) and P10-P90 band, "
            + (f"widened to cover 80% on {len(cal['source'])} recorded rush(es)." if cal else
               "as trained on Disney: not yet checked on this queue (model/backtest.py)."))
    st.caption(note + (" Grey dotted: what actually happened next." if actual is not None else ""))


def accuracy_panel(df: pd.DataFrame, history: pd.DataFrame, cam: pd.DataFrame | None):
    with st.expander("Sensor accuracy: camera / beams vs hand count", expanded=False):
        src = "camera tracker" if df.attrs.get("source") == "camera" else "beam"
        counts = df[["in_total", "out_total", "btn_total"]].resample("30s").last().dropna()
        counts.columns = [f"arrivals ({src})", f"served ({src})", "served (hand count)"]
        st.line_chart(counts)
        btn, out = int(df["btn_total"].iloc[-1]), int(df["out_total"].iloc[-1])
        if btn:
            st.write(f"The {src} counted **{out}** served vs **{btn}** by hand: "
                     f"error **{100 * (out - btn) / btn:+.1f}%**.")
        if cam is not None and not cam.empty:
            cam = cam[cam.index <= history.index[-1]]["people_in_zone"].resample("1min").median()
            both = pd.DataFrame({"queue length (sensors)": history["queue_length"],
                                 "people in zone (camera)": cam.reindex(history.index, method="nearest")})
            st.line_chart(both)
            gap = (both.iloc[:, 0] - both.iloc[:, 1]).abs().mean()
            st.write(f"Sensors vs camera: mean difference **{gap:.1f} people** per minute.")


def replay_clock(df: pd.DataFrame) -> pd.Timestamp:
    """Replay time = data start + offset + (real seconds since t0) x speed, frozen while paused."""
    ss = st.session_state
    elapsed = (ss.paused_at or time.monotonic()) - ss.t0
    return min(df.index[-1], (df.index[0] + pd.Timedelta(seconds=ss.offset_s + elapsed * ss.speed)).floor("s"))


def replay_sidebar(df: pd.DataFrame, args) -> tuple[bool, bool]:
    ss = st.session_state
    st.sidebar.header("Replay")
    st.sidebar.caption(str(args.replay))
    speed = st.sidebar.slider("Speed (x real time)", 1, 120, int(args.speed))
    total_min = int((df.index[-1] - df.index[0]).total_seconds() // 60)
    jump = st.sidebar.slider("Jump to minute", 0, total_min, 0)
    if "t0" not in ss:
        ss.t0, ss.offset_s, ss.paused_at, ss.speed = time.monotonic(), 0.0, None, speed
    if speed != ss.speed:  # keep the current position
        pos = (replay_clock(df) - df.index[0]).total_seconds()
        ss.t0, ss.offset_s, ss.speed = time.monotonic(), pos, speed
        if ss.paused_at:
            ss.paused_at = ss.t0
    c = st.sidebar.columns(2)
    if c[0].button("Jump"):
        ss.t0, ss.offset_s, ss.paused_at = time.monotonic(), jump * 60.0, None
    if c[1].button("Pause / play"):
        if ss.paused_at:
            ss.t0 += time.monotonic() - ss.paused_at
            ss.paused_at = None
        else:
            ss.paused_at = time.monotonic()
    show_actual = st.sidebar.checkbox("Show what actually happened", value=False)
    use_gemini = st.sidebar.checkbox("Use Gemini (one call per 5 min of replay)",
                                     value=False, disabled=not os.environ.get("GEMINI_API_KEY"))
    smooth = st.sidebar.checkbox("Kalman smoothing (error bars)", value=False)
    return show_actual, use_gemini, smooth


def replay_prediction(state: dict, use_gemini: bool) -> dict:
    """Rules every refresh; Gemini at most once per 5-minute window of replay time."""
    key = (state["timestamp"][:14] + str(int(state["timestamp"][14:16]) // 5), use_gemini)
    cache = st.session_state.setdefault("pred_cache", {})
    if use_gemini and key in cache:
        pred = run_once(state, use_gemini=False, out=None)  # fresh numbers, cached words
        return {**pred, **{k: cache[key][k] for k in ("recommendation", "best_time", "minutes_saved",
                                                      "explanation", "source")}}
    pred = run_once(state, use_gemini=use_gemini, out=None)
    cache[key] = pred
    return pred


def main():
    args = parse_args()
    st.set_page_config(page_title="QueueTime", page_icon="⏱", layout="wide")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if args.replay:
        df_full = load(args.replay, args.camera)
        if df_full is None:
            st.error(f"Replay file not found: {args.replay}")
            return
        show_actual, use_gemini, smooth = replay_sidebar(df_full, args)
        live = {} if smooth else None
        us = ({"us_empty_cm": args.us_empty, "us_cm_per_person": args.us_cm_per_person}
              if args.us_empty and args.us_cm_per_person else None)
        cam_full = load_camera(args.camera) if args.camera else None
    else:
        st.sidebar.header("Live")
        st.sidebar.caption("Reads data/live/. No sensor? Run with `-- --replay data/sample/fake_rush_run.csv`.")

    def render():
        if args.replay:
            now = replay_clock(df_full)
            df = df_full[df_full.index <= now]
            schedule = [] if args.venue_open else load_schedule()
            r = (resolve_rush(now.to_pydatetime(), schedule) if schedule
                 else {"name": None, "open": args.venue_open, "close": args.venue_close})
            cam = cam_full[cam_full.index <= now] if cam_full is not None else None
            state, history = state_and_history(df, now, r["open"], r["close"], live=live, rush=r["name"],
                                               ultrasonic=us, camera=cam)
            if state["status"] == "closed" and schedule:
                state["next_rush"] = next_rush(now.to_pydatetime(), schedule)
            pred = replay_prediction(state, use_gemini)
            mode = f"REPLAY {st.session_state.speed}x" + (" (paused)" if st.session_state.paused_at else "")
        else:
            state = read_json(LIVE / "queue_state.json")
            df = load(LIVE / "run.csv", LIVE / "camera.csv", (state or {}).get("counts_from", "auto"))
            pred = read_json(LIVE / "prediction.json")
            if df is None or state is None:
                st.info("No live data yet: start ingest/logger.py, ingest/state.py --watch and "
                        "agent/queue_agent.py --watch, or run the dashboard in replay mode.")
                return
            now = pd.Timestamp(state["timestamp"])
            age = prediction_age_s(pred)
            if age is None or age > PREDICTION_MAX_AGE_S:  # same rule as the LEDs
                pred = {**(pred or {}), "recommendation": "unknown",
                        "explanation": "No update from the agent " + (f"for {age / 60:.0f} min" if age else "yet")
                        + ": is agent/queue_agent.py --watch running?"}
            df = df[df.index <= now]
            cam = load_camera(LIVE / "camera.csv")
            # same inputs as ingest/state.py, so the chart matches the numbers above it
            _, history = state_and_history(df, now, state["venue_open"], state["venue_close"],
                                           live={} if "smoothing" in state else None,
                                           ultrasonic=load_ultrasonic(), camera=cam)
            mode = "LIVE"
        df = df[df.index >= history.index[0] - pd.Timedelta(minutes=1)]

        st.markdown(f"### ⏱ QueueTime · food point &nbsp; <span style='opacity:.6;font-size:1rem'>"
                    f"{mode} · {now:%H:%M:%S}</span>", unsafe_allow_html=True)
        recommendation_card(pred)
        metrics(state)
        actual = None
        if args.replay and show_actual:
            _, future = state_and_history(df_full, df_full.index[-1], r["open"], r["close"], live=live,
                                          ultrasonic=us, camera=cam_full)
            actual = future[(future.index >= now) & (future.index <= now + pd.Timedelta(minutes=60))]
        forecast_chart(history, now, pred, actual)
        accuracy_panel(df, history, cam)

    @st.fragment(run_every=REFRESH_S)
    def body():
        try:
            render()
        except Exception as e:  # keep refreshing through a bad file or a bug; show it, don't die
            st.error(f"Dashboard error (still refreshing): {e}")
            traceback.print_exc()

    body()
    repo = os.environ.get("REPO_URL")
    st.caption("Forecast: LightGBM trained on Disney World wait times (unit-free features), "
               "P10/P50/P90. Numbers from Python; Gemini only explains." + (f" · Code: {repo}" if repo else ""))


main()
