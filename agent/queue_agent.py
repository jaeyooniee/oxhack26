"""Gemini queue agent (owner: S2).

When queue_state.json changes meaningfully (new 5-minute window or the wait moves > 2 min), ask
Gemini (google-genai, automatic function calling with the tools in agent/tools.py) for a
recommendation, validate it and write data/live/prediction.json (docs/data_contract.md §5).

Numbers always come from Python: the forecasts and minutes_saved are recomputed here, and an
explanation quoting a number no tool returned is replaced by the rule-based one. Without
GEMINI_API_KEY, or when Gemini fails, the rule-based recommendation is written instead.

Usage:
  python agent/queue_agent.py                 # once
  python agent/queue_agent.py --watch 10      # check every 10 s
  python agent/queue_agent.py --no-gemini     # rules only
"""
import argparse
import functools
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import tools  # noqa: E402
from ingest.state import write_json  # noqa: E402
from model.forecast import forecast  # noqa: E402

PREDICTION_JSON = ROOT / "data" / "live" / "prediction.json"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_WAIT_MIN = 5.0     # "a short queue" for the probability tool
MIN_SAVING_MIN = 3     # recommend waiting only if it saves at least this much
TIE_MIN = 0.5          # slots within this of the best P50 count as equally good; take the earliest
STALE_S = 120          # live mode: state older than this -> "unknown"
RECOMMENDATIONS = ("go_now", "wait", "unknown")

SYSTEM = f"""You advise people queueing for food at a hackathon whether to join the queue now or later.
Rules:
- Call the tools to get the current queue state, the forecast and slot comparisons.
- Never state a number that did not come from a tool result. Do not do arithmetic yourself.
- Recommend "wait" only if a later slot's expected wait is at least {MIN_SAVING_MIN} minutes lower
  than the wait now; otherwise "go_now". Use "unknown" only if the tools fail.
- Reply with JSON only, no markdown:
  {{"recommendation": "go_now" | "wait" | "unknown", "best_time": "HH:MM",
    "explanation": "one friendly sentence, at most 25 words"}}"""


def build_prompt(user_question: str | None = None, state: dict | None = None) -> str:
    state = state or tools.get_queue_state()
    slots = tools.candidate_slots(state)
    q = user_question or "Should I join the food queue now, or come back later? If later, when?"
    return (f"Current time: {state['timestamp'][11:16]}. Candidate slots: {', '.join(slots)}. "
            f"A short wait means under {MAX_WAIT_MIN:g} minutes.\n{q}")


def _slot_table(state: dict, fcs: list[dict]) -> list[dict]:
    return [s for s in tools.slot_forecasts(state, fcs, tools.candidate_slots(state), MAX_WAIT_MIN)
            if "error" not in s]


def rule_decision(state: dict, fcs: list[dict]) -> dict:
    """Go at the earliest slot whose expected wait is (near) the lowest; 'go_now' unless it saves enough."""
    table = _slot_table(state, fcs)
    best_p50 = min(s["expected_wait_min"] for s in table)
    best = next(s for s in table if s["expected_wait_min"] <= best_p50 + TIE_MIN)
    wait_now = float(state["wait_now_min"])
    saved = round(wait_now - best["expected_wait_min"])
    now_hhmm = state["timestamp"][11:16]
    if saved >= MIN_SAVING_MIN and best["minutes_ahead"] > 0:
        expl = (f"Wait now is about {wait_now:g} min; at {best['slot']} it should be "
                f"{best['expected_wait_min']:g} min (likely {best['p10']:g}-{best['p90']:g}), "
                f"saving about {saved} min.")
        return {"recommendation": "wait", "best_time": best["slot"], "minutes_saved": saved,
                "explanation": expl}
    trend = "isn't expected to get much shorter" if saved > 0 else "is likely to grow"
    if wait_now == 0:
        trend = "is empty"
    return {"recommendation": "go_now", "best_time": now_hhmm, "minutes_saved": 0,
            "explanation": f"Wait now is about {wait_now:g} min and the queue {trend}: go now."}


def _allowed_numbers(results: list) -> set[float]:
    nums = {10.0, 50.0, 80.0, 90.0, 100.0}  # P10/P50/P90, "80% range"

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            nums.update({float(x), round(float(x)), float(x) * 100})
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(results)
    return nums


def _explanation_ok(text: str, allowed: set[float], times: set[str]) -> bool:
    """True if every number/time in the text came from a tool result (±0.5)."""
    for tok in re.findall(r"\d{1,2}:\d{2}|\d+(?:\.\d+)?", text):
        if ":" in tok:
            if tok.zfill(5) not in times:
                return False
        elif not any(abs(float(tok) - a) <= 0.5 for a in allowed):
            return False
    return True


def gemini_decision(state: dict, fcs: list[dict], user_question: str | None = None) -> dict:
    """Ask Gemini with function calling; raise on any failure so the caller can fall back."""
    from google import genai
    from google.genai import types

    results = [state, fcs]

    def record(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            results.append(out)
            return out
        return wrapper

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=30_000))
    saved_state, tools.STATE = tools.STATE, state  # tools see the same state we decide on
    try:
        chat = client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=[record(tools.get_queue_state), record(tools.get_forecast), record(tools.compare_slots)],
                temperature=0.2,
            ),
        )
        resp = chat.send_message(build_prompt(user_question, state))
    finally:
        tools.STATE = saved_state
    text = re.sub(r"^```(?:json)?|```$", "", (resp.text or "").strip(), flags=re.M).strip()
    reply = json.loads(text)

    rec = reply.get("recommendation")
    if rec not in RECOMMENDATIONS:
        raise ValueError(f"bad recommendation {rec!r}")
    now_hhmm = state["timestamp"][11:16]
    table = {s["slot"]: s for s in _slot_table(state, fcs)}
    best_time = now_hhmm if rec == "go_now" else str(reply.get("best_time", ""))
    if rec == "wait" and best_time not in table:
        raise ValueError(f"best_time {best_time!r} is not a candidate slot")
    saved = round(float(state["wait_now_min"]) - table[best_time]["expected_wait_min"]) if rec == "wait" else 0
    explanation = str(reply.get("explanation", "")).strip()
    times = set(table) | {now_hhmm, state["venue_open"], state["venue_close"]}
    if not explanation or not _explanation_ok(explanation, _allowed_numbers(results), times):
        raise ValueError(f"explanation quotes numbers not from tools: {explanation!r}")
    return {"recommendation": rec, "best_time": best_time, "minutes_saved": saved,
            "explanation": explanation}


def run_once(state: dict | None = None, use_gemini: bool = True, user_question: str | None = None,
             out: Path | None = PREDICTION_JSON, max_age_s: float | None = None) -> dict:
    """Build prediction.json for the given (or latest) state; write it unless out is None."""
    state = state or tools.get_queue_state()
    fcs = forecast(state)
    pred = {"timestamp": state["timestamp"], "generated_at": datetime.now().isoformat(timespec="seconds"),
            "wait_now_min": state["wait_now_min"], "forecasts": fcs}
    age = (datetime.now() - datetime.fromisoformat(state["timestamp"])).total_seconds()
    if max_age_s is not None and age > max_age_s:
        pred.update(recommendation="unknown", best_time=None, minutes_saved=0,
                    explanation=f"No fresh sensor data for {age / 60:.0f} min.", source="stale")
    else:
        decision, source = None, "model:generic_v2 + rules"
        if use_gemini and os.environ.get("GEMINI_API_KEY"):
            try:
                decision, source = gemini_decision(state, fcs, user_question), "model:generic_v2 + gemini"
            except Exception as e:  # network, quota, bad JSON, invented numbers: fall back
                print(f"gemini failed, using rules: {e}", file=sys.stderr)
        pred.update(decision or rule_decision(state, fcs), source=source)
    if out is not None:
        write_json(pred, out)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=tools.STATE_PATH)
    ap.add_argument("--out", type=Path, default=PREDICTION_JSON)
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--question", help="ask a custom question (one-off)")
    ap.add_argument("--watch", type=float, metavar="SECONDS")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    tools.STATE_PATH = args.state

    last_key = last_wait = last_rec = None
    while True:
        try:
            state = json.loads(args.state.read_text())
            ts = datetime.fromisoformat(state["timestamp"])
            key = (ts.date(), ts.hour, ts.minute // 5)
            wait = float(state["wait_now_min"])
            stale = args.watch and (datetime.now() - ts).total_seconds() > STALE_S
            changed = key != last_key or last_wait is None or abs(wait - last_wait) > 2
            if changed or (stale and last_rec != "unknown"):
                pred = run_once(state, not args.no_gemini, args.question, args.out,
                                STALE_S if args.watch else None)
                last_key, last_wait, last_rec = key, wait, pred["recommendation"]
                print(f"{pred['timestamp']}  {pred['recommendation']:<7} {pred['best_time']}  "
                      f"[{pred['source']}] {pred['explanation']}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"waiting for queue state: {e}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
