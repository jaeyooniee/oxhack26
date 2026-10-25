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
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from agent import tools  # noqa: E402
from ingest.state import read_json, write_json  # noqa: E402
from model.forecast import forecast  # noqa: E402

PREDICTION_JSON = ROOT / "data" / "live" / "prediction.json"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")  # gemini-2.5-flash is retired for new keys
# tried in order when a model is overloaded (503), slow (504), rate-limited (429) or retired (404);
# the free tier allows 20 requests/day per model, so each fallback is also a fresh quota
GEMINI_MODELS = list(dict.fromkeys([GEMINI_MODEL] + [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite").split(",") if m.strip()]))
GEMINI_CALL_TIMEOUT_MS = 12_000  # per HTTP request: give up on a hung, overloaded model quickly
GEMINI_BUDGET_S = 35             # whole decision, all models; then rules (heartbeat stays under the LEDs' 60 s)
MAX_WAIT_MIN = 5.0     # "a short queue" for the probability tool
MIN_SAVING_MIN = 3     # recommend waiting only if the net saving (below) is at least this much
DELAY_COST = 0.1       # each minute you put off going must save 0.1 min of queue: 30 min later must save 3 more
LAST_CALL_MIN = 15     # never send people back later than this before closing (food runs out)
TIE_MIN = 0.5          # slots within this of the best net saving count as equally good; take the earliest
STALE_S = 120          # live mode: state older than this -> "unknown"
RECOMMENDATIONS = ("go_now", "wait", "unknown")

SYSTEM = f"""You advise people queueing for food at a hackathon whether to join the queue now or later.
Rules:
- Call the tools to get the current queue state, the forecast and slot comparisons.
- Never state a number that did not come from a tool result. Do not do arithmetic yourself.
- Only the candidate slots you are given are allowed (they stop {LAST_CALL_MIN} minutes before closing).
- Coming back later costs people time: a slot is only worth it if its expected wait is lower than
  the wait now by at least {MIN_SAVING_MIN} minutes plus {DELAY_COST:g} minutes for every minute until it.
  Prefer the earliest such slot; otherwise "go_now". Use "unknown" only if the tools fail.
- Reply with JSON only, no markdown:
  {{"recommendation": "go_now" | "wait" | "unknown", "best_time": "HH:MM",
    "explanation": "one friendly sentence, at most 25 words"}}"""


def build_prompt(user_question: str | None = None, state: dict | None = None) -> str:
    state = state or tools.get_queue_state()
    slots = tools.candidate_slots(state, last_call_min=LAST_CALL_MIN)
    q = user_question or "Should I join the food queue now, or come back later? If later, when?"
    return (f"Current time: {state['timestamp'][11:16]}. Candidate slots: {', '.join(slots)}. "
            f"A short wait means under {MAX_WAIT_MIN:g} minutes.\n{q}")


def _slot_table(state: dict, fcs: list[dict]) -> list[dict]:
    """Candidate slots up to LAST_CALL_MIN before closing, each with its net value."""
    wait_now = float(state["wait_now_min"])
    table = [s for s in tools.slot_forecasts(state, fcs, tools.candidate_slots(state, last_call_min=LAST_CALL_MIN),
                                             MAX_WAIT_MIN) if "error" not in s]
    for s in table:
        s["net_saving_min"] = net_saving(wait_now, s)
    return table


def net_saving(wait_now: float, slot: dict) -> float:
    """Minutes saved by going at this slot instead of now, minus DELAY_COST per minute of delay."""
    return wait_now - slot["expected_wait_min"] - DELAY_COST * slot["minutes_ahead"]


def rule_decision(state: dict, fcs: list[dict]) -> dict:
    """Go at the earliest slot whose net saving is (near) the best; 'go_now' unless that is worth it."""
    table = _slot_table(state, fcs)
    top = max(s["net_saving_min"] for s in table)
    best = next(s for s in table if s["net_saving_min"] >= top - TIE_MIN)
    wait_now = float(state["wait_now_min"])
    saved = round(wait_now - best["expected_wait_min"])
    now_hhmm = state["timestamp"][11:16]
    if best["net_saving_min"] >= MIN_SAVING_MIN and best["minutes_ahead"] > 0:
        expl = (f"Wait now is about {wait_now:g} min; at {best['slot']} it should be "
                f"{best['expected_wait_min']:g} min (likely {best['p10']:g}-{best['p90']:g}), "
                f"saving about {saved} min.")
        return {"recommendation": "wait", "best_time": best["slot"], "minutes_saved": saved,
                "explanation": expl}
    best_raw = max(round(wait_now - s["expected_wait_min"]) for s in table)
    trend = "isn't expected to get much shorter soon" if best_raw > 0 else "is likely to grow"
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
    """True if every number/time in the text came from a tool result: a decimal must equal a tool
    value rounded to the same precision ("6.9"), a whole number must be a tool value rounded down
    or up ("about 7" for 6.9), a time must be one of `times`."""
    for tok in re.findall(r"\d{1,2}:\d{2}|\d+(?:\.\d+)?", text):
        if ":" in tok:
            if tok.zfill(5) not in times:
                return False
        elif "." in tok:
            d = len(tok.split(".")[1])
            if not any(abs(round(a, d) - float(tok)) < 1e-9 for a in allowed):
                return False
        elif not any(int(tok) in (math.floor(a), math.ceil(a)) for a in allowed):
            return False
    return True


TREND_MIN = 0.5  # minutes of forecast change before "rising" or "dropping" is a fair word
UP_WORDS = re.compile(r"\b(increas\w*|grow\w*|rise|rising|rises|longer|worse|build\w* up|peak\w*)\b", re.I)
DOWN_WORDS = re.compile(r"\b(decreas\w*|drop\w*|fall\w*|shrink\w*|shorter|eas\w+|clear\w*|improv\w*)\b", re.I)


def _direction_ok(text: str, wait_now: float, fcs: list[dict]) -> bool:
    """Words about the trend must match the forecast medians: 'increase' needs some P50 at least
    TREND_MIN above the wait now, 'drop' at least TREND_MIN below it (a 0.1-min wobble is no trend).
    Negated phrases ("won't improve") are not judged."""
    if re.search(r"\bnot\b|n't\b|n’t\b|\bno\b|\bnever\b|\bwithout\b", text, re.I):  # "won't", "doesn't", "no"
        return True
    p50 = [f["p50"] for f in fcs] or [wait_now]
    if UP_WORDS.search(text) and max(p50) < wait_now + TREND_MIN:
        return False
    if DOWN_WORDS.search(text) and min(p50) > wait_now - TREND_MIN:
        return False
    return True


def _retryable(e: Exception) -> bool:
    """Overloaded, timed out, rate-limited or retired model: worth trying the next model."""
    code = getattr(e, "code", None)
    return code in (404, 429, 500, 502, 503, 504) or "timeout" in type(e).__name__.lower()


def _bound_tools(state: dict, results: list) -> list:
    """The three tools, answering for exactly this state (never a shared global, so a slow call that
    was abandoned can't change what a later call sees). Every result is kept for number checking."""
    def get_queue_state() -> dict:
        """Latest live queue state: length, arrival and service rates, current wait estimate."""
        results.append(state)
        return state

    def get_forecast(horizons: list[int]) -> list[dict]:
        """Quantile forecasts (P10/P50/P90, minutes) of the wait for each horizon in minutes."""
        have = tools.available_horizons()
        wanted = sorted({int(h) for h in horizons if int(h) in have}) or have
        out = tools.forecast(state, wanted)
        results.append(out)
        return out

    def compare_slots(slots: list[str], max_wait_min: float) -> list[dict]:
        """For each clock time in `slots` (HH:MM), expected wait and probability it exceeds max_wait_min."""
        out = tools.slot_forecasts(state, tools.forecast(state), slots, max_wait_min)
        results.append(out)
        return out
    return [get_queue_state, get_forecast, compare_slots]


def gemini_decision(state: dict, fcs: list[dict], user_question: str | None = None) -> dict:
    """Ask Gemini with function calling, trying GEMINI_MODELS in turn when one is overloaded or gone;
    raise on any other failure so the caller can fall back. The reply's "_model" names who answered."""
    from google import genai
    from google.genai import types

    results = [state, fcs]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(),
                          http_options=types.HttpOptions(timeout=GEMINI_CALL_TIMEOUT_MS))
    resp, model, errors = None, None, []
    for model in GEMINI_MODELS:
        try:
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(system_instruction=SYSTEM, tools=_bound_tools(state, results),
                                                   temperature=0.2),
            )
            resp = chat.send_message(build_prompt(user_question, state))
            break
        except Exception as e:
            if not _retryable(e):
                raise
            errors.append(f"{model}: {getattr(e, 'code', type(e).__name__)}")
    if resp is None:
        raise RuntimeError("no Gemini model answered (" + "; ".join(errors) + ")")
    text = (resp.text or "").strip()
    found = re.search(r"\{.*\}", text, flags=re.S)  # the JSON object, even with prose or ``` fences around it
    if not found:
        raise ValueError(f"no JSON object in Gemini's reply: {text[:120]!r}")
    reply = json.loads(found.group(0))
    if not isinstance(reply, dict):
        raise ValueError("Gemini's reply is not a JSON object")

    rec = reply.get("recommendation")
    if rec not in RECOMMENDATIONS:
        raise ValueError(f"bad recommendation {rec!r}")
    now_hhmm = state["timestamp"][11:16]
    table = {s["slot"]: s for s in _slot_table(state, fcs)}
    best_time = now_hhmm if rec == "go_now" else str(reply.get("best_time", ""))
    if rec == "wait" and best_time not in table:
        raise ValueError(f"best_time {best_time!r} is not a candidate slot")
    saved = round(float(state["wait_now_min"]) - table[best_time]["expected_wait_min"]) if rec == "wait" else 0
    if rec == "wait" and table[best_time]["net_saving_min"] < MIN_SAVING_MIN:
        raise ValueError(f"'wait' until {best_time} saves {saved} min for a {table[best_time]['minutes_ahead']}-min "
                         f"delay: not worth it (net {table[best_time]['net_saving_min']:.1f} < {MIN_SAVING_MIN})")
    explanation = str(reply.get("explanation", "")).strip()
    times = set(table) | {now_hhmm, state["venue_open"], state["venue_close"]}
    # savings per slot (computed here, so Gemini may say "save about 4 minutes")
    results.append([float(state["wait_now_min"]) - s["expected_wait_min"] for s in table.values()])
    if not explanation or not _explanation_ok(explanation, _allowed_numbers(results), times):
        raise ValueError(f"explanation quotes numbers not from tools: {explanation!r}")
    if not _direction_ok(explanation, float(state["wait_now_min"]), fcs):
        raise ValueError(f"explanation's trend contradicts the forecast: {explanation!r}")
    return {"recommendation": rec, "best_time": best_time, "minutes_saved": saved,
            "explanation": explanation, "_model": model}


def gemini_within_budget(state: dict, fcs: list[dict], user_question: str | None = None) -> dict:
    """gemini_decision in a background thread, given at most GEMINI_BUDGET_S. A slower call is
    abandoned (its result is never used), so the agent's heartbeat and the LEDs never stall."""
    box: dict = {}

    def work():
        try:
            box["decision"] = gemini_decision(state, fcs, user_question)
        except Exception as e:
            box["error"] = e
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(GEMINI_BUDGET_S)
    if th.is_alive():
        raise TimeoutError(f"Gemini took longer than {GEMINI_BUDGET_S} s")
    if "error" in box:
        raise box["error"]
    return box["decision"]


def run_once(state: dict | None = None, use_gemini: bool = True, user_question: str | None = None,
             out: Path | None = PREDICTION_JSON, max_age_s: float | None = None) -> dict:
    """Build prediction.json for the given (or latest) state; write it unless out is None."""
    state = state or tools.get_queue_state()
    closed = state.get("status") == "closed"
    fcs = [] if closed else forecast(state)
    made = datetime.now().isoformat(timespec="seconds")
    pred = {"timestamp": state["timestamp"], "generated_at": made, "checked_at": made,
            "wait_now_min": state["wait_now_min"], "forecasts": fcs}
    age = (datetime.now() - datetime.fromisoformat(state["timestamp"])).total_seconds()
    if max_age_s is not None and age > max_age_s:
        pred.update(recommendation="unknown", best_time=None, minutes_saved=0,
                    explanation=f"No fresh sensor data for {age / 60:.0f} min.", source="stale")
    elif closed:
        nxt = state.get("next_rush")
        pred.update(recommendation="closed", best_time=None, minutes_saved=0, source="schedule",
                    explanation="The food point is closed." + (f" Next: {nxt['name']} at {nxt['open']}." if nxt else ""))
    else:
        decision, source = None, "model:generic_v2 + rules"
        if use_gemini and os.environ.get("GEMINI_API_KEY"):
            try:
                decision = gemini_within_budget(state, fcs, user_question)
                source = f"model:generic_v2 + gemini ({decision.pop('_model')})"
            except Exception as e:  # network, quota, overload, slow, bad JSON, invented numbers: fall back
                msg, key = str(e), os.environ.get("GEMINI_API_KEY", "").strip()
                print(f"gemini failed, using rules: {msg.replace(key, '[KEY]') if key else msg}", file=sys.stderr)
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
    if args.no_gemini:
        print("Gemini disabled by --no-gemini; using rules.")
    elif not os.environ.get("GEMINI_API_KEY", "").strip():
        print(f"Gemini not configured: set GEMINI_API_KEY in {ROOT / '.env'} "
              "or the environment; using rules.", file=sys.stderr)
    else:
        print(f"Gemini configured: {' then '.join(GEMINI_MODELS)} (called for fresh, open queue states; "
              f"rules after {GEMINI_BUDGET_S} s).")
    tools.STATE_PATH = args.state

    last_key = last_wait = last_rec = last_pred = None
    while True:
        try:
            state = read_json(args.state)
            if state is None:
                raise FileNotFoundError(f"{args.state} missing or being written")
            ts = datetime.fromisoformat(state["timestamp"])
            key = (ts.date(), ts.hour, ts.minute // 5)
            wait = float(state["wait_now_min"])
            stale = args.watch and (datetime.now() - ts).total_seconds() > STALE_S
            changed = (key != last_key or last_wait is None or abs(wait - last_wait) > 2
                       or (state.get("status") == "closed") != (last_rec == "closed"))
            if changed or (stale and last_rec != "unknown"):
                pred = run_once(state, not args.no_gemini, args.question, args.out,
                                STALE_S if args.watch else None)
                last_key, last_wait, last_rec, last_pred = key, wait, pred["recommendation"], pred
                print(f"{pred['timestamp']}  {pred['recommendation']:<7} {pred['best_time']}  "
                      f"[{pred['source']}] {pred['explanation']}")
            elif last_pred is not None and args.watch:
                # heartbeat: the logger turns the LEDs off if this stops updating
                last_pred["checked_at"] = datetime.now().isoformat(timespec="seconds")
                write_json(last_pred, args.out)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"waiting for queue state: {e}")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
