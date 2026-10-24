# Gemini agent (owner: S2 + Gemini team)

Gemini uses **function calling**: it decides which of our Python tools to call, we run them,
and Gemini combines the results into a recommendation and explanation.

| Tool | Returns | Implemented in |
|---|---|---|
| `get_queue_state()` | latest queue_state.json | `ingest/state.py` output |
| `get_forecast(horizons)` | P10/P50/P90 per horizon | `model/forecast.py` |
| `compare_slots(slots, max_wait_min)` | expected wait and P(wait > max) per time slot | `agent/tools.py` |

Rules for the prompt: never state a number that did not come from a tool; output must match
`docs/data_contract.md` §5 (use structured JSON output).
Run Gemini only when the state changes meaningfully (e.g. each new 5-minute window).
API key: `GEMINI_API_KEY` in `.env` (key from mlh.link/aistudio).
