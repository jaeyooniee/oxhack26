"""Gemini queue agent (owner: S2).

Loop: when queue_state.json changes meaningfully, ask Gemini (google-genai SDK, automatic function
calling with the tools in agent/tools.py) for a recommendation, validate the JSON and write
data/live/prediction.json (docs/data_contract.md §5).

TODO
- [ ] client from GEMINI_API_KEY; register tools from agent/tools.py
- [ ] system prompt: numbers only from tools; reply in the prediction.json schema
- [ ] validate the reply (recommendation in go_now / wait / unknown; numbers match tool output)
- [ ] fallback when Gemini fails: write the forecast with recommendation "unknown"
- [ ] trigger: new 5-minute window or |wait change| > 2 min
"""


def build_prompt(user_question: str | None = None) -> str:
    raise NotImplementedError


def run_once() -> dict:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
