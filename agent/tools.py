"""Python tools Gemini can call (owner: S2). All maths lives here, never in the LLM.

TODO
- [ ] get_queue_state() -> dict          (read data/live/queue_state.json)
- [ ] get_forecast(horizons) -> list     (call model.forecast.forecast)
- [ ] compare_slots(slots, max_wait_min) -> list
      for each slot: expected wait (P50) and P(wait > max_wait_min), interpolated from P10/P50/P90
"""


def get_queue_state() -> dict:
    """Latest live queue state: length, arrival and service rates, current wait estimate."""
    raise NotImplementedError


def get_forecast(horizons: list[int]) -> list[dict]:
    """Quantile forecasts (P10/P50/P90, minutes) of the wait for each horizon in minutes."""
    raise NotImplementedError


def compare_slots(slots: list[str], max_wait_min: float) -> list[dict]:
    """For each clock time in `slots`, expected wait and probability it exceeds max_wait_min."""
    raise NotImplementedError
