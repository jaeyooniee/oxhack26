"""Live queue state (owner: S1 / H1).

Reads data/live/run.csv and writes the latest state to data/live/queue_state.json
(docs/data_contract.md §4).

  arrivals_last_5m = in_total(now) - in_total(now - 5 min)
  served_last_5m   = out_total(now) - out_total(now - 5 min)
  queue_length     = in_total - out_total (reset to 0 when the venue is empty for 2+ min)
  service_rate     = served_last_5m / 5          (people per minute)
  wait_now_min     = queue_length / service_rate (Little's law)

The wait history (t-15, t-30, t-60) and today's mean wait are what model/forecast.py needs.

TODO
- [ ] compute the fields above; guard against service_rate == 0
- [ ] optional: pass the counts through model/live (Kalman) for smoothed values + error bars
- [ ] run every 10 s (or on each new row)
"""


def compute_state(run_rows) -> dict:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
