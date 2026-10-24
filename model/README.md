# Model (owner: S1)

## Use it (this is all the agent and dashboard need)
```python
from model.forecast import forecast
forecast(queue_state)   # -> [{"horizon_min": 15, "p10": .., "p50": .., "p90": ..}, {"horizon_min": 30, ...}, {"horizon_min": 60, ...}]
```
`queue_state` = `data/live/queue_state.json` (`docs/data_contract.md` §4). Try it:
`python model/forecast.py data/sample/queue_state_sample.json`

## What's inside
- `artifacts/h15`, `h30`, `h60`: `q10.txt` / `q50.txt` / `q90.txt` (LightGBM) + `meta.json`
  (feature order, target, how to convert to minutes, test scores). About 10 MB, safe to commit.
- `export.py`: rebuilds the artifacts from the Disney data (`python model/export.py --horizons 15 30 60`).
- `live/`: Kalman filter + Bayesian rate model. Written from scratch: `model.py` from
  queuetime.zip was never added to the repo.
- `backtest.py`: replays a recorded rush, scores the forecast and fits the band calibration
  (`artifacts/food_calibration.json`, applied automatically by `forecast.py` when present).
- Training and validation experiments: `claude_model/`.

## Test scores (Disney 2019, never seen in training; MAE in minutes)
| Horizon | "No change" | Model | Model, no 7-day history (cold start) | 80% band coverage |
|---|---:|---:|---:|---:|
| 15 min | 6.85 | 6.61 | 6.69 | 0.81 |
| 30 min | 10.83 | 9.91 | 10.29 | 0.81 |
| 60 min | 15.24 | 12.53 | 13.59 | 0.81 |

These are Disney numbers. On the simulated food rush the uncalibrated P10-P90 band held the truth
only 20-55% of the time; after calibrating on two rushes it held 75-78% on a third. Check on the
real rushes with `model/backtest.py` before quoting any coverage for the food queue.
