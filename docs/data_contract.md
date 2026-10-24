# Data contract

Every component reads and writes these exact formats. Change them only after telling everyone.
Times: ISO 8601 from the logging laptop's clock (`2026-10-24T18:05:12`), local time.

## 1. Serial line from the Arduino (one JSON object per line, about once per second)
```json
{"ms": 123456, "dist_cm": 182.4, "in": 37, "out": 31, "btn": 12}
```
| Field | Meaning |
|---|---|
| `ms` | Board uptime in ms (`millis()`) |
| `dist_cm` | Ultrasonic distance; `-1` = no echo |
| `in` | Cumulative arrivals counted at the queue entry beam |
| `out` | Cumulative people served, counted at the exit beam |
| `btn` | Cumulative hand-count button presses (ground truth for "served") |

Counters are **cumulative** so a dropped line loses nothing.

## 2. `data/live/run.csv` (written by `ingest/logger.py`)
```
wall_time,ms,dist_cm,in_total,out_total,btn_total
2026-10-24T18:05:12,123456,182.4,37,31,12
```

## 3. `data/live/camera.csv` (written by `ingest/camera_count.py`)
```
wall_time,people_in_zone
2026-10-24T18:05:12,9
```
Counts only. No frames are saved.

## 4. `data/live/queue_state.json` (written by `ingest/state.py`, latest state only)
```json
{
  "timestamp": "2026-10-24T18:05:12",
  "queue_length": 6,
  "arrivals_last_5m": 10,
  "served_last_5m": 8,
  "service_rate_per_min": 1.6,
  "wait_now_min": 3.8,
  "wait_history_min": {"t-15": 2.5, "t-30": 1.0, "t-60": null},
  "mean_wait_today_min": 2.1,
  "median_wait_last_7d_min": null,
  "mins_since_change": 10,
  "venue_open": "18:00",
  "venue_close": "19:30",
  "is_holiday": 0,
  "days_to_holiday": 14
}
```
Fields used by `model/forecast.py`: `timestamp`, `wait_now_min` (required), `wait_history_min`,
`mean_wait_today_min`, `median_wait_last_7d_min` (null at the hackathon = cold start),
`mins_since_change`, `venue_open`, `venue_close` (required), `is_holiday`, `days_to_holiday`.
`mins_since_change`: round the wait to the nearest 5 min first (Disney posted waits move in
5-minute steps), then count minutes since that rounded value last changed.
Optional fields when `ingest/state.py --smooth` is on (Kalman / Bayesian, `model/live/`):
`queue_length_sd`, `arrival_rate_per_min`, `arrival_rate_ci80`, `service_rate_ci80`,
`wait_now_ci80_min` ([P10, P90]), `smoothing`. Readers must ignore fields they don't know.
`wait_instant_min` = queue_length / service_rate_per_min (Little's law), and `wait_now_min` = its
trailing 5-minute average: the venue's equivalent of Disney's (staff-smoothed) posted wait.
That is why the Disney-trained model can take `wait_now_min` as input. Also present when they
apply: `rush` (name from data/rushes.json) and `board_resets` (Arduino resets stitched so far).

## 5. `data/live/prediction.json` (written by `agent/queue_agent.py`)
```json
{
  "timestamp": "2026-10-24T18:05:20",
  "forecasts": [
    {"horizon_min": 15, "p10": 2.0, "p50": 4.5, "p90": 8.0},
    {"horizon_min": 30, "p10": 1.0, "p50": 3.5, "p90": 7.5}
  ],
  "recommendation": "wait",
  "best_time": "18:40",
  "minutes_saved": 9,
  "explanation": "Queue is growing faster than it is served; it usually clears 25 min after the drop.",
  "source": "model:generic_v2 + gemini",
  "wait_now_min": 6.9,
  "generated_at": "2026-10-24T18:05:21",
  "checked_at": "2026-10-24T18:07:41"
}
```
`recommendation` is one of `go_now`, `wait`, `unknown`. The LEDs map these to green / amber / off.
`timestamp` is the queue state's time; `generated_at` when the recommendation was made;
`checked_at` the agent's heartbeat (rewritten every `--watch` interval). The logger turns the
LEDs off when `checked_at` is more than 60 s old.

## 6. `data/ground_truth/hand_count_<rush>.csv`
```
wall_time,event
2026-10-24T18:07:03,served
```

## 7. `data/replay/<rush>_run.csv`
Copy of `run.csv` for one rush (dinner, midnight, breakfast). The demo replays these if the live sensor fails.
