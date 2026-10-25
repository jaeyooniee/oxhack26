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
wall_time,people_in_zone,in_total,out_total,distinct_total,tracked_in_zone
2026-10-24T18:05:12,9,41,32,40,9
```
| Field | Meaning |
|---|---|
| `people_in_zone` | Heads counted in the zone (median over the row's few seconds) |
| `in_total` | Cumulative people who entered the zone (tracker, `ingest/tracker.py`) |
| `out_total` | Cumulative people who left it (served, or walked off) |
| `distinct_total` | Different person IDs counted so far |
| `tracked_in_zone` | `in_total - out_total`: the queue now, including anyone hidden for a moment |

**The camera is the main sensor**: the beams sit too close together to count people, so
`ingest/state.py` takes arrivals and served from `in_total` / `out_total` (and the hand-count
button and ultrasonic from `run.csv`) whenever camera.csv has these columns and is up to date
(`--source auto`, the default; `state.load_counts`). The state then has `"counts_from": "camera"`.
Counters are cumulative and carry on after a restart of `camera_count.py`.
Old files with only `wall_time,people_in_zone` are still read (drift evidence only).
No frames are saved.

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
`wait_instant_min` = queue_length / service_rate_per_min (Little's law), and `wait_now_min` = the
median of its last 3 one-minute values: the venue's equivalent of Disney's (staff-smoothed)
posted wait. That is why the Disney-trained model can take `wait_now_min` as input.
`status`: `open`, or `closed` outside the rush's window (30 min before opening to 15 min after
closing) when nobody is queueing; then `next_rush` = `{"name", "open"}` if a schedule is used.
Also present when they apply: `rush` (name from data/rushes.json), `board_resets` and
`counts_from` (`camera` or `beams`, see §3).

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
`recommendation` is one of `go_now`, `wait`, `unknown`, `closed` (then `forecasts` is empty).
The LEDs map these to green / amber / off / off.
`timestamp` is the queue state's time; `generated_at` when the recommendation was made;
`checked_at` the agent's heartbeat (rewritten every `--watch` interval). The logger turns the
LEDs off when `checked_at` is more than 60 s old.

## 6. `data/ground_truth/hand_count_<rush>.csv`
```
wall_time,event
2026-10-24T18:07:03,served
```
Written by `python ingest/camera_accuracy.py tally` (Enter = one person served) when the
button is not used; otherwise the button's `btn_total` in run.csv is the hand count.

## 7. `data/replay/<rush>_run.csv` and `<rush>_camera.csv`
`run.csv` and `camera.csv` cut to one rush (dinner, midnight, breakfast), written by
`ingest/archive.py`. The demo replays these if the live sensor fails; the camera file next to a
run file is picked up automatically (and its tracker counts used, as live).

## 8. `data/live/detections.jsonl` (written by `ingest/camera_count.py`, unless `--no-log`)
One JSON object per line. A header each time camera_count.py starts, then one line per frame:
```
{"start": "2026-10-24T18:00:02", "detector": "yolov8n", "frame": [480, 640], "zone": [[64, 200], ...]}
{"t":"2026-10-24T18:05:12.345","b":[[x1, y1, x2, y2, "<64 hex chars>"], ...]}
```
Boxes in pixels of the resized frame; the hex string is the person's 32-number colour fingerprint
(hue x saturation histogram of the torso). **No pixels.** Used to re-run and tune the tracker on
a recorded rush: `python ingest/camera_accuracy.py tune`. About 5-15 MB per hour.

## 9. `data/ground_truth/queue_count_<rush>.csv` (written by `python ingest/camera_accuracy.py note N`)
```
wall_time,people
2026-10-24T18:12:40,7
```
Someone counts the people in the queue by eye now and then; the camera's queue length is scored
against it.

## 10. Where the data is kept (`ingest/archive.py`)
`data/live/` is not committed. `python ingest/archive.py --watch 10 --commit --push` copies it
every 10 minutes into folders that are:
| Path | What |
|---|---|
| `data/archive/<date>/run.csv`, `camera.csv` | Everything logged (date = the day logging started) |
| `data/archive/<date>/detections_<rush>.jsonl.gz` | §8, cut to each rush |
| `data/archive/<date>/predictions.jsonl` | Every new `prediction.json` (§5), timestamped |
| `data/archive/<date>/*.json` | Camera zone, ultrasonic calibration, tracker settings |
| `data/replay/<rush>_run.csv`, `<rush>_camera.csv` | Each rush cut out (§7) |
| `data/ground_truth/` | §6, §9 |
