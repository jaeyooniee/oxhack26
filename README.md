# QueueTime: forecast the queue, spend your time elsewhere

A £10 sensor box and a phone camera measure a real queue. A model trained on years of public
Disney World wait-time data forecasts the wait 15–60 minutes ahead with honest uncertainty
bands. A Gemini agent turns the numbers into a decision, "go now" or "go at 12:40, save ~9 min",
spoken aloud by ElevenLabs, shown on a web dashboard and an installable phone app, and lit up
on LEDs at the queue. Every reading streams into Tiger Cloud (TimescaleDB) so the whole day can
be queried.

Built in 24 hours at OxHack 2026 for the HRT challenge "best use of data for prediction &
decision making" (theme: TIME) and tested on the hackathon's own food queue. MIT licensed:
take it, point it at your queue, and change whatever you like. The last section says where to
start.

## How it fits together

```
 Arduino UNO R4 WiFi ──USB──► ingest/logger.py ──► data/live/run.csv ─┐
 (ultrasonic, button, LEDs;                                            │
  on-board queue estimate)                                             ▼
                                                              ingest/state.py ──► queue_state.json
 iPhone / webcam ──► ingest/camera_count.py ──► camera.csv ───┘              │
 (YOLO people tracker;                                                       ▼
  counts only, no images)                              model/forecast.py (LightGBM, P10/P50/P90)
                                                                             │
                                                                             ▼
                              agent/queue_agent.py (Gemini function calling) ──► prediction.json
                                                                             │
                 ┌───────────────────────────┬───────────────────────────────┤
                 ▼                           ▼                               ▼
   app/server.py (dashboard + phone app)   app/alerts.py (ElevenLabs voice)   LEDs (back over USB)

 ingest/tigerdata.py streams every row into Tiger Cloud; ingest/archive.py commits it to git.
```

Numbers always come from Python (sensors, model, queueing maths). Gemini chooses which tools to
call, combines the results and explains the decision; if it invents a number the rule-based
answer replaces it, and if it is unavailable the rules take over and the system keeps running.

## Quick start without any hardware (5 minutes)

```
git clone https://github.com/jaeyooniee/oxhack26.git && cd oxhack26
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/server.py --replay data/replay/breakfast_run.csv --speed 30
```

Open the address it prints. That is a real morning at the OxHack food point replayed at 30×:
the queue, the wait, the forecast bands and the recommendation, with pause and jump controls.
`data/sample/fake_rush_run.csv` is a simulated 90-minute rush if you want something busier.

Try the forecast on its own:
```python
from model.forecast import forecast
forecast(queue_state)  # -> [{"horizon_min": 15, "p10": .., "p50": .., "p90": ..}, ... 30, 60]
```
`python model/forecast.py data/sample/queue_state_sample.json` prints one. Scores in `model/README.md`.

Tests: `python tests/test_pipeline.py` (45 checks, about 50 s). On Apple Silicon, LightGBM needs
`libomp` (`brew install libomp` with an arm64 Homebrew).

## Running it on a real queue

### Parts (about £10 plus a phone or webcam)
- Arduino UNO R4 WiFi (Arm Cortex-M4) and a USB **data** cable
- HC-SR04 ultrasonic distance sensor
- Push button (hand count for checking accuracy)
- LEDs (green = go now, amber = wait) with 220 Ω resistors
- Breadboard and jumpers
- A camera the laptop can open: an iPhone over Continuity Camera, or any USB webcam

Wiring and a step-by-step build with a check after each step: `hardware/README.md`.
Firmware: `queuetime_v2/queue_sensor.ino` (the version that ran at the hackathon; it also
estimates arrival rate, service rate and the wait on the board itself). Upload it with the
Arduino IDE, open the Serial Monitor at 115200 baud and you should see one JSON line per second.
Close the Serial Monitor before running the logger: only one program can use the port.

### Keys
```
cp .env.example .env
```
Add `GEMINI_API_KEY` (aistudio.google.com; the free tier allows 20 requests per day per
model), `ELEVENLABS_API_KEY` (optional; without it the laptop's own voice is used) and
`TIGER_DATABASE_URL` (optional; from Tiger Console). Everything runs without any key: rules
replace Gemini, the OS voice replaces ElevenLabs, and the CSV files are the database.

### Start it (one terminal each; leave them running)
```
python ingest/camera_count.py --set-zone --cam 1   # once: click the corners of the queue floor, Enter
python ingest/camera_count.py --cam 1 --show       # people tracker -> data/live/camera.csv
python ingest/logger.py --port /dev/cu.usbmodem…   # board -> data/live/run.csv (run with no --port to list ports)
python ingest/state.py --smooth --watch 10         # queue state from the counts
python agent/queue_agent.py --watch 10             # Gemini agent (rules only without a key, or --no-gemini)
python app/server.py                               # web dashboard + phone app, prints the address
python app/alerts.py --watch 5                     # voice alert when the recommendation changes
python ingest/tigerdata.py sync --watch 10         # stream into Tiger Cloud (after `tigerdata.py setup` once)
python ingest/archive.py --watch 10 --commit --push  # keep every reading in git
```
`--cam 0` is usually the laptop's own camera and `--cam 1` an attached phone or webcam. Draw
the zone generously: it should contain everyone who is queueing, with a margin, and the only
edge people cross should be the front, where they are served and walk away. A zone that is too
tight makes people flicker in and out and inflates the served count (we learned this the hard
way, see `data/ground_truth/NOTES.md`).

The processes are independent: the logger reconnects if the board resets, the camera carries
its counters across a restart, and `state.py` stitches the counters back together. Opening and
closing times of each rush live in `data/rushes.json`; between rushes the dashboard says *Closed*.

### Phone app
`python app/server.py` prints the laptop's wifi address. Anyone on the same network opens
`http://<ip>:8000/phone` and chooses *Add to Home Screen*. "Ping me at 12:35" alarms when the
time comes while the page is open. If the venue wifi isolates devices, use a phone hotspot.

### Checking the camera against a hand count
```
python ingest/camera_accuracy.py tally            # Enter once per person served
python ingest/camera_accuracy.py note 7           # "7 people in the queue right now"
python ingest/camera_accuracy.py report --rush lunch
python ingest/camera_accuracy.py tune --rush lunch --save   # fit the tracker to the hand count
```
Tune on one rush and check with `report` on another before trusting the result: on a single
10-minute recording, tuned settings fitted that recording and nothing else.

### Time series in Tiger Cloud
`ingest/tigerdata.py` keeps one hypertable per source (`board_readings`, `camera_counts`,
`queue_states`, `predictions`, hand counts) and a per-minute continuous aggregate `queue_1min`.
Only new rows are sent, so it can be stopped and restarted freely. In Tiger Console's SQL editor:
`SELECT * FROM queue_1min ORDER BY minute DESC LIMIT 30;` or, in a terminal,
`python ingest/tigerdata.py query`. Tables: `docs/data_contract.md` §11.

## What we measured

| Test | Result |
|---|---|
| People detection (5 people, 10:00 test) | 4.4 seen per frame on average; within ±1 of the truth in 87% of frames |
| People served (breakfast, 3 people looping, untuned) | camera 35 vs hand count 30 (+17%) |
| Forecast vs "no change" baseline (breakfast, MAE) | 15 min: 0.71 vs 0.72 · 30 min: **0.58 vs 0.66** · 60 min: **0.76 vs 0.84** |
| 80% band coverage (breakfast) | 43% / 80% / 70% at 15 / 30 / 60 min |
| Disney 2019 holdout (never seen in training) | model beats "no change" at every horizon; see `model/README.md` |

Breakfast waits never exceeded two minutes, so the live errors are small because the queue was.
`data/ground_truth/NOTES.md` has what really happened in each test, including our own counting
mistakes. The camera never stores images: only box coordinates and counts.

## Folder map

| Folder | What lives there |
|---|---|
| `queuetime_v2/` | Firmware that ran on the board: sensors, on-board queue estimate, one JSON line per second |
| `hardware/` | Parts, wiring, build steps; `firmware/` holds the earlier, simpler firmware |
| `ingest/` | Camera people tracker (`camera_count.py`, `tracker.py`), serial logger, queue state, accuracy check, archive, Tiger Cloud sync |
| `model/` | `forecast.py` (queue state → P10/P50/P90 for 15/30/60 min), exported LightGBM models, Kalman live model, backtest |
| `claude_model/` | Training and validation on the Disney data |
| `agent/` | Gemini function-calling agent and its tools |
| `app/` | `server.py` + `web/` (dashboard and phone app), Streamlit dashboard, ElevenLabs alert |
| `data/` | `live/` (today, not in git), `archive/` (everything logged), `replay/` (each rush), `ground_truth/`, `sample/` |
| `docs/` | Data contract, architecture, open questions, pitch notes, hackathon schedule |

## Adapting it to your queue, and where to take it next

The pieces talk through small CSV and JSON files described in `docs/data_contract.md`, so each
can be replaced on its own.

- **A different queue.** Edit `data/rushes.json` with your opening hours and draw a new zone.
  Nothing else assumes a food point.
- **A different camera or detector.** `camera_count.py` takes any OpenCV camera index or a
  video file. It uses YOLOv8n when `ultralytics` is installed and OpenCV's HOG detector
  otherwise; swap in your own detector by returning boxes to `Tracker`.
- **No laptop: run it on Arm.** The tracker is plain Python + OpenCV + YOLOv8n, so it runs on a
  Raspberry Pi 5 or any Arm single-board computer with a camera, with the Arduino on its USB
  port. The UNO R4's Cortex-M4 already computes arrival rate, service rate and an Erlang wait
  probability on the board (`queuetime_v2/queue_sensor.ino`), so a next step is to move the
  "go now / wait" decision there too and drive the LEDs with no computer at all.
- **A different model.** `model/forecast.py` is one function from a queue state to three
  quantiles. Retrain with `model/export.py` on your own history, or replace it with anything
  that returns the same shape. The Disney data and experiments are in `claude_model/`.
- **A different LLM or voice.** `agent/queue_agent.py` uses Gemini function calling against
  the tools in `agent/tools.py`; the tools are the contract, the model is a few lines.
  `app/alerts.py` calls ElevenLabs' REST API and falls back to the OS voice.
- **A different database.** `ingest/tigerdata.py` is plain SQL through psycopg; it creates
  hypertables when TimescaleDB is present and ordinary tables when it is not.

Known gaps, honestly: `docs/open_questions.md`. Pull requests welcome.

## Team
Built at OxHack 2026 by the QueueTime team. MIT License.
