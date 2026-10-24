# QueueTime: forecast the queue, spend your time elsewhere

OxHack 2026 · HRT challenge "Best use of data for prediction & decision making" · theme: TIME

A £10 sensor measures a real queue (the OxHack food point). A model trained on years of public
Disney World wait-time data forecasts the wait 15–60 minutes ahead with honest uncertainty
bands. A Gemini agent turns the numbers into a decision: "go now" or "go at 18:40, save ~9 min".

## How it fits together

```
 hardware/          ingest/              model/                 agent/              app/
 Arduino sensors -> logger.py ---------> forecast.py ---------> queue_agent.py ---> app.py (Streamlit)
 (ultrasonic,       data/live/run.csv    (Disney-trained         (Gemini function   LEDs, voice alert
  beam, button)     state.py             generic v2 model,       calling: calls
 camera ----------> camera_count.py      P10/P50/P90)            our tools, writes
                    data/live/           live/ (Kalman +         prediction.json)
                    queue_state.json     Bayesian rates)
```

Numbers always come from Python (sensors, model, queueing maths). Gemini chooses which tools to
call, combines the results and explains the decision. It never invents a wait time.

## Folder map

| Folder | What lives there | Owner |
|---|---|---|
| `hardware/` | Wiring, parts list, Arduino firmware | H1 (board), H2 (build + camera) |
| `ingest/` | Serial → CSV logger, camera head count, live queue state | H1 / H2 |
| `model/` | `forecast.py` (one call: queue state → P10/P50/P90 for 15/30/60 min), exported models in `artifacts/`, live Kalman model | S1 |
| `claude_model/` | Training and validation on Disney data (already done, see its README) | S1 |
| `agent/` | Gemini function-calling agent | S2 + Gemini team |
| `app/` | Streamlit dashboard, ElevenLabs alert | S2 |
| `data/` | `live/` (today), `replay/` (recorded rushes), `ground_truth/`, `sample/` (fake rows for dev) | everyone |
| `docs/` | Data contract, architecture, timeline, pitch notes, open questions | everyone |
| `notebooks/` | Disney backtest notebook and result charts | S1 |

## Using the forecast
```python
from model.forecast import forecast
forecast(queue_state)  # -> [{"horizon_min": 15, "p10": .., "p50": .., "p90": ..}, ... 30, 60]
```
Try it: `python model/forecast.py data/sample/queue_state_sample.json`. Test scores in `model/README.md`.

## Schedule

Hacking ends **12:00 Sunday**; judging **13:00–15:45**. Three food rushes to record before then.
H1 = board, H2 = build + camera, S1 = model, S2 = dashboard + Gemini.

### Saturday afternoon: everything working separately
| Time | Hardware (H1, H2) | Software (S1, S2) |
|---|---|---|
| 15:30 | Push repo to GitHub (one person) | |
| 15:40 | H1 blink test (swap board by 16:00 if it fails); H2 webcam + USB extension | S1 load Disney data; S2 install requirements, run dashboard |
| 16:15 | H1 ultrasonic → beam → button → LEDs, one at a time; H2 scout food point, ask organisers, build mounts | S1 baseline backtest; S2 dashboard |
| 17:00 | H1 `ingest/logger.py` writing `data/live/run.csv`; H2 `ingest/camera_count.py --show`, set the zone | S1 LightGBM vs baselines ✅ done; S2 Gemini sentence (MLH workshop 17:30, C1) |
| 17:30 | All four: fake queue test (10 min): beam vs button vs camera | |
| 17:45 | Everything mounted and logging at the Food Collection Point | S2 point dashboard at live `run.csv` |

### Saturday evening: first real data
| Time | What happens |
|---|---|
| 18:00–19:00 | **Dinner rush.** One person presses the button for each person served for 20 min. **Write down food announcement + serving start/end times.** |
| 19:00–21:00 | S1 `ingest/state.py` on dinner data, sensor error (beam vs button), forecast-vs-actual for dinner, "minutes to next food drop" feature. S2 ElevenLabs alert, dinner chart. H1 two-beam direction. H2 fix and tidy the mount. |
| 21:00–23:30 | Connect it all: state → forecast → Gemini → `prediction.json` → LEDs + dashboard. C++ estimator only if everything else works. |
| 23:45 | Check logging and camera are running |
| 23:55 | **Save the forecast and commit it to git before the pizza arrives** (timestamped proof) |
| 00:00–00:30 | **Midnight pizza** recorded; compare with the 23:55 forecast |
| 00:30 | Make the demo replayable from the CSV; sleep in shifts, check the logger every 1–2 h |

### Sunday morning: finish and freeze
| Time | What happens |
|---|---|
| 08:00–09:00 | **Breakfast rush** recorded (third test) |
| 09:00 | **Feature freeze.** One person to the Devpost workshop (C1) |
| 09:00–11:00 | Final charts (Disney backtest, three rushes forecast vs actual, sensor vs camera), README, wiring photos, Devpost text, backup demo video |
| 11:00–11:45 | Rehearse the 60-second pitch and judge questions twice (`docs/pitch_notes.md`) |
| 12:00 | Hacking ends. Devpost submitted before this |
| 12:00–13:00 | Leave the logger running: lunch is a live fourth rush during judging |
| 13:00–15:45 | Judging at our table: sensor box, live dashboard, replay as backup, repo QR code |

## Start here
1. Read `docs/data_contract.md`. It is the agreement between all four people.
2. Build against `data/sample/` until real data arrives (rule 1: nobody waits on anyone).
3. Check `docs/open_questions.md` before the pitch.

## Rules
1. Nobody waits on anyone: use sample or replay data until the real data arrives.
2. Keep the hand count going: 20 minutes of button presses per rush makes the accuracy claim believable.
3. 09:00 Sunday feature freeze is real. Unfinished extras get cut, not rushed.

## Setup
```
pip install -r requirements.txt
cp .env.example .env      # add GEMINI_API_KEY and ELEVENLABS_API_KEY
```

## Run it
Live (four terminals, `--port` from `python ingest/logger.py` with no arguments):
```
python ingest/logger.py --port COM3
python ingest/state.py --open 18:00 --close 19:30 --watch 10
python agent/queue_agent.py --watch 10          # rules only without GEMINI_API_KEY, or --no-gemini
streamlit run app/app.py
```
Extras (each optional, each its own terminal):
```
python app/alerts.py --watch 5                  # reads the recommendation aloud (ElevenLabs, else laptop voice)
python ingest/camera_count.py --set-zone        # once: click the queue area's corners
python ingest/camera_count.py --show            # head count -> data/live/camera.csv
python ingest/state.py ... --smooth             # Kalman/Bayesian smoothing + 80% wait range
python ingest/state.py ... --us-empty 400 --us-cm-per-person 60   # + ultrasonic fusion (calibrate!)
```
Firmware: `hardware/firmware/queue_sensor/queue_sensor.ino` (pins in `hardware/README.md`).

Replay (no sensor needed; also the backup demo):
```
streamlit run app/app.py -- --replay data/sample/fake_rush_run.csv --speed 30
```
`data/sample/fake_rush_run.csv` (+ `fake_rush_camera.csv`) is a simulated 90-minute rush
(`python ingest/fake_rush.py`). Add `--us-empty 400 --us-cm-per-person 25` and tick
"Kalman smoothing" in the sidebar to see the error bars. Copy
each real rush to `data/replay/<rush>_run.csv` and replay that instead.
