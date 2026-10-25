# QueueTime: forecast the queue, spend your time elsewhere

OxHack 2026 · HRT challenge "Best use of data for prediction & decision making" · theme: TIME

A £10 sensor measures a real queue (the OxHack food point). A model trained on years of public
Disney World wait-time data forecasts the wait 15–60 minutes ahead with honest uncertainty
bands. A Gemini agent turns the numbers into a decision: "go now" or "go at 18:40, save ~9 min".

## How it fits together

```
 hardware/          ingest/              model/                 agent/              app/
 Arduino sensors -> logger.py ---------> forecast.py ---------> queue_agent.py ---> server.py (web + phone app)
 (ultrasonic,       data/live/run.csv    (Disney-trained         (Gemini function   LEDs, voice alert
  beam, button)     state.py             generic v2 model,       calling: calls
 camera ----------> camera_count.py +    P10/P50/P90)            our tools, writes
(counts the queue)  tracker.py           live/ (Kalman +         prediction.json)
                    data/live/           Bayesian rates)
                    queue_state.json
```

Numbers always come from Python (sensors, model, queueing maths). Gemini chooses which tools to
call, combines the results and explains the decision. It never invents a wait time.

## Folder map

| Folder | What lives there | Owner |
|---|---|---|
| `hardware/` | Wiring, parts list, Arduino firmware | H1 (board), H2 (build + camera) |
| `ingest/` | Camera people tracker (main sensor) and its accuracy check, serial → CSV logger, live queue state, data archive | H1 / H2 |
| `model/` | `forecast.py` (one call: queue state → P10/P50/P90 for 15/30/60 min), exported models in `artifacts/`, live Kalman model | S1 |
| `claude_model/` | Training and validation on Disney data (already done, see its README) | S1 |
| `agent/` | Gemini function-calling agent | S2 + Gemini team |
| `app/` | `server.py` + `web/` (web dashboard and installable phone app), Streamlit dashboard (`app.py`), ElevenLabs alert | S2 |
| `data/` | `live/` (today, not in git), `archive/` (everything logged, committed by `ingest/archive.py`), `replay/` (each rush), `ground_truth/`, `sample/` (fake rows for dev) | everyone |
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
The beams sit too close together to count people, so **the camera counts the queue**: it follows
every person with an ID and counts who arrives and who is served (`ingest/tracker.py`). The board
still gives the hand-count button, the ultrasonic and the LEDs.

Live (six terminals, `--port` from `python ingest/logger.py` with no arguments):
```
python ingest/camera_count.py --set-zone        # once: click the queue area's corners, Enter
python ingest/camera_count.py                   # people tracker -> data/live/camera.csv (--show to watch it)
python ingest/logger.py --port COM3             # button, ultrasonic, LEDs -> data/live/run.csv
python ingest/state.py --smooth --watch 10      # counts from the camera; rush times from data/rushes.json
python agent/queue_agent.py --watch 10          # rules only without GEMINI_API_KEY, or --no-gemini
python app/server.py                            # web + phone app; prints the address to open
python ingest/archive.py --watch 10 --commit --push   # keeps all the sensor data (below)
python ingest/tigerdata.py sync --watch 10      # streams it into Tiger Cloud (below)
```
Leave them all running all night: the logger reconnects if the board resets or is unplugged,
`camera_count.py` carries its counters on after a restart, `state.py` stitches the counters back
together and switches rush automatically. `state.py` prints `counts from camera`; if the camera
stops for more than 2 minutes it falls back to the beams (`--source camera` / `--source beams` to force).

### Where the sensor data is kept
`data/live/` is not in git. `ingest/archive.py --watch 10 --commit --push` copies it every 10 min
into `data/archive/<date>/` (everything logged, every forecast made, camera detections per rush)
and cuts each rush into `data/replay/<rush>_run.csv` + `<rush>_camera.csv`, then commits and
pushes, so the data is on GitHub, timestamped, and off the laptop. Details: `docs/data_contract.md` §10.
Leave out `--push` if the wifi is bad; it commits locally and pushes on a later round.

### Time series in Tiger Cloud
`ingest/tigerdata.py` streams every new camera, board, queue-state, forecast and hand-count row
into Tiger Cloud hypertables (TimescaleDB), with a per-minute continuous aggregate `queue_1min`
that the database keeps up to date. Once:
1. Tiger Console → *New service* → download the config; put its connection string in `.env` as
   `TIGER_DATABASE_URL=postgres://...`.
2. `python ingest/tigerdata.py setup` (creates the hypertables and `queue_1min`).

Then leave `python ingest/tigerdata.py sync --watch 10` running. Only new rows are sent, so it can
be stopped, restarted or left offline and catches up. `python ingest/tigerdata.py query` shows the
last 30 min per minute; in Tiger Console's SQL editor, e.g.
`SELECT * FROM queue_1min ORDER BY minute DESC LIMIT 30;`
A recorded night loads the same way: `sync --live data/archive/<date> --site replay`.
Tables: `docs/data_contract.md` §11.

### How well does the camera count? (and tuning it)
```
python ingest/camera_accuracy.py note 7          # whenever you can: "7 people in the queue now"
python ingest/camera_accuracy.py tally           # if the button is not there: Enter per person served
python ingest/camera_accuracy.py report --rush dinner      # camera vs hand count
python ingest/camera_accuracy.py tune --rush dinner --save # re-run the tracker on the recorded
                                                           # detections, keep the best settings
```
The served count is the test for distinct people: someone lost behind another person and given
a new ID is counted twice, so *camera served > hand count* means double counting, *<* means
missed people. On a simulated rush (people hidden 1-6 s at a time) re-identification by clothing
colour cut 296 counted arrivals to 28 (truth 28); tuning on one rush took the served count from
38 to 30 (truth 30). Tune on one rush, check with `report` on the next before quoting a number.

### Web and phone app
`python app/server.py` prints two addresses (the laptop's wifi IP, port 8000):
- `http://<ip>:8000/` is the web dashboard (projector, judges' table).
- `http://<ip>:8000/phone` is the phone app. Anyone on the same wifi opens it; *Add to Home
  Screen* (iOS Share menu, Android browser menu) puts it on their phone like an app.
- *Ping me at 18:35* beeps, vibrates and shows a full-screen alarm when the time comes, while the
  page is open. System notifications in the background need https (browsers block them over
  plain http on a local network), so tell people to keep the page open.
- If Windows asks whether Python may accept connections, allow it on *private* networks, or
  phones can't connect. Hackathon wifi that isolates devices also blocks it: then use a phone
  hotspot from the laptop.
- Replay for the demo backup: `python app/server.py --replay data/replay/dinner_run.csv --speed 30`;
  the web page shows pause and jump controls.
- The Streamlit dashboard (`streamlit run app/app.py`) still works and shows the same numbers.

### Checklist for each rush
1. **Times:** as soon as the organisers announce it, correct the rush's `open`/`close` in
   `data/rushes.json` (no restart needed). `state.py` prints `rush: <name>` when it switches; check it.
2. **Ultrasonic** (once, at the food point, logger running): with nobody queueing run
   `python ingest/calibrate_ultrasonic.py --empty`, then with N people standing in line
   `python ingest/calibrate_ultrasonic.py --people N`. `state.py` picks it up on its next cycle.
   Beam drift (the exit beam missing people) is only reset when another sensor proves nobody
   is there: the calibrated ultrasonic for 1 min or the camera for 2 min (each only after it has
   seen someone, so a dead sensor never counts), or after closing time. A short queue standing
   still before the food comes out is never wiped. Without ultrasonic or camera, up to 3
   phantom people can stay on screen until closing.
3. **Hand count:** someone presses the button (or runs `camera_accuracy.py tally`) for every
   person served for 20 minutes, and now and then counts the queue: `camera_accuracy.py note N`.
4. **After the rush:** `archive.py` has already cut it into `data/replay/<rush>_run.csv` and
   `<rush>_camera.csv` (run `python ingest/archive.py` once to be sure). Check the camera:
   `python ingest/camera_accuracy.py report --rush <rush>`, then the forecast:
   `python model/backtest.py data/replay/<rush>_run.csv`. After two rushes, calibrate the bands:
   `python model/backtest.py data/replay/dinner_run.csv data/replay/midnight_run.csv --save`
   and check on the next one with `--check`.
5. **Between rushes** the dashboard says *Closed* (with the next rush) and the LEDs are off.
   If the LEDs go off or the dashboard says *No update from the agent* during a rush, a
   process has stopped: restart it.

Tests: `python tests/test_pipeline.py` (45 checks, about 50 s, includes both dashboards, the app server,
the people tracker and the archive).

Extras (each optional, each its own terminal):
```
python app/alerts.py --watch 5                  # reads the recommendation aloud (ElevenLabs, else laptop voice)
python ingest/fake_camera.py                    # fake detections + true serves, to try camera_accuracy.py
```
Firmware: `hardware/firmware/queue_sensor/queue_sensor.ino` (pins in `hardware/README.md`).

Replay (no sensor needed; also the backup demo):
```
python app/server.py --replay data/sample/fake_rush_run.csv --speed 30
streamlit run app/app.py -- --replay data/sample/fake_rush_run.csv --speed 30
```
`data/sample/fake_rush_run.csv` (+ `fake_rush_camera.csv`) is a simulated 90-minute rush
(`python ingest/fake_rush.py`). Add `--us-empty 400 --us-cm-per-person 25` and tick
"Kalman smoothing" in the sidebar to see the error bars. Replay a real rush from
`data/replay/<rush>_run.csv` instead (its `_camera.csv` is picked up automatically).
