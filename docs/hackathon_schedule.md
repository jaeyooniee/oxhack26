# OxHack 2026 team schedule (historical)

The plan the four of us worked to during the hackathon. Kept for the record; the README now
describes the project for people outside the team. H1 = board, H2 = build + camera, S1 = model,
S2 = dashboard + Gemini.

Hacking ends **12:00 Sunday**; judging **13:00–15:45**. Three food rushes to record before then.

## Saturday afternoon: everything working separately
| Time | Hardware (H1, H2) | Software (S1, S2) |
|---|---|---|
| 15:30 | Push repo to GitHub (one person) | |
| 15:40 | H1 blink test (swap board by 16:00 if it fails); H2 webcam + USB extension | S1 load Disney data; S2 install requirements, run dashboard |
| 16:15 | H1 ultrasonic → beam → button → LEDs, one at a time; H2 scout food point, ask organisers, build mounts | S1 baseline backtest; S2 dashboard |
| 17:00 | H1 `ingest/logger.py` writing `data/live/run.csv`; H2 `ingest/camera_count.py --show`, set the zone | S1 LightGBM vs baselines ✅ done; S2 Gemini sentence (MLH workshop 17:30, C1) |
| 17:30 | All four: fake queue test (10 min): beam vs button vs camera | |
| 17:45 | Everything mounted and logging at the Food Collection Point | S2 point dashboard at live `run.csv` |

## Saturday evening: first real data
| Time | What happens |
|---|---|
| 18:00–19:00 | **Dinner rush.** One person presses the button for each person served for 20 min. **Write down food announcement + serving start/end times.** |
| 19:00–21:00 | S1 `ingest/state.py` on dinner data, sensor error (beam vs button), forecast-vs-actual for dinner, "minutes to next food drop" feature. S2 ElevenLabs alert, dinner chart. H1 two-beam direction. H2 fix and tidy the mount. |
| 21:00–23:30 | Connect it all: state → forecast → Gemini → `prediction.json` → LEDs + dashboard. C++ estimator only if everything else works. |
| 23:45 | Check logging and camera are running |
| 23:55 | **Save the forecast and commit it to git before the pizza arrives** (timestamped proof) |
| 00:00–00:30 | **Midnight pizza** recorded; compare with the 23:55 forecast |
| 00:30 | Make the demo replayable from the CSV; sleep in shifts, check the logger every 1–2 h |

## Sunday morning: finish and freeze
| Time | What happens |
|---|---|
| 08:00–09:00 | **Breakfast rush** recorded (third test) |
| 09:00 | **Feature freeze.** One person to the Devpost workshop (C1) |
| 09:00–11:00 | Final charts (Disney backtest, three rushes forecast vs actual, sensor vs camera), README, wiring photos, Devpost text, backup demo video |
| 11:00–11:45 | Rehearse the 60-second pitch and judge questions twice (`docs/pitch_notes.md`) |
| 12:00 | Hacking ends. Devpost submitted before this |
| 12:00–13:00 | Leave the logger running: lunch is a live fourth rush during judging |
| 13:00–15:45 | Judging at our table: sensor box, live dashboard, replay as backup, repo QR code |

## What actually happened
Dinner and midnight were not captured (hardware and camera setup took the evening). Two staged
tests were recorded on Sunday morning (`data/ground_truth/NOTES.md`), and lunch ran live during
judging.

## Rules we worked to
1. Nobody waits on anyone: use sample or replay data until the real data arrives.
2. Keep the hand count going: 20 minutes of button presses per rush makes the accuracy claim believable.
3. 09:00 Sunday feature freeze is real. Unfinished extras get cut, not rushed.
