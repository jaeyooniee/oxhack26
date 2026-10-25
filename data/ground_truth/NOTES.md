# Ground truth: what actually happened (read before analysing)

Two staged tests on 2026-10-25, both with the iPhone as the camera (`--cam 1`). The
`hand_count_breakfast.csv` and `queue_count_breakfast.csv` files hold **both** sessions;
split them by time.

## Session 1: "breakfast", 08:30:43–08:40:15
- **3 people** looping: served at the front, walked out of view, rejoined at the back.
- **30 served** (hand count, `tally`). The first 20 presses are within 75 s (08:30:43–08:31:58),
  then a 5-min gap: check with the team whether that was real or catching up.
- `queue_count_breakfast.csv` says **5** for the three 08:35–08:40 notes; the real number was **3**.
- Detections: `data/archive/2026-10-25/detections_breakfast.jsonl.gz`; replay files
  `data/replay/breakfast_*.csv`.
- Result with default settings: camera 35 served vs 30 hand (+17%); queue length about right
  on average once the truth is 3.
- Use this session for the **served** count: people really left the camera's view.

## Session 2: "staged_1000", 10:01:11–10:05:25
- **5 people** moving around **inside** the camera's view the whole time, 12 "serves".
- **12 served** (hand count). `queue_count_breakfast.csv` says **4** for the 09:51–10:05 notes;
  the real number was **5**.
- Detections: `data/archive/2026-10-25/detections_staged_1000.jsonl.gz` (header line has the zone).
- The zone drawn at 09:58 was too small: about a quarter of foot positions sat on its edge, so
  people flickered in and out and the served count came out at 50. Replaying with a zone that
  covers where people stood gives queue length within about 1 person, but the served count is
  meaningless here because nobody left the view (same 5 people got 74 track IDs).
- Use this session for **detection** only: 4.4 people seen per frame for 5 present, within 1 in
  87% of frames.

## Commands
```
python ingest/camera_accuracy.py report --rush breakfast --to 2026-10-25T09:00:00
python ingest/camera_accuracy.py report --rush breakfast --from 2026-10-25T09:50:00 --to 2026-10-25T10:08:00 \
    --log data/archive/2026-10-25/detections_staged_1000.jsonl.gz
python ingest/camera_accuracy.py tune --rush breakfast --to 2026-10-25T09:00:00      # 82 settings, ~1 min
python model/backtest.py data/replay/breakfast_run.csv
```
Tuning on the breakfast session alone did not hold up: tune on one half, test on the other, and
the tuned settings were no better than the defaults. Don't `--save` without a second real rush.
