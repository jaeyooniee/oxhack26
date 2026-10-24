# Architecture and process

## Step by step (one live update)
1. **Sense**: the Arduino sends one JSON line per second over USB serial (`docs/data_contract.md` §1).
2. **Log**: `ingest/logger.py` adds the laptop time and appends to `data/live/run.csv`.
   The same script also sends the LED command back down the serial port, because only one
   program can hold the port open at a time.
3. **Count check**: `ingest/camera_count.py` writes head counts to `data/live/camera.csv`
   (used to measure the sensor's accuracy, not as the main input).
4. **State**: `ingest/state.py` turns cumulative counts into arrival rate λ, service rate μ,
   queue length L and wait W = L / μ, and writes `data/live/queue_state.json`.
   The live Kalman / Bayesian model (`model/live/`) smooths these and gives error bars.
5. **Forecast**: `model/forecast.py` feeds the recent wait history into the Disney-trained
   generic v2 model and returns P10 / P50 / P90 for each horizon.
6. **Decide**: `agent/queue_agent.py` calls Gemini with function calling. Gemini calls our tools
   (`get_queue_state`, `get_forecast`, `compare_slots`), then writes `data/live/prediction.json`
   with a recommendation and a one-sentence explanation.
7. **Show**: `app/app.py` (Streamlit) shows the live queue, forecast band and recommendation;
   the logger lights the LEDs; `app/alerts.py` reads the recommendation out (ElevenLabs).

## Why the Disney model can run on the food queue
The generic v2 model uses only unit-free inputs (recent trend ratios, wait vs today so far,
wait vs the venue's previous 7 days, time since opening / to closing, how long the wait has
been flat). It was tested on rides it never saw: 18% better than "no change" with no adaptation,
21% after one week, with 80% bands that are right 80% of the time. See `claude_model/README.md`.

## Where Gemini sits, and where it does not
- Gemini **does**: choose which tools to call, combine their results, write the recommendation and explanation.
- Gemini **does not**: compute wait times or forecasts. Those come from Python so they are
  reproducible and testable.
- Gemini only runs when the state changes meaningfully (for example every new 5-minute window),
  not every second, to save latency and API calls.
