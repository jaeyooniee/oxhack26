# Pitch notes

## Opening line
"HRT models how long an order sits in a queue before it fills. We built that for the queue you were just standing in."

## Numbers we can defend (generic v2 model, 60 min ahead, Disney data)
- Unseen venues (each ride hidden from training, 2019): MAE 12.4 min vs 15.2 for "no change"
  (18% better); 12.0 min after one week of adaptation (21%).
- Café-sized queues (0–15 min): 6.3 min error after one week vs 8.7 for "no change".
- Calibration: 80% bands contain the truth 80.3% of the time.
- Untouched 2021 holdout: 7.98 vs 8.04 min, roughly a tie. Be upfront about it.

## Judge questions to rehearse
- Why Disney data for a food queue? → unit-free features + unseen-venue test.
- What if someone walks back and forth past the sensor? → we measure the queue, check against
  camera and hand count, and report our own measured error.
- How do you know it isn't overfitting? → same gain on validation, test and unseen rides; untouched holdout.
- What does Gemini add? → it decides which tools to use and explains the decision; numbers come from Python.

## Closing line
"The sensor cost £10, the code is free, the maths is the same one trading firms use for order queues. Fork it and point it at whatever queue bugs you."
