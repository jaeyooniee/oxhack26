# Live queue model

`kalman.py`: Bayesian arrival/service rates (Gamma-Poisson with 5-minute memory) and a Kalman
filter on queue length that fuses beam counts with the ultrasonic. Gives 80% intervals for the
wait now. Written from scratch here because `model.py` from queuetime.zip was never added.

Turn it on with `python ingest/state.py --smooth`. The ultrasonic is used once calibrated
(`ingest/calibrate_ultrasonic.py`, or `--us-empty <cm> --us-cm-per-person <cm>`). It is ignored
until it has seen at least one person (a dead or misaimed sensor reads "empty" forever) and
readings more than 3 standard deviations from the estimate are rejected as glitches.
Drift resets use the same evidence as plain mode (`ingest/state.py`, `empty_evidence`).
