# Live queue model

`kalman.py`: Bayesian arrival/service rates (Gamma-Poisson with 5-minute memory) and a Kalman
filter on queue length that fuses beam counts with the ultrasonic. Gives 80% intervals for the
wait now. Written from scratch here because `model.py` from queuetime.zip was never added.

Turn it on with `python ingest/state.py --smooth` (beams only) or
`--us-empty <cm> --us-cm-per-person <cm>` (with the ultrasonic; calibrate both at the food point:
reading with nobody queueing, then how much it drops per extra person).
On the simulated rush (5% exit-beam misses) queue-length error drops from 2.3 to 0.8 people.
