# Model A: posted wait-time forecast (simple version)

Forecasts a Disney World ride's posted wait `h` minutes ahead, as P10/P50/P90 quantiles (LightGBM).
Two feature sets are trained and compared on identical rows: `simple` and `calendar` (simple + date features known in advance and the park's hours that day).

- Data: TouringPlans via github.com/LucyMcGowan/touringplans (`data-raw/*.csv` copied into `data/`, git-ignored). `-999` = ride closed, dropped.
- Grid: 15-minute slots per ride; lags never cross into another day.
- X (simple): posted now, posted 15/30/60 min ago, 30-min change, time of day, weekday, minutes since first reading of the day, ride.
- X (calendar): simple + day of year, month, holiday proximity/rank/flag, season label, % schools in session, park opening hours, extra-magic-hour flags, minutes since opening, minutes to closing. Realised temperature and capacity lost are excluded (only known after the day).
- Park day = timestamp − 4 h (metadata DATE is the park day; parks can close after midnight). Rows without metadata (after Aug 2021) are dropped.
- Y: posted wait at t + h.
- Split: train 2015–2017, early-stopping validation 2018, test 2019, stress test 2021 (2020 excluded).

Run: `pip install pandas lightgbm matplotlib` then `python model_a.py --horizon 60`
Outputs in `results/`: metrics JSON, feature importance, example-day plot, saved models.

## Results (MAE in minutes, P50)

| horizon | split | persistence | simple | calendar |
|---|---|---:|---:|---:|
| 60 min | test 2019 | 15.24 | 12.56 | 12.41 |
| 60 min | stress 2021 (Jan–Aug) | 8.04 | 9.67 | 11.16 |
| 120 min | test 2019 | 19.59 | 14.00 | 13.70 |
| 120 min | stress 2021 (Jan–Aug) | 10.04 | 11.49 | 14.45 |

`results/v1_simple_only/` holds the first run (simple features only, calendar-day grouping).

## Generic (venue-agnostic) model — `model_generic.py`, `evaluate_accuracy.py`

Unit-free features only (trend ratios, level vs today so far, time of day, weekday, minutes since opening / to closing, holiday). Target = log change in wait over the horizon. 60-min horizon, test year 2019.

| | MAE (min) | P10–P90 coverage |
|---|---:|---:|
| Persistence | 15.24 | – |
| Generic model, all rides | 13.47 | 0.807 |
| Unseen ride (leave-one-ride-out, weighted mean) | 13.89 | 0.795 |
| Unseen ride + 7-day adaptation | 13.23 | 0.804 |

Unseen-ride model beats persistence on 12 of 14 rides; fails on Flight of Passage and Slinky Dog (newest, longest-wait rides) and on waits over 120 min.
Calibration: 9.7% of outcomes below P10, 52.5% below P50, 90.5% below P90. Within ±10 min: 54.5% (persistence 52.3%). Median absolute error 8.8 min.
Details: `results_generic/leave_one_ride_out_h60.csv`, `results_generic/accuracy_*_h60.*`.

## Generic model v2 — `model_generic_v2.py`

Adds two past-only, scale-free features: `rel_week` (wait now vs the venue's median over its previous 7 days) and `mins_since_change` (how long the wait has been flat). Same rows for v1 and v2 (rows without 7 days of history dropped). 60-min horizon.

| MAE (min) | persistence | v1 | v2 |
|---|---:|---:|---:|
| Validation 2018 | 15.51 | 13.76 | 12.37 |
| Test 2019 | 15.24 | 13.48 | 12.06 |
| Untouched holdout 2021 Jan–Aug | 8.04 | 10.63 | 7.98 |
| Unseen ride, zero-shot (2019) | 15.15 | 13.88 | 12.41 |
| Unseen ride, 7-day adapted (2019) | 15.15 | 13.22 | 12.04 |

v2 beats v1 on all 14 unseen rides; beats persistence on 13/14 zero-shot (all 14 after 7-day adaptation). P10–P90 coverage 0.80–0.85 throughout.
