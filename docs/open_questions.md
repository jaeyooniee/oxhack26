# Open questions and known gaps (resolve before the pitch)

1. **"Beat Disney's posted wait" benchmark.** The current model *forecasts* the posted wait, so the
   posted wait is its target, not a rival. To beat Disney's own estimate we would need a model
   from posted → actual wait (the sparse `SACTMIN` column, about 96k rows). Decide: build it, or
   drop that claim and keep "beats the no-change baseline".
2. **Horizons.** Trained and validated at 60 min. The plan asks for 15 and 30 min:
   run `python claude_model/model_generic_v2.py --horizon 15` and `--horizon 30`.
3. **"Minutes to next scheduled event" feature.** Not in the model yet. The Disney metadata has
   parade and fireworks times (`MKPRDDT1`, `MKFIRET1`, ...), which map to food drops. Worth testing.
4. **Cold start at the hackathon.** `rel_week` needs 7 days of venue history, which the food point
   will not have. Options: leave it missing (LightGBM handles NaN) or use today's mean. Test which
   is better on Disney before the demo.
5. **Sensor wait vs posted wait.** The food queue's wait is estimated with Little's law
   (L / μ), while Disney's is an operator estimate. State this difference openly.
6. **2021 holdout.** The model only ties "no change" on post-COVID 2021 data (7.98 vs 8.04 min) and
   over-predicts by ~1.8 min. Say "retrain when conditions change".
