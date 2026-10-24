"""Train and export the deployable forecast models (owner: S1).

For each horizon writes model/artifacts/h{H}/ with
  q10.txt, q50.txt, q90.txt   LightGBM boosters (predict z = log1p(wait at t+H) - log1p(wait now))
  meta.json                   feature order, target definition, training split, test scores
Same protocol as claude_model/model_generic_v2.py (train 2015-17, early stop 2018, test 2019),
plus cold-start training: rel_week is hidden on 30% of training rows, so the model also works
for a venue with no 7-day history (the OxHack food point).

Run:  python model/export.py --horizons 15 30 60     (resumable: finished horizons are skipped)
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_model"))
import model_generic as mg  # noqa: E402
from model_generic_v2 import FEATURES_V2, add_features  # noqa: E402

ART = ROOT / "model" / "artifacts"
COLD_START_SHARE = 0.3


def scores(models, part):
    out = mg.score(part, mg.to_minutes(part, mg.predict_z(models, part)))
    mid = mg.to_minutes(part, mg.predict_z(models, part))[:, 1]
    out["within_pm10"] = round(float((np.abs(mid - part["y"].to_numpy()) <= 10).mean()), 3)
    return out


def export(h: int):
    d = ART / f"h{h}"
    if (d / "meta.json").exists():
        print(f"h{h}: already exported, skipping")
        return
    d.mkdir(parents=True, exist_ok=True)
    cache = mg.OUT / f"cache_df_h{h}.pkl"
    mg.OUT.mkdir(exist_ok=True)
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = mg.build(h)
        df.to_pickle(cache)
    df = add_features(df)
    rng = np.random.default_rng(0)
    df.loc[rng.random(len(df)) < COLD_START_SHARE, "rel_week"] = np.nan

    mg.FEATURES = FEATURES_V2
    train = df[df["year"].isin(mg.TRAIN_YEARS)]
    valid = df[df["year"] == mg.VALID_YEAR]
    test = df[df["year"] == mg.TEST_YEAR].copy()
    models = mg.fit(train, valid)
    for q, m in models.items():
        m.booster_.save_model(str(d / f"q{int(q * 100)}.txt"))

    test_cold = test.assign(rel_week=np.nan)
    meta = {
        "name": "queuetime-generic-v2",
        "exported": str(date.today()),
        "horizon_min": h,
        "quantiles": mg.QUANTILES,
        "features": FEATURES_V2,
        "target": "z = log1p(wait at t+h) - log1p(wait now)",
        "to_minutes": "forecast_min = expm1(log1p(wait_now_min) + z), clipped at 0; sort the 3 quantiles",
        "training": {"train_years": list(mg.TRAIN_YEARS), "early_stop_year": mg.VALID_YEAR,
                     "test_year": mg.TEST_YEAR, "cold_start_share": COLD_START_SHARE,
                     "data": "TouringPlans Disney World wait times via github.com/LucyMcGowan/touringplans"},
        "test_2019": scores(models, test),
        "test_2019_cold_start_no_7day_history": scores(models, test_cold),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"h{h}:", json.dumps({k: meta[k] for k in ("test_2019", "test_2019_cold_start_no_7day_history")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    for h in ap.parse_args().horizons:
        export(h)


if __name__ == "__main__":
    main()
