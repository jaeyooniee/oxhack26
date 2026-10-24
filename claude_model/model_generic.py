"""Venue-agnostic queue forecaster, tested on venues it has never seen.

Goal: show that a model trained on some queues transfers to a new queue (e.g. a café),
using only information any venue has.

Features (no ride ID, no Disney-specific variables, no absolute wait level):
  r_lag15/30/60 : log(1+wait 15/30/60 min ago) - log(1+wait now)    (recent trend, unit-free)
  rel_today     : log(1+wait now) - log(1+mean wait so far today)   (busy vs today so far, unit-free)
  hour, dow, mins_since_open, mins_to_close, frac_of_day           (venue opening hours)
  HOLIDAY, HOLIDAYPX                                               (public holiday flag / proximity)
Target: z = log(1+wait at t+h) - log(1+wait now)  -> converted back to minutes for evaluation.

Experiments (horizon h, default 60 min):
  1. time split   : train 2015-17, early-stop 2018, test 2019, all rides.
  2. leave-one-ride-out (LORO): for each ride, train on the other 13, test on that ride in 2019.
  3. adaptation   : give the unseen ride its first 7 park days of 2019; learn a shrunk
                    hour-of-day correction of the residual; test on the rest of 2019.

Run:  python model_generic.py --horizon 60
"""
import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model_a import PARK_OF, QUANTILES, load, load_meta, pinball, value_at

warnings.filterwarnings("ignore", message=".*eval_set.*")
HERE = Path(__file__).parent
OUT = HERE / "results_generic"
FEATURES = ["r_lag15", "r_lag30", "r_lag60", "rel_today", "hour", "dow",
            "mins_since_open", "mins_to_close", "frac_of_day", "HOLIDAY", "HOLIDAYPX"]
TRAIN_YEARS, VALID_YEAR, TEST_YEAR = (2015, 2016, 2017), 2018, 2019
CALIB_DAYS = 7
SHRINK_K = 20  # pseudo-observations for the hour-of-day correction (Bayesian shrinkage)
LGB_PARAMS = dict(learning_rate=0.1, n_estimators=400, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def build(horizon: int) -> pd.DataFrame:
    df = load()
    lp = np.log1p(df["posted"])
    for k in (15, 30, 60):
        df[f"r_lag{k}"] = np.log1p(value_at(df, -k)) - lp
    df["y"] = value_at(df, horizon)
    df["z"] = np.log1p(df["y"]) - lp
    g = df.groupby(["ride", "date"])["posted"]
    mean_so_far = g.cumsum() / (g.cumcount() + 1)
    df["rel_today"] = lp - np.log1p(mean_so_far)
    t = df["datetime"]
    df["hour"] = t.dt.hour + t.dt.minute / 60
    df["dow"] = t.dt.dayofweek
    df["year"] = t.dt.year
    df["park"] = df["ride"].map(PARK_OF)
    df = df.merge(load_meta()[["date", "park", "HOLIDAY", "HOLIDAYPX", "open_min", "close_min"]],
                  on=["date", "park"], how="left")
    tod = (t - df["date"]).dt.total_seconds() / 60
    df["mins_since_open"] = tod - df["open_min"]
    df["mins_to_close"] = df["close_min"] - tod
    df["frac_of_day"] = df["mins_since_open"] / (df["close_min"] - df["open_min"])
    return df.dropna(subset=["y", "HOLIDAYPX"]).reset_index(drop=True)


def fit(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
        m.fit(train[FEATURES], train["z"], eval_set=[(valid[FEATURES], valid["z"])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        models[q] = m
    return models


def predict_z(models: dict, part: pd.DataFrame) -> np.ndarray:
    z = np.column_stack([models[q].predict(part[FEATURES]) for q in QUANTILES])
    z.sort(axis=1)
    return z


def to_minutes(part: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    return np.expm1(np.log1p(part["posted"].to_numpy())[:, None] + z).clip(min=0)


def score(part: pd.DataFrame, preds_min: np.ndarray) -> dict:
    y = part["y"].to_numpy()
    lo, mid, hi = preds_min.T
    return {
        "n": int(len(y)),
        "MAE_persistence": round(float(np.mean(np.abs(y - part["posted"]))), 2),
        "MAE_model": round(float(np.mean(np.abs(y - mid))), 2),
        "pinball_P50": round(pinball(y, mid, 0.5), 3),
        "coverage_P10_P90": round(float(np.mean((y >= lo) & (y <= hi))), 3),
        "interval_width": round(float(np.mean(hi - lo)), 2),
    }


def hour_correction(calib: pd.DataFrame, z_calib: np.ndarray) -> pd.Series:
    """Shrunk mean residual of the P50 (in z units) per hour of day: sum(e) / (n + K)."""
    e = calib["z"].to_numpy() - z_calib[:, 1]
    h = calib["hour"].astype(int).to_numpy()
    s = pd.DataFrame({"h": h, "e": e}).groupby("h")["e"].agg(["sum", "count"])
    return s["sum"] / (s["count"] + SHRINK_K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    h = ap.parse_args().horizon
    OUT.mkdir(exist_ok=True)

    cache = OUT / f"cache_df_h{h}.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        print("Loading…", flush=True)
        df = build(h)
        df.to_pickle(cache)
    train_all = df[df["year"].isin(TRAIN_YEARS)]
    valid_all = df[df["year"] == VALID_YEAR]
    test_all = df[df["year"] == TEST_YEAR]
    print(f"rows: train {len(train_all):,} valid {len(valid_all):,} test {len(test_all):,}", flush=True)

    # 1. time split, all rides (models and score cached so the run can resume)
    gpaths = {q: OUT / f"global_h{h}_q{int(q*100)}.txt" for q in QUANTILES}
    ts_path = OUT / f"time_split_h{h}.json"
    if all(p.exists() for p in gpaths.values()) and ts_path.exists():
        global_models = {q: lgb.Booster(model_file=str(p)) for q, p in gpaths.items()}
        time_split = json.loads(ts_path.read_text())
    else:
        global_models = fit(train_all, valid_all)
        for q, p in gpaths.items():
            global_models[q].booster_.save_model(str(p))
        time_split = score(test_all, to_minutes(test_all, predict_z(global_models, test_all)))
        ts_path.write_text(json.dumps(time_split))
        imp = pd.Series(global_models[0.5].booster_.feature_importance("gain"), index=FEATURES)
        (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(OUT / f"feature_importance_h{h}.csv",
                                                                         header=["gain_share"])
    results = {"horizon_min": h, "1_time_split_all_rides_2019": time_split}
    print("time split:", time_split, flush=True)

    # 2 + 3. leave-one-ride-out, then 7-day adaptation
    partial = OUT / f"loro_partial_h{h}.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {r["ride"] for r in rows}
    for ride in sorted(df["ride"].unique()):
        if ride in done:
            continue
        test = test_all[test_all["ride"] == ride]
        days = np.sort(test["date"].unique())
        calib = test[test["date"].isin(days[:CALIB_DAYS])]
        evalp = test[~test["date"].isin(days[:CALIB_DAYS])]  # same rows for every method

        loro_models = fit(train_all[train_all["ride"] != ride], valid_all[valid_all["ride"] != ride])
        z_zero = predict_z(loro_models, evalp)
        corr = hour_correction(calib, predict_z(loro_models, calib))
        shift = evalp["hour"].astype(int).map(corr).fillna(0).to_numpy()[:, None]

        r = {"ride": ride}
        for name, z in {"in_domain": predict_z(global_models, evalp),
                        "unseen_zero_shot": z_zero,
                        "unseen_adapted_7d": z_zero + shift}.items():
            s = score(evalp, to_minutes(evalp, z))
            r["n"], r["MAE_persistence"] = s["n"], s["MAE_persistence"]
            r[f"MAE_{name}"] = s["MAE_model"]
            r[f"cov_{name}"] = s["coverage_P10_P90"]
        rows.append(r)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(r, flush=True)

    loro = pd.DataFrame(rows)
    loro.to_csv(OUT / f"leave_one_ride_out_h{h}.csv", index=False)
    w = loro["n"] / loro["n"].sum()
    results["2_3_leave_one_ride_out_2019_weighted_mean"] = {
        c: round(float((loro[c] * w).sum()), 3) for c in loro.columns if c.startswith(("MAE_", "cov_"))}
    results["2_3_rides_where_unseen_model_beats_persistence"] = int(
        (loro["MAE_unseen_zero_shot"] < loro["MAE_persistence"]).sum())
    (OUT / f"metrics_h{h}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
