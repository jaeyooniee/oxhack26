"""Generic model v2: adds two past-only, scale-free features to model_generic.py and re-runs
every check, so v1 and v2 can be compared on identical rows.

  rel_week          : log(1+wait now) - log(1+median wait at this venue over its previous 7 park days)
                      -> gives a sense of queue size relative to the venue's normal level
                         (a new café has this after one week; uses past days only, no leakage)
  mins_since_change : minutes the posted wait has stayed unchanged (lets the model learn when
                      "no change" is the best forecast)

Checks: validation 2018, test 2019, untouched holdout 2021 (Jan-Aug), leave-one-ride-out + 7-day
adaptation on 2019. Resumable: re-run the same command if it stops part-way.

Run:  python model_generic_v2.py --horizon 60
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import model_generic as mg

HERE = Path(__file__).parent
OUT = HERE / "results_generic_v2"
FEATURES_V1 = list(mg.FEATURES)
FEATURES_V2 = FEATURES_V1 + ["rel_week", "mins_since_change"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ride", "datetime"]).reset_index(drop=True)
    daily = df.groupby(["ride", "date"])["posted"].median().reset_index()
    daily["med7"] = (daily.groupby("ride")["posted"]
                          .transform(lambda s: s.shift(1).rolling(7, min_periods=3).median()))
    df = df.merge(daily[["ride", "date", "med7"]], on=["ride", "date"], how="left")
    df["rel_week"] = np.log1p(df["posted"]) - np.log1p(df["med7"])
    changed = df.groupby(["ride", "date"])["posted"].diff().fillna(1).ne(0)
    run_id = changed.cumsum()
    df["mins_since_change"] = df.groupby(run_id).cumcount() * 15
    return df


def run_variant(name, features, splits, train, valid, test_2019):
    mg.FEATURES = features
    res = {}
    paths = {q: OUT / f"{name}_global_q{int(q*100)}.txt" for q in mg.QUANTILES}
    ts_path = OUT / f"{name}_time_split.json"
    if ts_path.exists():
        return json.loads(ts_path.read_text())
    models = mg.fit(train, valid)
    for q, p in paths.items():
        models[q].booster_.save_model(str(p))
    for sname, part in splits.items():
        res[sname] = mg.score(part, mg.to_minutes(part, mg.predict_z(models, part)))
        y = part["y"].to_numpy()
        mid = mg.to_minutes(part, mg.predict_z(models, part))[:, 1]
        res[sname]["within_pm5"] = round(float((np.abs(mid - y) <= 5).mean()), 3)
        res[sname]["within_pm10"] = round(float((np.abs(mid - y) <= 10).mean()), 3)
    imp = pd.Series(models[0.5].booster_.feature_importance("gain"), index=features)
    (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(OUT / f"{name}_feature_importance.csv",
                                                                     header=["gain_share"])
    ts_path.write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    h = ap.parse_args().horizon
    OUT.mkdir(exist_ok=True)

    df = add_features(pd.read_pickle(mg.OUT / f"cache_df_h{h}.pkl"))
    df = df.dropna(subset=["med7"])  # first days of each ride have no 7-day history; same rows for v1 and v2
    train = df[df["year"].isin(mg.TRAIN_YEARS)]
    valid = df[df["year"] == mg.VALID_YEAR]
    test_2019 = df[df["year"] == mg.TEST_YEAR]
    splits = {"valid_2018": valid, "test_2019": test_2019, "holdout_2021_jan_aug": df[df["year"] == 2021]}

    results = {"horizon_min": h}
    for name, feats in (("v1", FEATURES_V1), ("v2", FEATURES_V2)):
        results[name] = run_variant(name, feats, splits, train, valid, test_2019)
        print(name, json.dumps(results[name]), flush=True)

    # leave-one-ride-out + adaptation, both variants, resumable
    partial = OUT / f"loro_partial_h{h}.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {r["ride"] for r in rows}
    for ride in sorted(df["ride"].unique()):
        if ride in done:
            continue
        test = test_2019[test_2019["ride"] == ride]
        days = np.sort(test["date"].unique())
        calib = test[test["date"].isin(days[:mg.CALIB_DAYS])]
        evalp = test[~test["date"].isin(days[:mg.CALIB_DAYS])]
        r = {"ride": ride, "n": len(evalp),
             "MAE_persistence": round(float(np.mean(np.abs(evalp["y"] - evalp["posted"]))), 2)}
        for name, feats in (("v1", FEATURES_V1), ("v2", FEATURES_V2)):
            mg.FEATURES = feats
            models = mg.fit(train[train["ride"] != ride], valid[valid["ride"] != ride])
            z0 = mg.predict_z(models, evalp)
            corr = mg.hour_correction(calib, mg.predict_z(models, calib))
            shift = evalp["hour"].astype(int).map(corr).fillna(0).to_numpy()[:, None]
            for tag, z in (("zero_shot", z0), ("adapted_7d", z0 + shift)):
                s = mg.score(evalp, mg.to_minutes(evalp, z))
                r[f"MAE_{name}_{tag}"] = s["MAE_model"]
                r[f"cov_{name}_{tag}"] = s["coverage_P10_P90"]
        rows.append(r)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(r, flush=True)

    loro = pd.DataFrame(rows)
    if len(loro) == df["ride"].nunique():
        loro.to_csv(OUT / f"leave_one_ride_out_h{h}.csv", index=False)
        w = loro["n"] / loro["n"].sum()
        results["loro_2019_weighted_mean"] = {c: round(float((loro[c] * w).sum()), 3)
                                              for c in loro.columns if c.startswith(("MAE_", "cov_"))}
        (OUT / f"metrics_h{h}.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
