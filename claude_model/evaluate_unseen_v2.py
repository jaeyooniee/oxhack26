"""Accuracy of the v2 generic model on unseen data.

  A. Untouched holdout: 2021 Jan-Aug, saved v2 global model (trained 2015-17).
  B. Unseen rides (2019): for each ride, a v2 model trained without it (zero-shot), and the same
     model after a 7-day hour-of-day adaptation. Per-row predictions saved, so the run resumes.

Writes results_generic_v2/accuracy_unseen_h{h}.json and accuracy_unseen_by_wait_size_h{h}.csv.
Run:  python evaluate_unseen_v2.py --horizon 60
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import model_generic as mg
from model_generic_v2 import FEATURES_V2, OUT, add_features

BINS, LABELS = [-1, 15, 30, 60, 120, 400], ["0-15", "15-30", "30-60", "60-120", "120+"]


def report(y, posted, lo, mid, hi) -> dict:
    err, nerr = mid - y, posted - y
    r = {"n": int(len(y)), "mean_actual_wait_min": round(float(y.mean()), 2),
         "MAE_min": round(float(np.abs(err).mean()), 2),
         "MAE_naive_min": round(float(np.abs(nerr).mean()), 2),
         "skill_vs_naive": round(float(1 - np.abs(err).mean() / np.abs(nerr).mean()), 3),
         "median_abs_error_min": round(float(np.median(np.abs(err))), 2),
         "RMSE_min": round(float(np.sqrt((err ** 2).mean())), 2),
         "bias_min": round(float(err.mean()), 2),
         "relative_MAE": round(float(np.abs(err).mean() / y.mean()), 3)}
    for k in (5, 10, 15):
        r[f"within_pm{k}_model"] = round(float((np.abs(err) <= k).mean()), 3)
        r[f"within_pm{k}_naive"] = round(float((np.abs(nerr) <= k).mean()), 3)
    r["below_P10 (0.10)"] = round(float((y < lo).mean()), 3)
    r["below_P50 (0.50)"] = round(float((y < mid).mean()), 3)
    r["below_P90 (0.90)"] = round(float((y < hi).mean()), 3)
    r["coverage_P10_P90 (0.80)"] = round(float(((y >= lo) & (y <= hi)).mean()), 3)
    return r


def by_size(y, posted, mid, label):
    d = pd.DataFrame({"size": pd.cut(y, BINS, labels=LABELS), "ae": np.abs(mid - y), "ae_n": np.abs(posted - y),
                      "w10": np.abs(mid - y) <= 10})
    g = d.groupby("size", observed=True).agg(n=("ae", "size"), MAE=("ae", "mean"), MAE_naive=("ae_n", "mean"),
                                             within_pm10=("w10", "mean")).round(3)
    g.insert(0, "dataset", label)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    h = ap.parse_args().horizon
    mg.FEATURES = FEATURES_V2
    df = add_features(pd.read_pickle(mg.OUT / f"cache_df_h{h}.pkl")).dropna(subset=["med7"])
    train = df[df["year"].isin(mg.TRAIN_YEARS)]
    valid = df[df["year"] == mg.VALID_YEAR]
    test = df[df["year"] == mg.TEST_YEAR]
    out, tables = {"horizon_min": h}, []

    # A. untouched 2021 holdout
    hold = df[df["year"] == 2021]
    models = {q: lgb.Booster(model_file=str(OUT / f"v2_global_q{int(q*100)}.txt")) for q in mg.QUANTILES}
    p = mg.to_minutes(hold, mg.predict_z(models, hold))
    y, posted = hold["y"].to_numpy(), hold["posted"].to_numpy()
    out["A_holdout_2021_jan_aug"] = report(y, posted, *p.T)
    tables.append(by_size(y, posted, p[:, 1], "holdout_2021"))

    # B. unseen rides, 2019 (resumable per ride)
    pdir = OUT / f"loro_preds_h{h}"
    pdir.mkdir(exist_ok=True)
    for ride in sorted(df["ride"].unique()):
        f = pdir / f"{ride}.pkl"
        if f.exists():
            continue
        t = test[test["ride"] == ride]
        days = np.sort(t["date"].unique())
        calib = t[t["date"].isin(days[:mg.CALIB_DAYS])]
        ev = t[~t["date"].isin(days[:mg.CALIB_DAYS])]
        m = mg.fit(train[train["ride"] != ride], valid[valid["ride"] != ride])
        z0 = mg.predict_z(m, ev)
        corr = mg.hour_correction(calib, mg.predict_z(m, calib))
        shift = ev["hour"].astype(int).map(corr).fillna(0).to_numpy()[:, None]
        p0, pa = mg.to_minutes(ev, z0), mg.to_minutes(ev, z0 + shift)
        pd.DataFrame({"ride": ride, "y": ev["y"].to_numpy(), "posted": ev["posted"].to_numpy(),
                      "z_lo": p0[:, 0], "z_mid": p0[:, 1], "z_hi": p0[:, 2],
                      "a_lo": pa[:, 0], "a_mid": pa[:, 1], "a_hi": pa[:, 2]}).to_pickle(f)
        print("done", ride, flush=True)

    files = sorted(pdir.glob("*.pkl"))
    if len(files) < df["ride"].nunique():
        print(f"{len(files)}/{df['ride'].nunique()} rides done; re-run to continue")
        return
    L = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    y, posted = L["y"].to_numpy(), L["posted"].to_numpy()
    out["B_unseen_rides_2019_zero_shot"] = report(y, posted, L["z_lo"].values, L["z_mid"].values, L["z_hi"].values)
    out["B_unseen_rides_2019_adapted_7d"] = report(y, posted, L["a_lo"].values, L["a_mid"].values, L["a_hi"].values)
    tables += [by_size(y, posted, L["z_mid"].values, "unseen_zero_shot"),
               by_size(y, posted, L["a_mid"].values, "unseen_adapted_7d")]
    pd.concat(tables).to_csv(OUT / f"accuracy_unseen_by_wait_size_h{h}.csv")
    (OUT / f"accuracy_unseen_h{h}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(pd.concat(tables).to_string())


if __name__ == "__main__":
    main()
