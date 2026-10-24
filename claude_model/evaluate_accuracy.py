"""Accuracy report for the generic model on the 2019 test year.

Uses the cached data and saved global models from model_generic.py (run that first).
Writes results_generic/accuracy_h{h}.json plus per-ride / per-hour / per-wait-size CSVs.

Run:  python evaluate_accuracy.py --horizon 60
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from model_generic import FEATURES, OUT, QUANTILES, TEST_YEAR, to_minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    h = ap.parse_args().horizon

    df = pd.read_pickle(OUT / f"cache_df_h{h}.pkl")
    test = df[df["year"] == TEST_YEAR].copy()
    models = {q: lgb.Booster(model_file=str(OUT / f"global_h{h}_q{int(q*100)}.txt")) for q in QUANTILES}
    z = np.column_stack([models[q].predict(test[FEATURES]) for q in QUANTILES])
    z.sort(axis=1)
    pred = to_minutes(test, z)
    y = test["y"].to_numpy()
    err = pred[:, 1] - y
    naive_err = test["posted"].to_numpy() - y

    report = {
        "horizon_min": h,
        "n_test_rows": int(len(y)),
        "mean_actual_wait_min": round(float(y.mean()), 2),
        "MAE_min": round(float(np.abs(err).mean()), 2),
        "MAE_naive_min": round(float(np.abs(naive_err).mean()), 2),
        "skill_vs_naive (1 - MAE/MAE_naive)": round(float(1 - np.abs(err).mean() / np.abs(naive_err).mean()), 3),
        "RMSE_min": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias_min (+ = over-predicts)": round(float(err.mean()), 2),
        "relative_MAE (MAE / mean wait)": round(float(np.abs(err).mean() / y.mean()), 3),
        "median_abs_error_min": round(float(np.median(np.abs(err))), 2),
    }
    for k in (5, 10, 15):
        report[f"within_pm{k}min_model"] = round(float((np.abs(err) <= k).mean()), 3)
        report[f"within_pm{k}min_naive"] = round(float((np.abs(naive_err) <= k).mean()), 3)
    report["share_below_P10 (target 0.10)"] = round(float((y < pred[:, 0]).mean()), 3)
    report["share_below_P50 (target 0.50)"] = round(float((y < pred[:, 1]).mean()), 3)
    report["share_below_P90 (target 0.90)"] = round(float((y < pred[:, 2]).mean()), 3)
    report["coverage_P10_P90 (target 0.80)"] = round(float(((y >= pred[:, 0]) & (y <= pred[:, 2])).mean()), 3)

    test["abs_err"] = np.abs(err)
    test["abs_err_naive"] = np.abs(naive_err)
    test["within10"] = np.abs(err) <= 10
    agg = dict(n=("abs_err", "size"), MAE=("abs_err", "mean"), MAE_naive=("abs_err_naive", "mean"),
               within_pm10=("within10", "mean"))
    by = {
        "ride": test.groupby("ride").agg(**agg),
        "hour": test.groupby(test["hour"].astype(int)).agg(**agg),
        "wait_size": test.groupby(pd.cut(test["y"], [-1, 15, 30, 60, 120, 400],
                                         labels=["0-15", "15-30", "30-60", "60-120", "120+"]),
                                  observed=True).agg(**agg),
    }
    for name, t in by.items():
        t.round(3).to_csv(OUT / f"accuracy_by_{name}_h{h}.csv")
    (OUT / f"accuracy_h{h}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for name, t in by.items():
        print(f"\n--- by {name} ---\n{t.round(2).to_string()}")


if __name__ == "__main__":
    main()
