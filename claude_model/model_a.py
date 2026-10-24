"""Model A: forecast the posted wait time of a Disney World ride `h` minutes ahead.

Two feature sets are trained and compared on exactly the same rows:
  simple   : current posted wait, posted wait 15/30/60 min ago, 30-min change,
             time of day, day of week, minutes since first reading of the day, ride.
  calendar : simple + date features known in advance (day of year, month, holiday proximity
             and rank, season label, % of US schools in session) + the ride's park hours that day
             (opening hours, extra-magic-hour flags, minutes since opening, minutes to closing).
Excluded on purpose (only known after the day): realised temperature, capacity lost to closures.
Y: posted wait at t + h (same ride, same park day). Park day = timestamp shifted back 4 h,
because the metadata's DATE is the park day and parks can close after midnight.

Run:  python model_a.py --horizon 60
"""
import argparse
import warnings
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*eval_set.*")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"
SKIP = {"entities", "meta-datadictionary", "touringplans_metadata"}
QUANTILES = [0.1, 0.5, 0.9]
FEATURE_SETS = {
    "simple": ["posted", "lag_15", "lag_30", "lag_60", "diff_30",
               "hour", "dow", "mins_since_first", "ride"],
}
FEATURE_SETS["calendar"] = FEATURE_SETS["simple"] + [
    "dayofyear", "month", "HOLIDAYPX", "HOLIDAYM", "HOLIDAY", "SEASON",
    "in_session", "in_session_enrol",
    "park_hours", "emh_morning", "emh_evening", "mins_since_open", "mins_to_close"]
PARK_OF = {  # ride -> metadata park prefix
    "seven_dwarfs_train": "MK", "pirates_of_caribbean": "MK", "splash_mountain": "MK",
    "soarin": "EP", "spaceship_earth": "EP",
    "alien_saucers": "HS", "rock_n_rollercoaster": "HS", "slinky_dog": "HS", "toy_story_mania": "HS",
    "dinosaur": "AK", "expedition_everest": "AK", "flight_of_passage": "AK",
    "kilimanjaro_safaris": "AK", "navi_river": "AK",
}
DAY_SHIFT = pd.Timedelta(hours=4)
TRAIN_YEARS, VALID_YEAR, TEST_YEAR, STRESS_YEAR = (2015, 2016, 2017), 2018, 2019, 2021


def load() -> pd.DataFrame:
    """All rides in one long table on a 15-minute grid: ride, datetime, date, posted."""
    frames = []
    for f in sorted(DATA.glob("*.csv")):
        if f.stem in SKIP:
            continue
        d = pd.read_csv(f, usecols=["datetime", "SPOSTMIN"], parse_dates=["datetime"])
        d = d[d["SPOSTMIN"].notna() & (d["SPOSTMIN"] >= 0)]  # -999 = ride closed
        d["ride"] = f.stem
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = raw["datetime"].dt.floor("15min")
    df = (raw.groupby(["ride", "datetime"], as_index=False)["SPOSTMIN"].mean()
             .rename(columns={"SPOSTMIN": "posted"}))
    df["date"] = (df["datetime"] - DAY_SHIFT).dt.normalize()  # park day
    return df.sort_values(["ride", "datetime"]).reset_index(drop=True)


def value_at(df: pd.DataFrame, minutes: int) -> np.ndarray:
    """Posted wait of the same ride `minutes` from each row's time (negative = past).
    NaN if there is no reading at that slot or it falls on a different day."""
    shifted = df[["ride", "datetime", "posted"]].copy()
    shifted["datetime"] = shifted["datetime"] - pd.Timedelta(minutes=minutes)
    out = df[["ride", "datetime"]].merge(shifted, on=["ride", "datetime"], how="left")["posted"].to_numpy()
    target_day = (df["datetime"] + pd.Timedelta(minutes=minutes) - DAY_SHIFT).dt.normalize().to_numpy()
    out[target_day != df["date"].to_numpy()] = np.nan
    return out


def hhmm_to_min(x: pd.Series) -> pd.Series:
    """'9:00' -> 540, '25:00' -> 1500 (after midnight on the same park day)."""
    hm = x.astype(str).str.extract(r"(\d+):(\d+)").astype(float)
    return hm[0] * 60 + hm[1]


def load_meta() -> pd.DataFrame:
    """Park-day x park table of features that are known before the day starts."""
    m = pd.read_csv(DATA / "touringplans_metadata.csv")
    m["date"] = pd.to_datetime(m["DATE"], format="%m/%d/%Y")
    base = pd.DataFrame({
        "date": m["date"],
        "HOLIDAYPX": m["HOLIDAYPX"], "HOLIDAYM": m["HOLIDAYM"], "HOLIDAY": m["HOLIDAY"],
        "SEASON": m["SEASON"].fillna("UNKNOWN"),
        "in_session": m["inSession"].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float),
        "in_session_enrol": m["inSession_Enrollment"].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float),
    })
    parts = []
    for park in ("MK", "EP", "HS", "AK"):
        p = base.copy()
        p["park"] = park
        p["open_min"] = hhmm_to_min(m[f"{park}OPEN"])
        p["close_min"] = hhmm_to_min(m[f"{park}CLOSE"])
        p["park_hours"] = m[f"{park}HOURS"]
        p["emh_morning"] = m[f"{park}EMHMORN"]
        p["emh_evening"] = m[f"{park}EMHEVE"]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def build(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    for k in (15, 30, 60):
        df[f"lag_{k}"] = value_at(df, -k)
    df["diff_30"] = df["posted"] - df["lag_30"]
    df["y"] = value_at(df, horizon)
    t = df["datetime"]
    df["hour"] = t.dt.hour + t.dt.minute / 60
    df["dow"] = t.dt.dayofweek
    first = df.groupby(["ride", "date"])["datetime"].transform("min")
    df["mins_since_first"] = (t - first).dt.total_seconds() / 60
    df["target_tod"] = ((t + pd.Timedelta(minutes=horizon)).dt.hour * 60
                        + (t + pd.Timedelta(minutes=horizon)).dt.minute)
    df["year"] = t.dt.year
    # calendar features (all known in advance)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["park"] = df["ride"].map(PARK_OF)
    df = df.merge(load_meta(), on=["date", "park"], how="left")
    tod = (t - df["date"]).dt.total_seconds() / 60  # minutes since park-day midnight (can exceed 1440)
    df["mins_since_open"] = tod - df["open_min"]
    df["mins_to_close"] = df["close_min"] - tod
    df["ride"] = df["ride"].astype("category")
    df["SEASON"] = df["SEASON"].astype("category")
    # keep only rows that have metadata, so both feature sets are compared on identical rows
    df = df.dropna(subset=["y", "HOLIDAYPX"])
    return df.reset_index(drop=True)


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(part: pd.DataFrame, preds: np.ndarray, profile: pd.Series) -> dict:
    y = part["y"].to_numpy()
    prof = part.set_index(["ride", "target_tod"]).index.map(profile).to_numpy(dtype=float)
    prof = np.where(np.isnan(prof), part["posted"].to_numpy(), prof)
    lo, mid, hi = preds[:, 0], preds[:, 1], preds[:, 2]
    return {
        "n": int(len(y)),
        "MAE_persistence (y = now)": round(float(np.mean(np.abs(y - part["posted"]))), 2),
        "MAE_ride_x_time_of_day_mean": round(float(np.mean(np.abs(y - prof))), 2),
        "MAE_model_P50": round(float(np.mean(np.abs(y - mid))), 2),
        "pinball_P10": round(pinball(y, lo, 0.1), 3),
        "pinball_P50": round(pinball(y, mid, 0.5), 3),
        "pinball_P90": round(pinball(y, hi, 0.9), 3),
        "coverage_P10_P90 (target 0.80)": round(float(np.mean((y >= lo) & (y <= hi))), 3),
        "mean_interval_width_min": round(float(np.mean(hi - lo)), 2),
    }


def plot_example(test: pd.DataFrame, preds: np.ndarray, horizon: int, path: Path):
    ride = "seven_dwarfs_train"
    t = test.assign(lo=preds[:, 0], mid=preds[:, 1], hi=preds[:, 2])
    t = t[t["ride"] == ride]
    day = t.groupby("date")["y"].mean().idxmax()  # busiest test day for this ride
    d = t[t["date"] == day].copy()
    d["target_time"] = d["datetime"] + pd.Timedelta(minutes=horizon)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(d["target_time"], d["lo"], d["hi"], alpha=0.25, label="P10–P90 forecast")
    ax.plot(d["target_time"], d["mid"], label=f"P50 forecast (made {horizon} min earlier)")
    ax.plot(d["target_time"], d["y"], "k.", ms=4, label="Observed posted wait")
    ax.set_title(f"{ride} — park day {day.date()} (test year) — {path.stem}")
    ax.set_ylabel("Posted wait (min)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60, help="minutes ahead (multiple of 15)")
    args = ap.parse_args()
    h = args.horizon
    OUT.mkdir(exist_ok=True)

    print("Loading…")
    df = build(load(), h)
    train = df[df["year"].isin(TRAIN_YEARS)]
    valid = df[df["year"] == VALID_YEAR]
    splits = {"test_2019": df[df["year"] == TEST_YEAR], "stress_2021": df[df["year"] == STRESS_YEAR]}
    print(f"rows: train {len(train):,}  valid {len(valid):,}  "
          + "  ".join(f"{k} {len(v):,}" for k, v in splits.items()))

    profile = (train.groupby(["ride", "target_tod"], observed=True)["y"].mean())

    results = {}
    for fs_name, features in FEATURE_SETS.items():
        print(f"[{fs_name}] {len(features)} features")
        models = {}
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, learning_rate=0.1, n_estimators=600,
                                  num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.9, verbose=-1)
            m.fit(train[features], train["y"], eval_set=[(valid[features], valid["y"])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            models[q] = m
            m.booster_.save_model(str(OUT / f"model_{fs_name}_h{h}_q{int(q*100)}.txt"))
            print(f"  q={q}: {m.best_iteration_} trees")

        results[fs_name] = {}
        for name, part in splits.items():
            preds = np.column_stack([models[q].predict(part[features]) for q in QUANTILES])
            preds.sort(axis=1)  # avoid quantile crossing
            results[fs_name][name] = evaluate(part, preds, profile)
            if name == "test_2019":
                plot_example(part, preds, h, OUT / f"example_day_{fs_name}_h{h}.png")

        imp = pd.Series(models[0.5].booster_.feature_importance("gain"), index=features)
        (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(
            OUT / f"feature_importance_{fs_name}_h{h}.csv", header=["gain_share"])

    (OUT / f"metrics_h{h}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
