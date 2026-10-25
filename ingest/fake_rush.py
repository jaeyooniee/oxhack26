"""Fake food rush in the run.csv format (docs/data_contract.md §2), for development and replay tests.

Arrivals spike at each food drop and decay; one server serves ~1.6 people/min while anyone is
queueing. The exit beam misses ~5% of people, the hand-count button is exact. A fake camera
counts the queue every 3 s, missing more people as it gets crowded, and its tracker counts
arrivals / served, giving ~4% of people a second ID (one extra in and out) (camera.csv, §3).

Usage: python ingest/fake_rush.py --start 2026-10-24T18:00:00 --out data/sample/fake_rush_run.csv
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DROPS_MIN = [(5, 4.0), (50, 2.5)]  # (minutes after start, peak arrivals per minute)
SERVICE_PER_MIN = 1.6
BEAM_MISS = 0.05
ID_SWITCH = 0.04  # fake tracker: share of people lost behind someone and counted twice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-24T18:00:00")
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sample" / "fake_rush_run.csv")
    ap.add_argument("--camera-out", type=Path, default=ROOT / "data" / "sample" / "fake_rush_camera.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rng_cam = np.random.default_rng(args.seed + 1000)  # its own stream: run.csv stays as it was
    cam_in = cam_out = switches = 0
    start = datetime.fromisoformat(args.start)
    arrived = served = beam_out = 0
    rows = ["wall_time,ms,dist_cm,in_total,out_total,btn_total"]
    cam = ["wall_time,people_in_zone,in_total,out_total,distinct_total,tracked_in_zone"]
    for s in range(args.minutes * 60):
        m = s / 60
        lam = 0.15 + sum(peak * np.exp(-(m - t0) / 12) for t0, peak in DROPS_MIN if m >= t0)
        arrived += rng.poisson(lam / 60)
        if arrived > served and rng.random() < SERVICE_PER_MIN / 60:
            served += 1
            beam_out += rng.random() > BEAM_MISS
        queue = arrived - served
        dist = max(30.0, 400 - 25 * queue + rng.normal(0, 5))
        t = start + timedelta(seconds=s)
        rows.append(f"{t:%Y-%m-%dT%H:%M:%S},{1000 * (s + 1)},{dist:.1f},{arrived},{beam_out},{served}")
        if s % 3 == 0:  # each person seen with a probability that falls as the queue grows
            heads = rng.binomial(queue, max(0.6, 0.95 - 0.01 * queue))
            new = arrived - (cam_in - switches)  # real arrivals since the last row
            twice = rng_cam.binomial(max(0, new), ID_SWITCH)  # lost behind someone, re-found as new
            switches += twice
            cam_in += max(0, new) + twice
            cam_out = served + switches
            cam.append(f"{t:%Y-%m-%dT%H:%M:%S},{heads},{cam_in},{cam_out},{cam_in},{cam_in - cam_out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n")
    args.camera_out.write_text("\n".join(cam) + "\n")
    print(f"{len(rows) - 1} rows, {arrived} arrivals, {served} served -> {args.out}")


if __name__ == "__main__":
    main()
