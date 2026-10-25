"""Fake camera detections for a queue, in the data/live/detections.jsonl format (docs/data_contract.md §8).

People walk in from the left, join the back of the line, shuffle forward, are served at the
front and walk off to the right. The fake detector misses people at random and hides them
for a few seconds at a time (someone standing in front), jitters the boxes, and some people
wear the same colours, which is what makes counting distinct people hard. The true arrivals and
serves are written too, so ingest/camera_accuracy.py can be tried before there is real footage.

Usage:
  python ingest/fake_camera.py --out data/sample/fake_detections.jsonl --truth data/sample/fake_served.csv
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest.tracker import HIST_BINS, encode_hist  # noqa: E402

W, H = 640, 480
ZONE = [[100, 200], [560, 200], [560, 470], [100, 470]]
FRONT_X, SPACING, LINE_Y = 520, 42, 420
BOX_W, BOX_H = 56, 170
WALK_PX_S = 90
N_BINS = HIST_BINS[0] * HIST_BINS[1]


def simulate(minutes: float = 20, fps: float = 5, seed: int = 0, arrivals_per_min: float = 1.5,
             service_per_min: float = 1.4, p_miss: float = 0.08, occlusions_per_min: float = 4,
             same_clothes: float = 0.3, start: datetime = datetime(2026, 10, 24, 18, 0)):
    """Returns (frames, truth): frames = [(wall_time, [(box, hist), ...]), ...];
    truth = {"arrived": [times entering the zone], "served": [times leaving it]}."""
    rng = np.random.default_rng(seed)
    palette = [rng.dirichlet(np.full(N_BINS, 0.3)) for _ in range(3)]  # "everyone wears black"
    people, line, frames, truth = [], [], [], {"arrived": [], "served": []}
    hidden_until: dict[int, float] = {}
    service_left = None
    dt = 1 / fps
    for k in range(int(minutes * 60 * fps)):
        t = k * dt
        wall = start + timedelta(seconds=t)
        if rng.random() < arrivals_per_min / 60 * dt:
            look = (palette[rng.integers(len(palette))] if rng.random() < same_clothes
                    else rng.dirichlet(np.full(N_BINS, 0.3)))
            people.append({"id": len(people), "x": 0.0, "look": look, "state": "walk_in", "in_zone": False})
            line.append(people[-1])
        # front of the line is served, then walks off
        if line and line[0]["state"] == "queue" and abs(line[0]["x"] - FRONT_X) < 2:
            service_left = rng.exponential(60 / service_per_min) if service_left is None else service_left - dt
            if service_left <= 0:
                p = line.pop(0)
                p["state"] = "walk_out"
                service_left = None
        for i, p in enumerate(line):
            target = FRONT_X - SPACING * i
            p["x"] = min(target, p["x"] + WALK_PX_S * dt) if p["x"] < target else p["x"]
            if p["state"] == "walk_in" and p["x"] >= target - 1:
                p["state"] = "queue"
        dets = []
        for p in people:
            if p["state"] == "walk_out":
                p["x"] += WALK_PX_S * dt
                if p["x"] > W + BOX_W:
                    p["state"] = "gone"
            if p["state"] == "gone":
                continue
            inside = ZONE[0][0] <= p["x"] <= ZONE[1][0]
            if inside != p["in_zone"]:
                truth["arrived" if inside else "served"].append(wall)
                p["in_zone"] = inside
            if rng.random() < occlusions_per_min / 60 * dt:  # someone stands in front for 1-6 s
                hidden_until[p["id"]] = t + rng.uniform(1, 6)
            if hidden_until.get(p["id"], -1) > t or rng.random() < p_miss:
                continue
            x, y = p["x"] + rng.normal(0, 3), LINE_Y + rng.normal(0, 3)
            box = [int(x - BOX_W / 2), int(y - BOX_H), int(x + BOX_W / 2), int(y)]
            hist = np.clip(p["look"] + rng.normal(0, 0.01, N_BINS), 0, None)
            dets.append((box, (hist / hist.sum()).astype(np.float32)))
        frames.append((wall, dets))
    return frames, truth


def write(out: Path, truth_out: Path | None, **kw) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, truth = simulate(**kw)
    with out.open("w") as f:
        f.write(json.dumps({"start": kw.get("start", datetime(2026, 10, 24, 18, 0)).isoformat(timespec="seconds"),
                            "detector": "fake", "frame": [H, W], "zone": ZONE}) + "\n")
        for wall, dets in frames:
            f.write(json.dumps({"t": wall.isoformat(timespec="milliseconds"),
                                "b": [[*b, encode_hist(h)] for b, h in dets]}, separators=(",", ":")) + "\n")
    if truth_out:
        rows = ["wall_time,event"] + [f"{t:%Y-%m-%dT%H:%M:%S},served" for t in truth["served"]]
        truth_out.write_text("\n".join(rows) + "\n")
    return {**truth, "frames": len(frames)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sample" / "fake_detections.jsonl")
    ap.add_argument("--truth", type=Path, default=ROOT / "data" / "sample" / "fake_served.csv",
                    help="true serves in the hand-count format (docs/data_contract.md §6)")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    truth = write(args.out, args.truth, minutes=args.minutes, seed=args.seed)
    print(f"{truth['frames']} frames, {len(truth['arrived'])} arrived, {len(truth['served'])} served -> {args.out}")


if __name__ == "__main__":
    main()
