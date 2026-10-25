"""Camera head count and people tracker (owner: H2).

The camera is the main sensor: the beams sit too close together to count people. Every frame:
detect people, follow each one with an ID (ingest/tracker.py) and append a row to
data/live/camera.csv every few seconds (docs/data_contract.md §3):

  wall_time,people_in_zone,in_total,out_total,distinct_total,tracked_in_zone

people_in_zone = heads counted in the zone (median over the interval, so one missed frame does
not make the line flicker); in_total / out_total = people who entered / left the zone so far
(cumulative, like the beam counters, so ingest/state.py counts the queue from them). A person
counts when the bottom-centre of their box (their feet) is inside the zone. After a restart the
counters carry on from the last row.

Frames are never saved. data/live/detections.jsonl keeps, per frame, the boxes and a 32-number
colour fingerprint per person (no pixels), so the tracker can be re-run and tuned on a recorded
rush: python ingest/camera_accuracy.py tune (docs/data_contract.md §8). --no-log turns it off.

Detector: YOLOv8n (ultralytics) if installed, otherwise OpenCV's built-in HOG person detector
(weaker with crowds and odd angles, but needs no PyTorch).

Usage:
  python ingest/camera_count.py --set-zone      # click the queue area's corners, Enter to save
  python ingest/camera_count.py --show          # count and draw boxes, zone and count
  python ingest/camera_count.py                 # count headless (logging)
  python ingest/camera_count.py --no-reid       # tracker without re-identification (to compare)
  python ingest/camera_count.py --image photo.jpg --show   # test on a still image
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest.tracker import PARAMS, Tracker, appearance, encode_hist  # noqa: E402

ZONE_JSON = ROOT / "data" / "live" / "camera_zone.json"
CAMERA_CSV = ROOT / "data" / "live" / "camera.csv"
DETECTIONS_LOG = ROOT / "data" / "live" / "detections.jsonl"
TRACKER_PARAMS_JSON = ROOT / "data" / "live" / "tracker_params.json"  # camera_accuracy.py tune --save
HEADER = "wall_time,people_in_zone,in_total,out_total,distinct_total,tracked_in_zone"
COUNTERS = ("in_total", "out_total", "distinct_total")
FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]
YOLO_CONF = 0.35
WIDTH = 640  # frames are resized to this width before detection


class Detector:
    def __init__(self, force_hog: bool = False):
        self.yolo = None
        if not force_hog:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO("yolov8n.pt")
            except ImportError:
                pass
        if self.yolo is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.name = "yolov8n" if self.yolo else "opencv-hog"

    def people(self, frame) -> list[tuple[int, int, int, int]]:
        """Person boxes as (x1, y1, x2, y2) in pixels."""
        if self.yolo is not None:
            r = self.yolo.predict(frame, classes=[0], conf=YOLO_CONF, verbose=False)[0]
            return [tuple(int(v) for v in b) for b in r.boxes.xyxy.tolist()]
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        keep = [(x, y, w, h) for (x, y, w, h), s in zip(rects, np.ravel(weights)) if s > 0.5]
        return [(x, y, x + w, y + h) for x, y, w, h in keep]


def zone_pixels(zone, shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[x * w, y * h] for x, y in zone], dtype=np.int32)


def count_in_zone(detections, zone) -> int:
    """People whose feet (bottom-centre of the box) are inside the zone polygon (pixels)."""
    poly = np.asarray(zone, dtype=np.int32).reshape(-1, 1, 2)
    return sum(cv2.pointPolygonTest(poly, ((x1 + x2) / 2, float(y2)), False) >= 0
               for x1, y1, x2, y2 in detections)


def resize(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (WIDTH, int(h * WIDTH / w))) if w > WIDTH else frame


def draw(frame, boxes, poly, n, detector_name, tracker: Tracker | None = None, offsets=None):
    cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, (0, 200, 255), 2)
    for x1, y1, x2, y2 in boxes:
        inside = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), ((x1 + x2) / 2, float(y2)), False) >= 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if inside else (128, 128, 128), 2)
    text = f"in zone: {n} ({detector_name})"
    if tracker is not None:
        for tr in tracker.tracks.values():
            if tr.t_last == tracker.t_now:  # seen in this frame
                x1, y1 = int(tr.box[0]), int(tr.box[1])
                cv2.putText(frame, f"#{tr.id}", (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 0) if tr.inside else (160, 160, 160), 2)
        c = totals(tracker, offsets or {})
        text += f"  in {c['in_total']} out {c['out_total']} queue {c['tracked_in_zone']} re-id {c['reids']}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def tracker_params(path: Path = TRACKER_PARAMS_JSON) -> dict:
    """Tracker settings saved by `camera_accuracy.py tune --save` ({} = the defaults)."""
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in saved.items() if k in PARAMS or k == "reid"}


def resume_offsets(path: Path) -> dict:
    """Counter values in the last complete row of camera.csv, so a restart carries on counting
    (cumulative counters must never go down). {} for a new file or an old two-column one."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != HEADER:
        return {}
    for line in reversed(lines[1:]):
        parts = line.strip().split(",")
        if len(parts) == len(HEADER.split(",")):
            try:
                return {k: int(parts[i]) for i, k in enumerate(HEADER.split(",")) if k in COUNTERS}
            except ValueError:
                continue
    return {}


def totals(tracker: Tracker, offsets: dict) -> dict:
    c = tracker.counts()
    for k in COUNTERS:
        c[k] += offsets.get(k, 0)
    c["tracked_in_zone"] = c["in_total"] - c["out_total"]
    return c


def csv_line(wall: datetime, people: int, c: dict) -> str:
    return (f"{wall:%Y-%m-%dT%H:%M:%S},{people},{c['in_total']},{c['out_total']},"
            f"{c['distinct_total']},{c['tracked_in_zone']}\n")


def log_frame(f, wall: datetime, boxes, hists) -> None:
    """One line of data/live/detections.jsonl: boxes + fingerprints, never pixels."""
    f.write(json.dumps({"t": wall.isoformat(timespec="milliseconds"),
                        "b": [[*map(int, b), encode_hist(h)] for b, h in zip(boxes, hists)]},
                       separators=(",", ":")) + "\n")


def set_zone(cap) -> None:
    deadline = time.monotonic() + 10  # an iPhone (Continuity Camera) takes a few seconds to send frames
    ok, frame = cap.read()
    while not ok and time.monotonic() < deadline:
        time.sleep(0.25)
        ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read from the camera (iPhone: lock it, keep it still, or plug it in)")
    frame = resize(frame)
    pts: list[list[float]] = []
    h, w = frame.shape[:2]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x / w, y / h])

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)
    print("Click the corners of the queue area. Enter = save, r = reset, Esc = cancel.")
    while True:
        ok, f = cap.read()
        f = resize(f) if ok else frame.copy()
        if pts:
            cv2.polylines(f, [zone_pixels(pts, f.shape).reshape(-1, 1, 2)], len(pts) > 2, (0, 200, 255), 2)
        cv2.imshow("zone", f)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(pts) >= 3:
            ZONE_JSON.parent.mkdir(parents=True, exist_ok=True)
            ZONE_JSON.write_text(json.dumps({"zone": pts}))
            print(f"saved {len(pts)} points -> {ZONE_JSON}")
            break
        if k == ord("r"):
            pts.clear()
        if k == 27:
            break
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--out", type=Path, default=CAMERA_CSV)
    ap.add_argument("--every", type=float, default=3.0, help="seconds per CSV row")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--set-zone", action="store_true")
    ap.add_argument("--hog", action="store_true", help="use OpenCV HOG even if ultralytics is installed")
    ap.add_argument("--image", type=Path, help="count people in one image and exit")
    ap.add_argument("--no-reid", action="store_true", help="turn re-identification off (to compare)")
    ap.add_argument("--log", type=Path, default=DETECTIONS_LOG, help="detections log for tuning")
    ap.add_argument("--no-log", action="store_true", help="do not write the detections log")
    args = ap.parse_args()

    zone = json.loads(ZONE_JSON.read_text())["zone"] if ZONE_JSON.exists() else FULL_FRAME
    det = Detector(args.hog)

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"cannot read image {args.image}")
        frame = resize(frame)
        boxes = det.people(frame)
        poly = zone_pixels(zone, frame.shape)
        n = count_in_zone(boxes, poly)
        print(f"{n} people in zone ({len(boxes)} detected, {det.name})")
        if args.show:
            draw(frame, boxes, poly, n, det.name)
            cv2.imshow("camera", frame)
            cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.cam} not found (try --cam 1)")
    if args.set_zone:
        set_zone(cap)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    offsets = resume_offsets(args.out)
    if offsets:  # whoever was in the queue at the restart is seen again as new: close them first
        offsets["out_total"] = offsets["in_total"]
    if args.out.exists() and args.out.stat().st_size and not offsets:
        # an old two-column camera.csv (or a broken one): keep it, start a fresh file
        old = args.out.with_name(f"{args.out.stem}_old_{datetime.now():%H%M%S}.csv")
        args.out.rename(old)
        print(f"{args.out.name} had the old format: moved to {old.name}")
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    tracker, poly = None, None
    params = tracker_params()
    if args.no_reid:
        params["reid"] = False
    counts, t_next = [], time.monotonic() + args.every
    print(f"counting with {det.name} + tracker {params or '(default settings)'}, zone "
          f"{'from ' + str(ZONE_JSON) if ZONE_JSON.exists() else 'full frame'} -> {args.out}"
          f"{'' if args.no_log else ', detections -> ' + str(args.log)}"
          f"{'  (resuming: ' + str(offsets) + ')' if offsets else ''}  (Ctrl+C or q to stop)")
    log = None
    if not args.no_log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open("a")
    with args.out.open("a", newline="") as f:
        if new_file:
            f.write(HEADER + "\n")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue
                frame = resize(frame)
                wall = datetime.now()
                if tracker is None:
                    poly = zone_pixels(zone, frame.shape)
                    tracker = Tracker(poly, **params)
                    if log:  # header line: the zone and frame size these boxes belong to
                        log.write(json.dumps({"start": wall.isoformat(timespec="seconds"), "detector": det.name,
                                              "frame": list(frame.shape[:2]), "zone": poly.tolist()}) + "\n")
                boxes = det.people(frame)
                hists = [appearance(frame, b) for b in boxes]
                tracker.update(list(zip(boxes, hists)), time.monotonic())
                if log:
                    log_frame(log, wall, boxes, hists)
                n = count_in_zone(boxes, poly)
                counts.append(n)
                if args.show:
                    draw(frame, boxes, poly, n, det.name, tracker, offsets)
                    cv2.imshow("camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if time.monotonic() >= t_next:
                    f.write(csv_line(wall, int(statistics.median(counts)), totals(tracker, offsets)))
                    f.flush()
                    if log:
                        log.flush()
                    counts, t_next = [], time.monotonic() + args.every
        except KeyboardInterrupt:
            pass
    if log:
        log.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
