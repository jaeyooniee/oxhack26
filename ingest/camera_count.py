"""Camera head count (owner: H2).

Counts people standing inside a fixed zone of the webcam image and appends
`wall_time,people_in_zone` to data/live/camera.csv every few seconds (docs/data_contract.md §3).
A person counts when the bottom-centre of their box (their feet) is inside the zone. Each row is
the median count over the interval, so one missed frame does not make the line flicker.
Frames are never saved; counts only.

Detector: YOLOv8n (ultralytics) if installed, otherwise OpenCV's built-in HOG person detector
(weaker with crowds and odd angles, but needs no PyTorch).

Usage:
  python ingest/camera_count.py --set-zone      # click the queue area's corners, Enter to save
  python ingest/camera_count.py --show          # count and draw boxes, zone and count
  python ingest/camera_count.py                 # count headless (logging)
  python ingest/camera_count.py --image photo.jpg --show   # test on a still image
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZONE_JSON = ROOT / "data" / "live" / "camera_zone.json"
CAMERA_CSV = ROOT / "data" / "live" / "camera.csv"
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


def draw(frame, boxes, poly, n, detector_name):
    cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, (0, 200, 255), 2)
    for x1, y1, x2, y2 in boxes:
        inside = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), ((x1 + x2) / 2, float(y2)), False) >= 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if inside else (128, 128, 128), 2)
    cv2.putText(frame, f"in zone: {n} ({detector_name})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 255), 2)


def set_zone(cap) -> None:
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read from the camera")
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
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    counts, t_next = [], time.monotonic() + args.every
    print(f"counting with {det.name}, zone {'from ' + str(ZONE_JSON) if ZONE_JSON.exists() else 'full frame'}"
          f" -> {args.out}  (Ctrl+C or q to stop)")
    with args.out.open("a", newline="") as f:
        if new_file:
            f.write("wall_time,people_in_zone\n")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue
                frame = resize(frame)
                boxes = det.people(frame)
                poly = zone_pixels(zone, frame.shape)
                n = count_in_zone(boxes, poly)
                counts.append(n)
                if args.show:
                    draw(frame, boxes, poly, n, det.name)
                    cv2.imshow("camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if time.monotonic() >= t_next:
                    f.write(f"{datetime.now():%Y-%m-%dT%H:%M:%S},{int(statistics.median(counts))}\n")
                    f.flush()
                    counts, t_next = [], time.monotonic() + args.every
        except KeyboardInterrupt:
            pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
