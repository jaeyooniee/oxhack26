"""Camera head count (owner: H2).

Counts people inside a fixed zone of the webcam image with YOLOv8n (ultralytics) and appends
`wall_time,people_in_zone` to data/live/camera.csv every few seconds (docs/data_contract.md §3).
Frames are never saved; counts only.

Usage: python ingest/camera_count.py --show      # --show draws the zone so you can aim the camera

TODO
- [ ] load yolov8n, keep only class "person"
- [ ] zone = polygon set once with --show
- [ ] write one row every 2-5 s
"""


def count_in_zone(detections, zone) -> int:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
