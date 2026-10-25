"""Distinct-person tracker for the camera (owner: H2).

The beams are too close together to count people, so the camera does the counting. Counting
heads in each frame gives the queue length, but not who arrived or who was served. This tracker
gives every person an ID and follows them from frame to frame:

  in_total        people who entered the zone (feet inside for CONFIRM_HITS frames)
  out_total       people who left it (feet outside for EXIT_S, or gone from view for REID_S)
  distinct_total  different IDs ever counted in the zone
  tracked_in_zone in_total - out_total: people in the queue now, including anyone hidden for a moment

in_total / out_total are cumulative, like the beam counters, so ingest/state.py can use them
in place of the beams (state.load_counts).

Distinct people: a person hidden behind someone else for a moment comes back as a new
detection. Without care they get a new ID and are counted twice (one extra in, one extra out).
To stop that, every person carries an appearance fingerprint (a colour histogram of their
torso, never the image itself), and a new detection close to a recently lost person who looks
the same gets the old ID back (re-identification). Tune the thresholds on a recorded rush with
`python ingest/camera_accuracy.py tune` (it replays data/live/detections.jsonl).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

CONFIRM_HITS = 3     # frames with the feet in the zone before a person counts as arrived
LOST_S = 1.5         # unseen this long: the track is "lost" and can only come back via re-ID
REID_S = 15.0        # a lost person matched again within this long keeps their ID
EXIT_S = 3.0         # feet outside the zone this long = left the queue
WALK_SPEED = 0.6     # fastest walk, in box heights per second (~1 m/s)
MATCH_DIST = 0.15    # frame-to-frame: max feet movement = this + WALK_SPEED x time unseen (box heights)
NEAR_SIM = 0.45      # frame-to-frame: min appearance similarity, so a neighbour never takes your ID
REID_DIST = 3.0      # re-ID: never further than this from where they were lost (box heights)
REID_SIM = 0.6       # re-ID: min appearance similarity (Bhattacharyya coefficient, 0-1)
HIST_BINS = (8, 4)   # hue x saturation bins of the fingerprint (32 numbers)
HIST_KEEP = 0.8      # fingerprint = running mean: this much old, the rest from the new frame
BIG = 1e6

PARAMS = ("confirm_hits", "lost_s", "reid_s", "exit_s", "match_dist", "near_sim", "reid_dist", "reid_sim")


def appearance(frame: np.ndarray, box) -> np.ndarray | None:
    """Colour fingerprint of a person: hue x saturation histogram of the torso (L1-normalised),
    or None if the box is too small. Only these 32 numbers are kept, never pixels."""
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    crop = frame[max(0, y1 + int(0.2 * h)):max(0, y1 + int(0.6 * h)), max(0, x1 + int(0.2 * w)):max(0, x2 - int(0.2 * w))]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256]).ravel()
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else None


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Bhattacharyya coefficient of two fingerprints: 1 = same colours, 0 = nothing in common."""
    if a is None or b is None:
        return None
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))


def encode_hist(h: np.ndarray | None) -> str:
    """Fingerprint as 64 hex characters (for data/live/detections.jsonl)."""
    return "" if h is None else bytes(np.clip(np.round(h * 255), 0, 255).astype(np.uint8)).hex()


def decode_hist(s: str) -> np.ndarray | None:
    if not s:
        return None
    h = np.frombuffer(bytes.fromhex(s), dtype=np.uint8).astype(np.float32)
    return h / h.sum() if h.sum() > 0 else None


def feet_in_zone(box, zone: np.ndarray | None) -> bool:
    """Bottom-centre of the box inside the zone polygon (pixels); no zone = whole frame."""
    if zone is None:
        return True
    x, y = (box[0] + box[2]) / 2, float(box[3])
    inside = False
    n = len(zone)
    for i in range(n):  # ray casting, so the tracker needs no OpenCV (tests, replays)
        (xa, ya), (xb, yb) = zone[i], zone[(i + 1) % n]
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            inside = not inside
    return inside


@dataclass
class Track:
    id: int
    box: np.ndarray
    t_last: float
    hist: np.ndarray | None
    hits_in_zone: int = 0
    inside: bool = False             # counted as in the queue right now
    outside_since: float | None = None
    ever_counted: bool = False

    @property
    def feet(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]], dtype=float)

    @property
    def height(self) -> float:
        return max(1.0, float(self.box[3] - self.box[1]))


class Tracker:
    def __init__(self, zone=None, confirm_hits=CONFIRM_HITS, lost_s=LOST_S, reid_s=REID_S, exit_s=EXIT_S,
                 match_dist=MATCH_DIST, near_sim=NEAR_SIM, reid_dist=REID_DIST, reid_sim=REID_SIM, reid=True):
        """zone: polygon in pixels (None = whole frame). reid=False turns re-identification off
        (to measure how much it helps)."""
        self.zone = None if zone is None else np.asarray(zone, dtype=float).reshape(-1, 2)
        self.confirm_hits, self.lost_s, self.reid_s, self.exit_s = confirm_hits, lost_s, reid_s, exit_s
        self.match_dist, self.near_sim = match_dist, near_sim
        self.reid_dist, self.reid_sim, self.reid = reid_dist, reid_sim, reid
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.in_total = self.out_total = self.reids = 0
        self.t_now: float | None = None
        self.distinct: set[int] = set()

    # ------------------------------------------------------------------ matching
    def _dist(self, tr: Track, box) -> float:
        feet = np.array([(box[0] + box[2]) / 2, box[3]], dtype=float)
        h = (tr.height + max(1.0, float(box[3] - box[1]))) / 2
        return float(np.linalg.norm(feet - tr.feet) / h)

    def _assign(self, tracks: list[Track], dets: list, cost_fn) -> list[tuple[int, int]]:
        if not tracks or not dets:
            return []
        cost = np.array([[cost_fn(tr, d) for d in dets] for tr in tracks])
        rows, cols = linear_sum_assignment(cost)
        return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < BIG]

    def _reach(self, tr: Track) -> float:
        """How far (box heights) this person can have walked since they were last seen."""
        return self.match_dist + WALK_SPEED * (self.t_now - tr.t_last)

    def _cost(self, tr: Track, det) -> float:
        """Cost of giving this detection the track's ID; BIG = not allowed. Tracks seen in the
        last LOST_S and lost ones (re-ID) compete in one assignment, so a person coming back
        from behind someone reclaims their own ID instead of a neighbour taking it."""
        d = self._dist(tr, det[0])
        sim = similarity(tr.hist, det[1])
        if d > min(self.reid_dist, self._reach(tr)):
            return BIG
        if self.t_now - tr.t_last <= self.lost_s:
            if sim is not None and sim < self.near_sim:
                return BIG
        elif not self.reid or sim is None or sim < self.reid_sim:
            return BIG
        return d + (1 - sim if sim is not None else 0.5)

    # ------------------------------------------------------------------ update
    def update(self, dets: list, t: float) -> dict:
        """One frame. dets: [(box (x1, y1, x2, y2), fingerprint or None), ...]; t: seconds
        (any clock that only goes forward). Returns counts()."""
        self.t_now = t
        dets = [(np.asarray(b, dtype=float), h) for b, h in dets]
        horizon = self.reid_s if self.reid else self.lost_s
        cands = [tr for tr in self.tracks.values() if t - tr.t_last <= horizon]
        matched = [(cands[r], c) for r, c in self._assign(cands, dets, self._cost)]
        self.reids += sum(t - tr.t_last > self.lost_s for tr, _ in matched)
        free = [i for i in range(len(dets)) if i not in {c for _, c in matched}]
        for tr, i in matched:
            box, hist = dets[i]
            tr.box, tr.t_last = box, t
            if hist is not None:
                tr.hist = hist if tr.hist is None else HIST_KEEP * tr.hist + (1 - HIST_KEEP) * hist
            self._zone_step(tr, t)
        for i in free:
            tr = Track(self.next_id, dets[i][0], t, dets[i][1])
            self.next_id += 1
            self.tracks[tr.id] = tr
            self._zone_step(tr, t)
        for tr in list(self.tracks.values()):
            if t - tr.t_last > self.reid_s:  # gone for good
                if tr.inside:                # left the view from inside the zone (served, walked off)
                    self.out_total += 1
                del self.tracks[tr.id]
            elif not tr.ever_counted and t - tr.t_last > self.lost_s:
                del self.tracks[tr.id]       # never confirmed: a false detection, never counted
        return self.counts()

    def _zone_step(self, tr: Track, t: float) -> None:
        if feet_in_zone(tr.box, self.zone):
            tr.hits_in_zone += 1
            tr.outside_since = None
            if not tr.inside and tr.hits_in_zone >= self.confirm_hits:
                tr.inside, tr.ever_counted = True, True
                self.in_total += 1
                self.distinct.add(tr.id)
        else:
            tr.hits_in_zone = 0
            if tr.inside:
                tr.outside_since = tr.outside_since if tr.outside_since is not None else t
                if t - tr.outside_since >= self.exit_s:
                    tr.inside, tr.outside_since = False, None
                    self.out_total += 1

    def counts(self) -> dict:
        return {"in_total": self.in_total, "out_total": self.out_total, "distinct_total": len(self.distinct),
                "tracked_in_zone": self.in_total - self.out_total, "reids": self.reids}
