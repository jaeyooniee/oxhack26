"""Live queue model (owner: S1): Bayesian rates + Kalman queue length, with error bars.

Rates. Arrivals in dt minutes ~ Poisson(lambda * dt), with a Gamma(a, b) belief about lambda.
Each update: a <- f*a + n, b <- f*b + dt, f = exp(-dt / TAU), so old minutes fade out and the
estimate follows the rush. Service rate mu is the same, but only learns while someone is
queueing (an idle server is not a slow server).

Queue length. Kalman filter on L:
  predict  L += arrivals - served          variance += BEAM_MISS * (arrivals + served)
  measure  L_us = (empty_cm - dist_cm) / cm_per_person   (only when the ultrasonic is calibrated)
The beams are precise per step but drift when they miss someone; the ultrasonic is noisy but
does not drift, so together they stay honest. Without a calibrated ultrasonic, a "queue" of
DRIFT_MAX or fewer with nobody moving for IDLE_RESET_MIN is treated as drift and reset to 0.

Wait. W = L / mu, with an 80% interval from sampling L ~ N(L, P) and mu ~ Gamma(a, b).

    from model.live.kalman import LiveQueueModel, run_model
    m = run_model(run_df, now)          # feeds run.csv rows up to now
    m.summary()                          # point estimates + intervals
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TAU_MIN = 5.0          # memory of the rate estimates (minutes)
PRIOR_RATE = 1.0       # people per minute before any data
PRIOR_WEIGHT_MIN = 2.0  # prior is worth this many minutes of data
BEAM_MISS = 0.05       # share of crossings a beam misses (fake queue test: measure it!)
US_SD_PEOPLE = 1.5     # ultrasonic noise, in people
MIN_RATE = 0.05        # floor on mu so W stays finite
DRIFT_MAX = 3          # a "queue" this small with nobody moving for IDLE_RESET_MIN is beam drift
IDLE_RESET_MIN = 3.0


@dataclass
class GammaRate:
    a: float = PRIOR_RATE * PRIOR_WEIGHT_MIN
    b: float = PRIOR_WEIGHT_MIN

    def update(self, n: float, dt: float, tau: float = TAU_MIN):
        f = math.exp(-dt / tau)
        # forget toward the prior, not toward nothing, so a long gap does not explode the variance
        self.a = f * self.a + (1 - f) * PRIOR_RATE * PRIOR_WEIGHT_MIN + n
        self.b = f * self.b + (1 - f) * PRIOR_WEIGHT_MIN + dt

    @property
    def mean(self) -> float:
        return self.a / self.b

    def sample(self, rng, k: int) -> np.ndarray:
        return rng.gamma(self.a, 1 / self.b, k)


@dataclass
class LiveQueueModel:
    us_empty_cm: float | None = None       # ultrasonic reading with nobody queueing
    us_cm_per_person: float | None = None  # how much closer each extra person makes it
    tau_min: float = TAU_MIN
    lam: GammaRate = field(default_factory=GammaRate)
    mu: GammaRate = field(default_factory=GammaRate)
    L: float = 0.0
    P: float = 1.0
    idle_min: float = 0.0
    last: dict | None = None

    def update(self, t: pd.Timestamp, in_total: float, out_total: float, dist_cm: float | None = None):
        if self.last is None:
            self.last = {"t": t, "in": in_total, "out": out_total}
            return
        dt = (t - self.last["t"]).total_seconds() / 60
        if dt <= 0:
            return
        arr = max(0.0, in_total - self.last["in"])
        srv = max(0.0, out_total - self.last["out"])
        busy = self.L > 0.5 or srv > 0
        self.lam.update(arr, dt, self.tau_min)
        if busy:
            self.mu.update(srv, dt, self.tau_min)

        # Kalman predict (counts are the control input) ...
        self.L = max(0.0, self.L + arr - srv)
        self.P += BEAM_MISS * (arr + srv) + 0.01 * dt
        # ... and correct with the ultrasonic, when calibrated and echoing
        if self.us_empty_cm and self.us_cm_per_person and dist_cm is not None and dist_cm > 0:
            z = max(0.0, (self.us_empty_cm - dist_cm) / self.us_cm_per_person)
            R = US_SD_PEOPLE ** 2
            k = self.P / (self.P + R)
            self.L = max(0.0, self.L + k * (z - self.L))
            self.P *= 1 - k
        # nobody arriving or leaving for a while with only a few "people" left: beam drift
        self.idle_min = 0.0 if arr or srv else self.idle_min + dt
        if self.idle_min >= IDLE_RESET_MIN and self.L <= DRIFT_MAX:
            self.L, self.P = 0.0, min(self.P, 0.25)
        self.last = {"t": t, "in": in_total, "out": out_total}

    def wait_mean(self) -> float:
        return self.L / max(self.mu.mean, MIN_RATE) if self.L > 0.5 else 0.0

    def summary(self, k: int = 4000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        L = np.maximum(0, rng.normal(self.L, math.sqrt(self.P), k))
        mu = np.maximum(self.mu.sample(rng, k), MIN_RATE)
        w = np.where(L > 0.5, L / mu, 0.0)
        lam_s = self.lam.sample(rng, k)

        def q(x):
            return [round(float(v), 2) for v in np.percentile(x, [10, 90])]
        return {
            "queue_length": round(self.L, 1),
            "queue_length_sd": round(math.sqrt(self.P), 2),
            "arrival_rate_per_min": round(self.lam.mean, 2),
            "arrival_rate_ci80": q(lam_s),
            "service_rate_per_min": round(self.mu.mean, 2),
            "service_rate_ci80": q(mu),
            "wait_now_min": round(self.wait_mean(), 1),
            "wait_now_ci80_min": [round(v, 1) for v in q(w)],
        }


def run_model(df: pd.DataFrame, now=None, served_col: str = "out_total", step_s: int = 5,
              grid: pd.DatetimeIndex | None = None, **kwargs) -> tuple[LiveQueueModel, pd.DataFrame]:
    """Feed run.csv rows (every step_s seconds) up to `now`; also return L, mu, wait at `grid` times."""
    now = pd.Timestamp(now).floor("s") if now is not None else df.index[-1]
    rows = df[df.index <= now]
    rows = rows.resample(f"{step_s}s").last().dropna(subset=["in_total"])
    m = LiveQueueModel(**kwargs)
    grid = grid if grid is not None else pd.DatetimeIndex([])
    gi, out = 0, []
    for t, r in zip(rows.index, rows.itertuples(index=False)):
        while gi < len(grid) and grid[gi] < t:
            out.append((grid[gi], m.L, m.mu.mean, m.wait_mean()))
            gi += 1
        m.update(t, r.in_total, getattr(r, served_col), getattr(r, "dist_cm", None))
    out += [(g, m.L, m.mu.mean, m.wait_mean()) for g in grid[gi:]]
    track = pd.DataFrame(out, columns=["time", "queue_length", "service_rate_per_min", "wait_min"])
    return m, track.set_index("time")
