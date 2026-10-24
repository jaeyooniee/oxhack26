"""Serial -> CSV logger (owner: H1).

Reads one JSON line per second from the Arduino, adds the laptop time and appends a row to
data/live/run.csv (docs/data_contract.md §1-2). Also writes the LED command back to the board,
because only one program can hold the serial port.

Usage: python ingest/logger.py --port COM3 --out data/live/run.csv
       python ingest/logger.py            # lists the serial ports
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = "wall_time,ms,dist_cm,in_total,out_total,btn_total"
FIELDS = ("ms", "dist_cm", "in", "out", "btn")
LED_EVERY_S = 5
LEDS = {"go_now": "G", "wait": "A"}
PREDICTION_MAX_AGE_S = 60  # the agent rewrites prediction.json every --watch interval


def parse_line(line: str) -> dict | None:
    """Return the decoded JSON dict, or None for a broken line."""
    try:
        d = json.loads(line.strip())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict) or any(k not in d for k in FIELDS):
        return None
    try:
        return {"ms": int(d["ms"]), "dist_cm": float(d["dist_cm"]),
                "in": int(d["in"]), "out": int(d["out"]), "btn": int(d["btn"])}
    except (TypeError, ValueError):
        return None


def led_command(prediction: dict | None, now: datetime | None = None) -> str:
    """Map prediction['recommendation'] to 'G' (go_now), 'A' (wait) or 'O' (unknown/missing).

    'O' as well when the agent's heartbeat (checked_at) is older than PREDICTION_MAX_AGE_S, so the
    LEDs never show an old recommendation after the agent stops."""
    age = prediction_age_s(prediction, now)
    if age is None or age > PREDICTION_MAX_AGE_S:
        return "O"
    return LEDS.get(prediction.get("recommendation"), "O")


def prediction_age_s(prediction: dict | None, now: datetime | None = None) -> float | None:
    """Seconds since the agent last confirmed this prediction (checked_at), None if unknown."""
    if not prediction:
        return None
    try:
        checked = datetime.fromisoformat(prediction.get("checked_at") or prediction["generated_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return ((now or datetime.now()) - checked).total_seconds()


def csv_row(d: dict, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y-%m-%dT%H:%M:%S},{d['ms']},{d['dist_cm']},{d['in']},{d['out']},{d['btn']}"


def read_prediction(path: Path) -> dict | None:
    """prediction.json, or None if missing, half-written or locked by another program (Windows)."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    import serial
    from serial.tools import list_ports

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="e.g. COM3 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "live" / "run.csv")
    ap.add_argument("--prediction", type=Path, default=ROOT / "data" / "live" / "prediction.json")
    args = ap.parse_args()
    if not args.port:
        print("Pass --port. Serial ports found:")
        for p in list_ports.comports():
            print(f"  {p.device:<14} {p.description}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    bad = rows = 0
    last_led_t, last_led = 0.0, None
    with args.out.open("a", newline="") as f:
        if new_file:
            f.write(HEADER + "\n")
        print(f"logging {args.port} -> {args.out}  (Ctrl+C to stop)")
        try:
            while True:  # reconnect if the board is unplugged or resets; state.py stitches the counters
                try:
                    with serial.Serial(args.port, args.baud, timeout=2) as ser:
                        while True:
                            raw = ser.readline()
                            d = parse_line(raw.decode("utf-8", errors="replace")) if raw else None
                            if d:
                                f.write(csv_row(d) + "\n")
                                f.flush()  # a crash loses at most one row
                                rows += 1
                                if rows % 60 == 0:
                                    print(f"{rows} rows  in={d['in']} out={d['out']} btn={d['btn']}  bad={bad}")
                            elif raw:
                                bad += 1
                            if time.monotonic() - last_led_t > LED_EVERY_S:
                                cmd = led_command(read_prediction(args.prediction))
                                ser.write(cmd.encode())
                                if cmd != last_led:
                                    print(f"LED -> {cmd}")
                                last_led_t, last_led = time.monotonic(), cmd
                except (serial.SerialException, OSError) as e:
                    print(f"serial error, reconnecting in 2 s: {e}")
                    time.sleep(2)
        except KeyboardInterrupt:
            print(f"stopped: {rows} rows, {bad} broken lines")


if __name__ == "__main__":
    main()
