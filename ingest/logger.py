"""Serial -> CSV logger (owner: H1).

Reads one JSON line per second from the Arduino, adds the laptop time and appends a row to
data/live/run.csv (docs/data_contract.md §1-2). Also writes the LED command back to the board,
because only one program can hold the serial port.

Usage: python ingest/logger.py --port COM3 --out data/live/run.csv

TODO
- [ ] open the port (pyserial, 115200 baud), skip lines that are not valid JSON
- [ ] write the header once; flush after every row so a crash loses nothing
- [ ] every ~5 s read data/live/prediction.json and send G / A / O to the board
- [ ] on start, print the port list if --port is missing
"""


def parse_line(line: str) -> dict | None:
    """Return the decoded JSON dict, or None for a broken line."""
    raise NotImplementedError


def led_command(prediction: dict | None) -> str:
    """Map prediction['recommendation'] to 'G' (go_now), 'A' (wait) or 'O' (unknown/missing)."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
