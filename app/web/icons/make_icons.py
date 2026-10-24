"""Draws icon-192.png and icon-512.png (same design as icon.svg) with the standard library only.

Run: python app/web/icons/make_icons.py
"""
import struct
import zlib
from pathlib import Path

BG, INK, SIGNAL = (0xF2, 0xF1, 0xEC), (0x0F, 0x0F, 0x0F), (0xE5, 0x48, 0x2B)


def draw(n: int) -> bytes:
    u = n / 64  # icon.svg's 64-unit grid
    px = [[BG] * n for _ in range(n)]

    def rect(x0, y0, x1, y1, colour):
        for y in range(round(y0 * u), round(y1 * u)):
            for x in range(round(x0 * u), round(x1 * u)):
                px[y][x] = colour
    # as in icon.svg: a signal square and three ink bars of falling length (a queue
    # getting shorter). The manifest declares it "any", not maskable, so it is never cropped.
    rect(14, 14, 26, 26, SIGNAL)
    rect(14, 32, 50, 37, INK)
    rect(14, 41, 42, 46, INK)
    rect(14, 50, 32, 55, INK)
    raw = b"".join(b"\x00" + bytes(c for p in row for c in p) for row in px)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for n in (192, 512):
        (here / f"icon-{n}.png").write_bytes(draw(n))
        print(f"icon-{n}.png")
