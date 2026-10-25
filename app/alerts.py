"""Voice alert (owner: S2). Reads the recommendation aloud when it changes, e.g.
"The food queue is short, about 3 minutes. Go now."

Voice: ElevenLabs text-to-speech over its REST API (ELEVENLABS_API_KEY in .env; no SDK needed).
Without a key, or if the request fails, the laptop's own voice is used (Windows SAPI, macOS
`say`, Linux `espeak`), so the alert still works offline. Clips are cached per sentence.

Usage:
  python app/alerts.py --watch 5          # speak whenever data/live/prediction.json changes
  python app/alerts.py --say "Testing"    # one sentence, to check the speakers
  python app/alerts.py --offline ...      # skip ElevenLabs
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

PREDICTION_JSON = ROOT / "data" / "live" / "prediction.json"
CACHE = ROOT / "data" / "live" / "tts_cache"
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TTS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
SAMPLE_RATE = 16000


def phrase(pred: dict | None) -> str | None:
    """The sentence to say for a prediction (numbers from Python), or None to stay quiet."""
    if not pred or pred.get("recommendation") not in ("go_now", "wait"):
        return None
    wait = round(float(pred.get("wait_now_min", 0)))
    if pred["recommendation"] == "go_now":
        if wait <= 1:
            return "The food queue is empty. Go now."
        return f"The food queue is about {wait} minutes and won't get much shorter. Go now."
    return (f"The food queue is about {wait} minutes. Come back at {pred['best_time']} "
            f"to save about {pred['minutes_saved']} minutes.")


def elevenlabs_wav(text: str, api_key: str) -> bytes:
    """Synthesize with ElevenLabs (raw 16 kHz PCM) and wrap it in a WAV container."""
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=pcm_{SAMPLE_RATE}",
        data=json.dumps({"text": text, "model_id": TTS_MODEL}).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        pcm = r.read()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def play_wav(path: Path) -> None:
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        subprocess.run(["aplay", "-q", str(path)], check=False)


def say_offline(text: str) -> None:
    """The operating system's built-in voice. Text goes through an env var, never the command line."""
    env = {**os.environ, "QT_SAY": text}
    if sys.platform == "win32":
        ps = ("Add-Type -AssemblyName System.Speech; "
              "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:QT_SAY)")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=False)
    elif sys.platform == "darwin":
        subprocess.run(["say", text], check=False)
    else:
        subprocess.run(["espeak", text], check=False)


def speak(text: str, offline: bool = False) -> str:
    """Say `text`; returns which voice was used."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key and not offline:
        CACHE.mkdir(parents=True, exist_ok=True)
        clip = CACHE / f"{hashlib.sha1(f'{VOICE_ID}|{TTS_MODEL}|{SAMPLE_RATE}|{text}'.encode()).hexdigest()[:16]}.wav"
        try:
            if not clip.exists():
                clip.write_bytes(elevenlabs_wav(text, key))
            play_wav(clip)
            return "elevenlabs"
        except Exception as e:  # no network, quota, bad key: fall back to the laptop voice
            print(f"elevenlabs failed, using offline voice: {e}", file=sys.stderr)
    say_offline(text)
    return "offline"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", type=Path, default=PREDICTION_JSON)
    ap.add_argument("--watch", type=float, metavar="SECONDS")
    ap.add_argument("--say", help="speak this sentence and exit")
    ap.add_argument("--offline", action="store_true", help="never call ElevenLabs")
    args = ap.parse_args()
    if args.offline:
        print("ElevenLabs disabled by --offline; using the laptop voice.")
    elif not os.environ.get("ELEVENLABS_API_KEY", "").strip():
        print(f"ElevenLabs not configured: set ELEVENLABS_API_KEY in {ROOT / '.env'} "
              "or the environment; using the laptop voice.", file=sys.stderr)
    else:
        print(f"ElevenLabs configured: model {TTS_MODEL}, voice {VOICE_ID}.")

    if args.say:
        print(f"[{speak(args.say, args.offline)}] {args.say}")
        return
    last = None
    while True:
        try:
            text = phrase(json.loads(args.prediction.read_text()))
        except (OSError, json.JSONDecodeError):  # missing, half-written or locked (Windows)
            text = None
        if text and text != last:
            print(f"[{speak(text, args.offline)}] {text}")
        last = text or last
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
