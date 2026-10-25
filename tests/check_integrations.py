"""Check configuration, or test real providers with --gemini / --elevenlabs.

Run from any directory. No keys are printed and no live prediction is overwritten.
Explicit provider flags make API calls (and can consume provider credits).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", action="store_true", help="test Gemini's queue tools and decision")
    parser.add_argument("--elevenlabs", action="store_true", help="synthesize a short clip without playing it")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    failed = False
    for provider, setting, requested in (
        ("Gemini", "GEMINI_API_KEY", args.gemini),
        ("ElevenLabs", "ELEVENLABS_API_KEY", args.elevenlabs),
    ):
        key = os.environ.get(setting, "").strip()
        print(f"{provider}: {'key configured' if key else 'missing ' + setting}")
        if not requested:
            continue
        if not key:
            failed = True
            continue
        try:
            if provider == "Gemini":
                from agent.queue_agent import gemini_decision
                from model.forecast import forecast
                state = json.loads((ROOT / "data/sample/queue_state_sample.json").read_text())
                decision = gemini_decision(state, forecast(state))
                print(f"Gemini: tool calling and response validation passed ({decision['recommendation']})")
            else:
                from app.alerts import elevenlabs_wav
                clip = elevenlabs_wav("QueueTime connection test.", key)
                print(f"ElevenLabs: speech received ({len(clip)} WAV bytes)")
        except Exception as error:
            # SDK exceptions may echo request details. Redact all configured provider keys.
            message = str(error)
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ELEVENLABS_API_KEY"):
                secret = os.environ.get(name)
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            print(f"{provider}: FAILED: {type(error).__name__}: {message}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
