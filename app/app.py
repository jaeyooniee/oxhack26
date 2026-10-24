"""Streamlit dashboard (owner: S2).   Run: streamlit run app/app.py

Panels
1. Big recommendation: "Go now" / "Go at 18:40 (save ~9 min)" + Gemini's explanation
2. Live queue length and wait estimate (from run.csv / queue_state.json), auto-refresh
3. Forecast band: P10-P90 shaded, P50 line, for the next 15/30 min
4. Accuracy: sensor vs camera vs hand count; forecast vs actual for each recorded rush
5. Footer: QR code to the GitHub repo
Replay mode: --replay data/replay/dinner_run.csv plays a recorded rush if the sensor fails.
Use data/sample/ until real data exists.
"""
