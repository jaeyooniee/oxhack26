# Hardware (H1 board, H2 build + camera)

## Parts
- Arduino UNO Q (classic Uno as backup) + USB cable
- HC-SR04 ultrasonic sensor(s)
- Beam / tripwire for arrivals (queue entry) and served (exit)
- Push button for the hand count (ground truth)
- LEDs: green = go now, amber = wait
- Logitech webcam + USB extension (head count check)
- Breadboard, jumpers, cardboard mount, tape, power bank

## Wiring (matches `firmware/queue_sensor/queue_sensor.ino`; change both if you rewire)
| Part | Pin(s) | Notes |
|---|---|---|
| Ultrasonic TRIG / ECHO | D9 / D10 | 3.3 V board (UNO Q): ECHO through a 1k/2k divider |
| Entry beam receiver | D2 | INPUT_PULLUP; `BEAM_BLOCKED_LEVEL` = LOW for Adafruit break-beams |
| Exit beam receiver | D3 | same |
| Button | D4 to GND | INPUT_PULLUP, debounced 30 ms, counts presses |
| Green LED | D6 | 220 Ω to GND |
| Amber LED | D7 | 220 Ω to GND |

Boot blink: green, amber, off. LEDs turn off if no command arrives for 30 s.
UNO Q: check that `Serial` output reaches the laptop's USB port (open the serial monitor once);
if it doesn't, flash the classic Uno backup.

## Firmware outline (`hardware/firmware/queue_sensor/`)
1. Read sensors every ~50–100 ms; fire ultrasonics one at a time (they interfere).
2. Count a crossing on the rising edge of "blocked", with a ~300 ms refractory period.
3. Debounce the button.
4. Once per second, print one JSON line exactly as in `docs/data_contract.md` §1.
5. Listen on serial for an LED command: `G` = green, `A` = amber, `O` = off.

## Checks before mounting
- Blink test, then each sensor alone, then all together.
- Fake queue test: 10 people walk through; compare beam vs button vs camera.
- Leave the Arduino IDE serial monitor **closed** while `logger.py` runs (only one program can use the port).
