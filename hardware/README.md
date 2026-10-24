# Hardware (H1 board, H2 build + camera)

## Parts
- Arduino UNO Q (classic Uno as backup) + USB cable
- HC-SR04 ultrasonic sensor(s)
- Beam / tripwire for arrivals (queue entry) and served (exit)
- Push button for the hand count (ground truth)
- LEDs: green = go now, amber = wait
- Logitech webcam + USB extension (head count check)
- Breadboard, jumpers, cardboard mount, tape, power bank

## Wiring (fill in as built)
| Part | Pin(s) | Notes |
|---|---|---|
| Ultrasonic 1 TRIG / ECHO | | Board at 3.3 V logic? Then ECHO needs a voltage divider |
| Entry beam | | |
| Exit beam | | |
| Button | | Use INPUT_PULLUP |
| Green LED | | With resistor |
| Amber LED | | With resistor |

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
