// Queue sensor firmware (owner: H1). Output: docs/data_contract.md §1.
//
// Once per second prints one JSON line over USB serial (115200 baud):
//   {"ms":123456,"dist_cm":182.4,"in":37,"out":31,"btn":12}
// Counters are cumulative, so a dropped line loses nothing.
// Listens for one-character LED commands from ingest/logger.py: G = green, A = amber, O = off.
// If no command arrives for LED_TIMEOUT_MS the LEDs turn off (laptop or logger stopped).
//
// Wiring (also in hardware/README.md):
//   HC-SR04 TRIG -> D9, ECHO -> D10 (3.3 V boards: ECHO through a 1k/2k divider)
//   Entry beam receiver -> D2, exit beam receiver -> D3 (INPUT_PULLUP)
//   Hand-count button -> D4 to GND (INPUT_PULLUP)
//   Green LED -> D6, amber LED -> D7 (each through a 220 ohm resistor to GND)

const uint8_t PIN_TRIG = 9;
const uint8_t PIN_ECHO = 10;
const uint8_t PIN_BEAM_IN = 2;
const uint8_t PIN_BEAM_OUT = 3;
const uint8_t PIN_BUTTON = 4;
const uint8_t PIN_GREEN = 6;
const uint8_t PIN_AMBER = 7;

// Level the beam receiver outputs while someone blocks it. Adafruit IR break-beams read LOW
// when broken; flip this if yours reads HIGH (check with the serial monitor before mounting).
const int BEAM_BLOCKED_LEVEL = LOW;

const unsigned long REPORT_MS = 1000;      // one JSON line per second
const unsigned long ULTRA_MS = 100;        // ultrasonic ping interval
const unsigned long ECHO_TIMEOUT_US = 25000;  // ~4 m; no echo -> -1
const unsigned long REFRACTORY_MS = 300;   // ignore re-triggers right after a crossing
const unsigned long DEBOUNCE_MS = 30;
const unsigned long LED_TIMEOUT_MS = 30000;

struct Beam {
  uint8_t pin;
  bool blocked;
  unsigned long lastCount;
  unsigned long count;
};

Beam beamIn = {PIN_BEAM_IN, false, 0, 0};
Beam beamOut = {PIN_BEAM_OUT, false, 0, 0};

unsigned long btnCount = 0;
int btnStable = HIGH, btnLastRead = HIGH;
unsigned long btnChangedAt = 0;

const int N_DIST = 5;
float dists[N_DIST];
int distIdx = 0;

unsigned long lastReport = 0, lastPing = 0, lastLedCmd = 0;

// Count one crossing on the edge into "blocked", then ignore it for REFRACTORY_MS.
void updateBeam(Beam &b, unsigned long now) {
  bool blocked = digitalRead(b.pin) == BEAM_BLOCKED_LEVEL;
  if (blocked && !b.blocked && now - b.lastCount > REFRACTORY_MS) {
    b.count++;
    b.lastCount = now;
  }
  b.blocked = blocked;
}

void updateButton(unsigned long now) {
  int r = digitalRead(PIN_BUTTON);
  if (r != btnLastRead) {
    btnChangedAt = now;
    btnLastRead = r;
  }
  if (now - btnChangedAt > DEBOUNCE_MS && r != btnStable) {
    btnStable = r;
    if (btnStable == LOW) btnCount++;  // count presses, not releases
  }
}

float pingCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, ECHO_TIMEOUT_US);
  return us == 0 ? -1.0 : us / 58.0;
}

// Median of the last N_DIST pings, ignoring -1 (no echo). -1 if none echoed.
float medianDist() {
  float v[N_DIST];
  int n = 0;
  for (int i = 0; i < N_DIST; i++)
    if (dists[i] > 0) v[n++] = dists[i];
  if (n == 0) return -1.0;
  for (int i = 1; i < n; i++)
    for (int j = i; j > 0 && v[j - 1] > v[j]; j--) {
      float t = v[j]; v[j] = v[j - 1]; v[j - 1] = t;
    }
  return v[n / 2];
}

void setLeds(char cmd) {
  digitalWrite(PIN_GREEN, cmd == 'G' ? HIGH : LOW);
  digitalWrite(PIN_AMBER, cmd == 'A' ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_BEAM_IN, INPUT_PULLUP);
  pinMode(PIN_BEAM_OUT, INPUT_PULLUP);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_AMBER, OUTPUT);
  for (int i = 0; i < N_DIST; i++) dists[i] = -1.0;
  // Boot blink: green, amber, off. If you don't see this, check the LEDs before anything else.
  setLeds('G'); delay(300);
  setLeds('A'); delay(300);
  setLeds('O');
}

void loop() {
  unsigned long now = millis();
  updateBeam(beamIn, now);
  updateBeam(beamOut, now);
  updateButton(now);

  if (now - lastPing >= ULTRA_MS) {
    lastPing = now;
    dists[distIdx] = pingCm();
    distIdx = (distIdx + 1) % N_DIST;
  }

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'G' || c == 'A' || c == 'O') {
      setLeds(c);
      lastLedCmd = now;
    }
  }
  if (lastLedCmd != 0 && now - lastLedCmd > LED_TIMEOUT_MS) {
    setLeds('O');
    lastLedCmd = 0;
  }

  if (now - lastReport >= REPORT_MS) {
    lastReport = now;
    Serial.print("{\"ms\":");
    Serial.print(now);
    Serial.print(",\"dist_cm\":");
    Serial.print(medianDist(), 1);
    Serial.print(",\"in\":");
    Serial.print(beamIn.count);
    Serial.print(",\"out\":");
    Serial.print(beamOut.count);
    Serial.print(",\"btn\":");
    Serial.print(btnCount);
    Serial.println("}");
  }
}
