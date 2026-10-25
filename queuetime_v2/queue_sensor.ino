/*
  OxHack queue sensor  —  Arduino Uno + HC-SR04 ultrasonic + push button

  What it does
  - The ultrasonic beam across the queue entrance counts ARRIVALS  (lambda)
  - A button, pressed each time someone gets their food, counts SERVICES (mu)
  - Queue length n = arrivals - served
  - Rates are estimated with a discounted Gamma-Poisson (Bayesian) update:
    they start from a prior on day one and the error bars tighten as data
    arrives, then drift back toward the prior when things go quiet.
  - Expected wait for someone joining now = n / mu  (n services must finish
    first), P(wait > 10 min) from the Erlang distribution, and a simple
    15-minute-ahead forecast from the queue's drift (lambda - mu).
  - Every event plus a summary every 10 s is streamed over USB serial as CSV,
    so a laptop script can forward it to the forecasting server.

  Wiring
    HC-SR04 VCC   -> 5V
    HC-SR04 TRIG  -> D9
    HC-SR04 ECHO  -> D10
    HC-SR04 GND   -> GND
    Button leg    -> D2
    Opposite leg  -> GND   (internal pull-up, so no resistor needed)
    Built-in LED on D13 lights while someone is in the beam.

  Serial output (9600 baud)
    EV,<ms>,ARR,<total arrivals>
    EV,<ms>,SRV,<total served>
    SUM,<ms>,arrivals,served,queue,lam_per_min,lam_sd,mu_per_min,mu_sd,
        wait_now_min,p_wait_gt_10min,wait_in_15min
*/

#include <math.h>

const uint8_t TRIG_PIN   = 9;
const uint8_t ECHO_PIN   = 10;
const uint8_t BUTTON_PIN = 2;
const uint8_t LED_PIN    = 13;

// ---- detection tuning -------------------------------------------------
const unsigned int  ENTER_CM         = 80;     // closer than this = person in beam
const unsigned int  EXIT_CM          = 100;    // must move beyond this to re-arm
const uint8_t       CONFIRM_READS    = 2;      // consecutive pings to accept a change
const unsigned long PING_INTERVAL_MS = 60;     // HC-SR04 needs ~60 ms between pings
const unsigned long ECHO_TIMEOUT_US  = 25000;  // ~4.3 m maximum range
const unsigned int  NO_ECHO_CM       = 999;    // no echo = nothing in range
const unsigned long DEBOUNCE_MS      = 30;

// ---- streaming and model ----------------------------------------------
const unsigned long SUMMARY_MS  = 10000;  // one summary line every 10 s
const float DISCOUNT    = 0.97;           // per window, roughly 5.5 min of memory
const float LAM_A0      = 2.0;            // arrival prior: Gamma(2, 2 min) ~ 1 /min
const float LAM_B0      = 2.0;
const float MU_A0       = 3.0;            // service prior: Gamma(3, 2 min) ~ 1.5 /min
const float MU_B0       = 2.0;
const float HORIZON_MIN = 15.0;           // forecast horizon
const float RISK_MIN    = 10.0;           // report P(wait > 10 min)
#define DEBUG_DISTANCE 0                  // set to 1 to print raw distance readings

// ---- state --------------------------------------------------------------
unsigned long arrivals = 0, served = 0;
unsigned long winArrivals = 0, winServed = 0;

bool    inBeam = false;
uint8_t streak = 0;

bool          btnStable = HIGH;
bool          btnLastRead = HIGH;
unsigned long btnLastChange = 0;

float lamA = LAM_A0, lamB = LAM_B0;
float muA  = MU_A0,  muB  = MU_B0;

unsigned long lastPing = 0;
unsigned long lastSummary = 0;
uint8_t debugCounter = 0;

// ---- helpers --------------------------------------------------------------
unsigned int readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long us = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (us == 0) return NO_ECHO_CM;
  return (unsigned int)(us / 58UL);
}

void emitEvent(const char *type, unsigned long total) {
  Serial.print(F("EV,"));
  Serial.print(millis());
  Serial.print(',');
  Serial.print(type);
  Serial.print(',');
  Serial.println(total);
}

// One count per person: they must enter the beam, then leave it, before the
// next person can be counted. Hysteresis + confirmation stops jitter.
void updateBeam(unsigned int d) {
  bool wantChange = inBeam ? (d > EXIT_CM) : (d < ENTER_CM);
  if (!wantChange) { streak = 0; return; }
  if (++streak < CONFIRM_READS) return;
  streak = 0;
  inBeam = !inBeam;
  digitalWrite(LED_PIN, inBeam ? HIGH : LOW);
  if (inBeam) {
    arrivals++;
    winArrivals++;
    emitEvent("ARR", arrivals);
  }
}

void updateButton(unsigned long now) {
  bool r = digitalRead(BUTTON_PIN);
  if (r != btnLastRead) { btnLastRead = r; btnLastChange = now; }
  if (now - btnLastChange >= DEBOUNCE_MS && r != btnStable) {
    btnStable = r;
    if (btnStable == LOW) {          // pressed (pull-up, so LOW = pressed)
      served++;
      winServed++;
      emitEvent("SRV", served);
    }
  }
}

// P(wait > T) when n services at rate mu must finish first:
// sum of n exponentials = Erlang(n, mu), so P = P(Poisson(mu*T) < n).
float probWaitExceeds(long n, float mu, float T) {
  if (n <= 0) return 0.0;
  float x = mu * T;
  float term = exp(-x);
  float sum = term;
  for (long k = 1; k < n && k < 200; k++) {
    term *= x / k;
    sum += term;
  }
  return sum > 1.0 ? 1.0 : sum;
}

void printField(float v, uint8_t digits) {
  Serial.print(',');
  Serial.print(v, digits);
}

void emitSummary() {
  const float dt = SUMMARY_MS / 60000.0;   // window length in minutes

  // Discounted Gamma-Poisson update, decaying toward the prior when idle.
  lamA = DISCOUNT * lamA + (1 - DISCOUNT) * LAM_A0 + winArrivals;
  lamB = DISCOUNT * lamB + (1 - DISCOUNT) * LAM_B0 + dt;
  muA  = DISCOUNT * muA  + (1 - DISCOUNT) * MU_A0  + winServed;
  muB  = DISCOUNT * muB  + (1 - DISCOUNT) * MU_B0  + dt;
  winArrivals = 0;
  winServed = 0;

  float lam   = lamA / lamB;
  float lamSd = sqrt(lamA) / lamB;
  float mu    = muA / muB;
  float muSd  = sqrt(muA) / muB;

  long n = (long)arrivals - (long)served;
  if (n < 0) n = 0;

  float waitNow = n / mu;
  float pLong   = probWaitExceeds(n, mu, RISK_MIN);
  float nFuture = n + (lam - mu) * HORIZON_MIN;
  if (nFuture < 0) nFuture = 0;
  float wait15  = nFuture / mu;

  Serial.print(F("SUM,"));
  Serial.print(millis());
  Serial.print(',');
  Serial.print(arrivals);
  Serial.print(',');
  Serial.print(served);
  Serial.print(',');
  Serial.print(n);
  printField(lam, 2);
  printField(lamSd, 2);
  printField(mu, 2);
  printField(muSd, 2);
  printField(waitNow, 1);
  printField(pLong, 2);
  printField(wait15, 1);
  Serial.println();
}

// ---- main -------------------------------------------------------------------
void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);
  Serial.println(F("# OxHack queue sensor v1"));
  Serial.println(F("# EV,ms,ARR|SRV,total"));
  Serial.println(F("# SUM,ms,arrivals,served,queue,lam/min,lam_sd,mu/min,mu_sd,wait_now_min,p_wait>10min,wait_in_15min"));
  lastSummary = millis();
}

void loop() {
  unsigned long now = millis();

  updateButton(now);

  if (now - lastPing >= PING_INTERVAL_MS) {
    lastPing = now;
    unsigned int d = readDistanceCm();
    updateBeam(d);
#if DEBUG_DISTANCE
    if (++debugCounter >= 8) {
      debugCounter = 0;
      Serial.print(F("# dist_cm="));
      Serial.println(d);
    }
#endif
  }

  if (now - lastSummary >= SUMMARY_MS) {
    lastSummary += SUMMARY_MS;
    emitSummary();
  }
}
