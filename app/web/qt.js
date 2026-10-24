// QueueTime web + phone app: polls /api/snapshot every 2 s and fills whichever elements the page has.
// Every number shown comes from the server (the same Python as the LEDs); nothing is forecast here.
"use strict";

const POLL_MS = 2000;
const REMINDER_KEY = "qt.reminder";
const INK = "#0F0F0F", GREY = "#595959", SIGNAL = "#E5482B";
const $ = (id) => document.getElementById(id);
const isPhone = () => document.body.classList.contains("phone");
let last = null;           // last good snapshot
let failures = 0;
let audio = null;          // AudioContext, created on the "remind me" tap (browsers need a gesture)
let memoryReminder = null; // used when localStorage is unavailable

// ------------------------------------------------------------------ small helpers
function setText(id, text) { const el = $(id); if (el) el.textContent = text; }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function num(x, d = 1) { return x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d); }
function parseLocal(iso) { return iso ? new Date(iso) : null; } // "2026-10-24T18:05:00" = local time
function hhmm(d) { return d ? String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0") : "--:--"; }
function addMin(d, m) { return new Date(d.getTime() + m * 60000); }
function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
function dmy(iso) { return iso ? `${iso.slice(8, 10)}.${iso.slice(5, 7)}.${iso.slice(0, 4)} ${iso.slice(11, 19)}` : "–"; }

function store(key, value) {
  try {
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(value));
  } catch (e) { /* private mode: keep it in memory */ }
  memoryReminder = value;
}
function load(key) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : null; } catch (e) { return memoryReminder; }
}

// forecast points up to closing time only (nobody should plan around 19:32 when dinner shuts at 19:30)
function openForecasts(snap) {
  const s = snap && snap.state, fc = ((snap && snap.prediction) || {}).forecasts || [];
  if (!s || !snap.closes_at) return fc;
  const now = parseLocal(s.timestamp), close = parseLocal(snap.closes_at);
  return fc.filter((f) => addMin(now, f.horizon_min) <= close);
}
function minutesToClose(snap) {
  if (!snap || !snap.state || !snap.closes_at) return null;
  return (parseLocal(snap.closes_at) - parseLocal(snap.state.timestamp)) / 60000;
}

// ------------------------------------------------------------------ what the page says
function peakOf(snap) {
  const s = snap.state, fc = openForecasts(snap);
  const top = fc.reduce((a, f) => (f.p50 > (a ? a.p50 : -1) ? f : a), null);
  // only worth saying when the line really grows (a minute or more above the wait now)
  return top && top.p50 >= s.wait_now_min + 1 ? hhmm(addMin(parseLocal(s.timestamp), top.horizon_min)) : null;
}

function viewOf(snap) {
  const phone = isPhone();
  if (!snap) return { on: false, label: "Offline", hero: "Offline", word: true,
    lead: "Can't reach the QueueTime server. Check that app/server.py is running and that this device is on the same wifi." };
  if (snap.status === "no_data") return { on: false, label: "No data yet", hero: "No data", word: true, lead: snap.message };
  if (snap.status === "error") return { on: false, label: "Error", hero: "Error", word: true, lead: snap.message };
  const s = snap.state || {}, p = snap.prediction || {};
  const flow = s.arrivals_last_5m / 5 > s.service_rate_per_min
    ? `Arrivals (${s.arrivals_last_5m} in 5 min) are outrunning service (${num(s.service_rate_per_min)} per min).`
    : `Service (${num(s.service_rate_per_min)} per min) is keeping up with arrivals (${s.arrivals_last_5m} in 5 min).`;
  switch (p.recommendation) {
    case "wait": {
      const peak = peakOf(snap);
      return { on: true, label: phone ? "Wait, then go at" : "Recommendation: wait, then go at", hero: p.best_time, word: false,
        lead: `Saves about ${p.minutes_saved} minutes. ` + (phone ? (peak ? `The peak is expected at ${peak}.` : "")
          : flow + (peak ? ` The model expects the peak at ${peak}.` : "")) };
    }
    case "go_now":
      return { on: true, label: phone ? "Go now" : "Recommendation: go now", hero: "Now", word: true,
        lead: s.queue_length === 0 ? "Nobody is queueing." : `About ${num(s.wait_now_min)} minutes with ${s.queue_length} in the queue. Waiting would not save time.` };
    case "closed": {
      const n = s.next_rush;
      return { on: false, label: n ? `Closed. ${cap(n.name)} opens at` : "Closed", hero: n ? n.open : "Closed", word: !n,
        lead: n ? `The food point is closed. Next: ${n.name} at ${n.open}.` : "The food point is closed." };
    }
    default:
      return { on: false, label: "No recommendation", hero: "No call", word: true, lead: snap.message || p.explanation || "" };
  }
}

function modeOf(snap) {
  if (!snap) return "Offline";
  if (snap.status === "no_data") return "No data";
  if (snap.status === "stale") return "Stale data";
  if (snap.status === "error") return "Error";
  if (snap.mode === "replay") return `Replay ${snap.replay.speed}x${snap.replay.paused ? ", paused" : ""}`;
  return "Live";
}

// ------------------------------------------------------------------ chart (web): measured, median, 80% range
function chartSvg(snap) {
  const s = snap && snap.state, p = (snap && snap.prediction) || {};
  const hist = (snap && snap.history) || [];
  if (!s || !hist.length) return { svg: "", label: "No data yet" };
  const H = Math.min(60, hist.length - 1);
  const h = hist.slice(hist.length - 1 - H).map((r, i) => ({ x: i - H, y: r.wait }));
  const fc = openForecasts(snap).filter((f) => f.horizon_min <= 60);
  const toClose = minutesToClose(snap);
  const F = fc.length ? Math.max(...fc.map((f) => f.horizon_min)) : 0;
  const lo = -Math.max(H, 1), hi = Math.max(F, toClose !== null && toClose > 0 && toClose <= 60 ? toClose : 0, 1);
  const vals = h.map((q) => q.y).concat(fc.map((f) => f.p90), [s.wait_now_min || 0]);
  const top = Math.max(10, Math.ceil(Math.max(...vals) / 5) * 5);
  const X = (m) => 40 + ((m - lo) / (hi - lo)) * 540, Y = (v) => 200 - (v / top) * 180;
  const step = top > 30 ? 10 : 5;
  const now = parseLocal(s.timestamp);
  const out = [`<line x1="40" y1="200" x2="580" y2="200" stroke="${INK}" stroke-width="1.5"></line>`];
  for (let v = 0; v <= top; v += step) {
    if (v) out.push(`<line x1="40" y1="${Y(v)}" x2="580" y2="${Y(v)}" stroke="${INK}" stroke-opacity="0.15"></line>`);
    out.push(`<text x="34" y="${Y(v) + 4}" fill="${INK}" font-size="10" text-anchor="end">${v}</text>`);
  }
  const w0 = s.wait_now_min || 0;
  if (fc.length) {
    const up = [[0, w0]].concat(fc.map((f) => [f.horizon_min, f.p90]));
    const dn = [[0, w0]].concat(fc.map((f) => [f.horizon_min, f.p10]));
    const pts = (a) => a.map(([m, v]) => `${X(m)},${Y(v)}`).join(" ");
    out.push(`<polygon points="${pts(up.concat(dn.slice().reverse()))}" fill="${INK}" fill-opacity="0.08"></polygon>`);
    out.push(`<polyline points="${pts(up)}" fill="none" stroke="${INK}" stroke-width="0.75"></polyline>`);
    out.push(`<polyline points="${pts(dn)}" fill="none" stroke="${INK}" stroke-width="0.75"></polyline>`);
    out.push(`<polyline points="${pts([[0, w0]].concat(fc.map((f) => [f.horizon_min, f.p50])))}" fill="none" stroke="${INK}" stroke-width="1.5" stroke-dasharray="4 4"></polyline>`);
    const best = snap.best_slot;
    fc.forEach((f) => {
      out.push(`<text x="${X(f.horizon_min)}" y="222" fill="${INK}" font-size="10" text-anchor="middle">${hhmm(addMin(now, f.horizon_min))}</text>`);
      const nearBest = best && Math.abs(X(f.horizon_min) - X(best.minutes_ahead)) < 44;  // the square carries that label
      if (!nearBest) out.push(`<text x="${X(f.horizon_min) - 6}" y="${Y(f.p50) - 8}" fill="${INK}" font-size="11" text-anchor="end">${num(f.p50)}</text>`);
    });
  }
  if (toClose !== null && toClose > 0 && toClose <= hi) {
    out.push(`<line x1="${X(toClose)}" y1="20" x2="${X(toClose)}" y2="200" stroke="${INK}" stroke-width="1"></line>`);
    out.push(`<text x="${X(toClose) - 4}" y="30" fill="${INK}" font-size="10" text-anchor="end">Closes ${esc(snap.closes_at.slice(11, 16))}</text>`);
  }
  out.push(`<polyline points="${h.map((q) => `${X(q.x)},${Y(q.y)}`).join(" ")}" fill="none" stroke="${INK}" stroke-width="3"></polyline>`);
  out.push(`<text x="40" y="222" fill="${INK}" font-size="10">${esc(hist[hist.length - 1 - H].t)}</text>`);
  out.push(`<text x="${X(0)}" y="222" fill="${INK}" font-size="10" text-anchor="middle">${esc(s.timestamp.slice(11, 16))}</text>`);
  const best = snap.best_slot;
  if (best && best.minutes_ahead <= hi) {
    const bx = X(best.minutes_ahead), by = Y(best.expected_wait_min);
    out.push(`<rect x="${bx - 8}" y="${by - 8}" width="16" height="16" fill="${SIGNAL}"></rect>`);
    out.push(`<text x="${bx - 12}" y="${by + 22}" fill="${INK}" font-size="11" text-anchor="end">${esc(best.slot)}  ${num(best.expected_wait_min)}</text>`);
  }
  const label = `Wait ${num(w0)} minutes now` + (fc.length ? `. Median forecast ${fc.map((f) => `${num(f.p50)} at ${hhmm(addMin(now, f.horizon_min))}`).join(", ")}` : "");
  return { svg: out.join(""), label };
}

// ------------------------------------------------------------------ bars (phone): median wait per time
function renderBars(snap) {
  const el = $("bars");
  if (!el) return;
  const s = snap && snap.state, p = (snap && snap.prediction) || {};
  el.textContent = "";
  if (!s) { el.setAttribute("aria-label", "No data"); setText("bars-note", ""); return; }
  const now = parseLocal(s.timestamp);
  const rows = [{ t: "Now", v: s.wait_now_min, m: 0, best: false }]
    .concat(openForecasts(snap).map((f) => ({ t: hhmm(addMin(now, f.horizon_min)), v: f.p50, m: f.horizon_min, best: false })));
  const best = snap.best_slot;
  if (best && p.recommendation === "wait") { // the recommended time gets its own bar, in the signal colour
    const same = rows.find((r) => r.t === best.slot);
    if (same) same.best = true; else rows.push({ t: best.slot, v: best.expected_wait_min, m: best.minutes_ahead, best: true });
    rows.sort((a, b) => a.m - b.m);
  }
  const scale = Math.max(10, Math.ceil(Math.max(...rows.map((r) => r.v || 0)) / 10) * 10);
  rows.forEach((r) => {
    const a = document.createElement("span"); a.textContent = r.t;
    const b = document.createElement("span"); b.className = "bar" + (r.best ? " best" : "");
    b.style.width = `${Math.max(1, Math.min(100, ((r.v || 0) / scale) * 100))}%`;
    const c = document.createElement("span"); c.textContent = num(r.v); c.style.textAlign = "right";
    el.append(a, b, c);
  });
  el.setAttribute("aria-label", rows.map((r) => `${r.t}: ${num(r.v)} minutes`).join(", "));
  const cut = (p.forecasts || []).length > openForecasts(snap).length;
  setText("bars-note", `Bar scale: ${scale} min.` + (cut ? ` Closes ${snap.closes_at.slice(11, 16)}, nothing shown after.` : "") +
    (best ? ` Range at ${best.slot}: ${num(best.p10)}–${num(best.p90)} (calibrated on Disney rides, unchecked here).` : ""));
}

// ------------------------------------------------------------------ reminder ("remind me at 18:35")
function reminderTarget(snap) {
  const now = parseLocal(snap.state.timestamp);
  const [h, m] = snap.prediction.best_time.split(":").map(Number);
  let at = new Date(now); at.setHours(h, m, 0, 0);
  if (at < now) at = addMin(at, 24 * 60); // 00:10 asked at 23:55
  return at;
}

function setReminder() {
  if (!last || !last.prediction || last.prediction.recommendation !== "wait") return;
  try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { audio = null; }
  store(REMINDER_KEY, { at: reminderTarget(last).toISOString(), label: last.prediction.best_time, mode: last.mode });
  if (window.isSecureContext && "Notification" in window && Notification.permission === "default") {
    Notification.requestPermission().catch(() => {});
  }
  render(last);
}

function cancelReminder() { store(REMINDER_KEY, null); render(last); }

function beep() {
  if (!audio) return;
  try {
    [0, 0.35, 0.7].forEach((t) => {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = 880; o.connect(g); g.connect(audio.destination);
      g.gain.setValueAtTime(0.2, audio.currentTime + t); g.gain.setValueAtTime(0, audio.currentTime + t + 0.2);
      o.start(audio.currentTime + t); o.stop(audio.currentTime + t + 0.25);
    });
  } catch (e) { /* no sound: the full-screen reminder still shows */ }
}

function fireAlarm(r, snap) {
  const s = snap.state || {};
  setText("alarm-text", `It's ${r.label}. Wait now ${num(s.wait_now_min)} min, ${s.queue_length} in the queue.`);
  const el = $("alarm"); if (el) el.classList.add("on");
  if (navigator.vibrate) navigator.vibrate([250, 120, 250, 120, 250]);
  beep();
  if (window.isSecureContext && "Notification" in window && Notification.permission === "granted") {
    const body = `It's ${r.label}. About ${num(s.wait_now_min)} min now.`;
    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      navigator.serviceWorker.ready.then((reg) => reg.showNotification("QueueTime: go now", { body, tag: "qt" })).catch(() => {});
    } else {
      try { new Notification("QueueTime: go now", { body }); } catch (e) { /* ignore */ }
    }
  }
}

function checkReminder(snap) {
  const r = load(REMINDER_KEY);
  if (r && snap && snap.state) {
    const now = parseLocal(snap.state.timestamp), at = new Date(r.at);
    if (r.mode !== snap.mode || now - at > 12 * 3600e3) store(REMINDER_KEY, null);
    else if (now >= at) { store(REMINDER_KEY, null); fireAlarm(r, snap); }
  }
  const cur = load(REMINDER_KEY);
  const canSet = !!(snap && snap.prediction && snap.prediction.recommendation === "wait");
  const btn = $("remind-btn");
  if (btn) {
    btn.hidden = !canSet && !cur;
    btn.disabled = !canSet;
    btn.textContent = cur ? `Reminder set for ${cur.label}` : canSet ? `Remind me at ${snap.prediction.best_time}` : "Remind me";
  }
  const status = $("remind-status");
  if (status) status.hidden = !cur;
  setText("remind-text", cur ? `Keep this page open. It beeps at ${cur.label}` +
    (window.isSecureContext ? "." : " (phone notifications need https).") : "");
}

// ------------------------------------------------------------------ render
function render(snap) {
  document.body.classList.remove("loading");
  const s = snap && snap.state, p = (snap && snap.prediction) || {};
  const v = viewOf(snap);
  const marker = $("marker"); if (marker) marker.className = "marker" + (v.on ? "" : " off");
  setText("label", v.label);
  const hero = $("hero"); if (hero) { hero.textContent = v.hero; hero.classList.toggle("word", v.word); }
  setText("lead", v.lead);
  setText("working", !s ? "" : [
    `Wait = queue ÷ service rate, as the median of the last 3 minutes. A single minute is noisy (the latest reads ${s.queue_length} ÷ ${num(s.service_rate_per_min, 2)} = ${num(s.wait_instant_min)} min), so it is not shown as the wait.`,
    p.forecasts && p.forecasts.length ? "Forecast: LightGBM, trained on Disney World 2015–17." : "",
    snap.best_slot ? `P(wait > ${snap.best_slot.max_wait_min} min) at ${snap.best_slot.slot}: ${num(snap.best_slot.prob_wait_over_max, 2)}.` : "",
    p.source ? `Decided by ${p.source.replace("model:generic_v2 + ", "")}.` : "",
  ].filter(Boolean).join(" "));
  setText("mode-label", modeOf(snap));
  setText("state-time", s ? `03 State ${dmy(s.timestamp)}` : "03 State –");
  setText("rush-meta", s && s.rush ? `02 ${cap(s.rush)} ${s.venue_open}–${s.venue_close}` : "02 No rush");
  setText("phone-meta", s ? `${cap(s.rush || "No rush")} · ${s.timestamp.slice(11, 16)}` : modeOf(snap));

  setText("m-wait", s ? num(s.wait_now_min) : "–");
  setText("m-queue", s ? String(s.queue_length) : "–");
  setText("m-joined", s ? String(s.arrivals_last_5m) : "–");
  setText("m-rate", s ? num(s.service_rate_per_min) : "–");
  setText("m-note", !s ? "" : `Wait: median of the last 3 minutes.` +
    (s.wait_now_ci80_min ? ` Latest minute, 80% range: ${num(s.wait_now_ci80_min[0])}–${num(s.wait_now_ci80_min[1])} min.` : ""));

  const chart = $("chart");
  if (chart) { const c = chartSvg(snap); chart.innerHTML = c.svg; chart.setAttribute("aria-label", c.label); }
  renderBars(snap);

  const sch = $("schedule");
  if (sch) {
    sch.textContent = "";
    ((snap && snap.schedule) || []).forEach((r) => {
      const a = document.createElement("span"), b = document.createElement("span");
      a.textContent = cap(r.name) + (r.current ? " ■" : "");
      b.textContent = `${r.open}–${r.close}`;
      if (r.current) { a.className = "now"; b.className = "now"; }
      sch.append(a, b);
    });
  }

  const sen = snap && snap.sensors, hb = snap && snap.heartbeat_age_s;
  setText("sensor-line", !snap ? "" : !sen ? "No sensor readings yet." : [
    `Sensor: HC-SR04 ${sen.dist_cm === null ? "no echo" : num(sen.dist_cm, 0) + " cm"}, beams in ${sen.in} out ${sen.out}, hand count ${sen.btn}.`,
    `Last reading ${sen.last_row}.`,
    snap.mode === "live" ? (hb === null || hb === undefined ? "No agent heartbeat." : `Agent checked ${Math.round(hb)} s ago.`) : "Replay of a recorded file.",
    sen.board_resets ? `Board resets stitched: ${sen.board_resets}.` : "",
  ].filter(Boolean).join(" "));

  const rp = $("replay");
  if (rp) {
    const on = !!(snap && snap.mode === "replay" && snap.replay);
    rp.classList.toggle("on", on);
    if (on) {
      setText("replay-info", `Replay ${snap.replay.file} at ${snap.replay.speed}x, minute ${num(snap.replay.minute, 0)} of ${snap.replay.total_min}.`);
      setText("replay-toggle", snap.replay.paused ? "Play" : "Pause");
      const inp = $("replay-min"); if (inp) inp.max = snap.replay.total_min;
    }
  }
  checkReminder(snap);
}

async function poll() {
  try {
    const r = await fetch("/api/snapshot", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    last = await r.json();
    failures = 0;
    render(last);
  } catch (e) {
    failures += 1;
    if (failures >= 2) render(null); // one blip is not "offline"
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

async function replayCmd(body) {
  try {
    const r = await fetch("/api/replay", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    setText("replay-error", j.ok ? "" : j.error || "Failed");
  } catch (e) { setText("replay-error", "Server unreachable"); }
}

document.addEventListener("DOMContentLoaded", () => {
  const rb = $("remind-btn"); if (rb) rb.addEventListener("click", setReminder);
  const rc = $("remind-cancel"); if (rc) rc.addEventListener("click", cancelReminder);
  const ad = $("alarm-dismiss"); if (ad) ad.addEventListener("click", () => $("alarm").classList.remove("on"));
  const rt = $("replay-toggle");
  if (rt) rt.addEventListener("click", () => replayCmd({ action: last && last.replay && last.replay.paused ? "play" : "pause" }));
  const rj = $("replay-jump");
  if (rj) rj.addEventListener("click", () => replayCmd({ action: "jump", minute: Number($("replay-min").value || 0) }));
  if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("/sw.js").catch(() => {});
  poll();
});
