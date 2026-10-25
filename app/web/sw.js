// QueueTime service worker: the app shell opens without the network; live data never comes from cache.
const CACHE = "queuetime-v4";  // bump on every release so phones drop the old look
const SHELL = ["/phone", "/", "/site.css", "/qt.js", "/terms", "/privacy", "/manifest.webmanifest", "/icons/icon.svg", "/icons/icon-192.png",
  "/fonts/IBMPlexSans-Regular.woff2", "/fonts/IBMPlexSans-SemiBold.woff2", "/fonts/IBMPlexSans-Bold.woff2",
  "/fonts/IBMPlexMono-Regular.woff2", "/fonts/IBMPlexMono-Medium.woff2"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin || url.pathname.startsWith("/api/")) return;
  // network first, so a new version shows at once; the cached shell only when offline
  e.respondWith(fetch(e.request)
    .then((r) => { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); return r; })
    .catch(() => caches.match(e.request)));
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({ type: "window" }).then((cs) => (cs.length ? cs[0].focus() : self.clients.openWindow("/phone"))));
});
