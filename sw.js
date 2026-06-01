const CACHE_VERSION = 'v3';
const CACHE_NAME = `arcana-${CACHE_VERSION}`;
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/og.jpg',
  '/og-ru.jpg',
  '/favicon.ico',
  '/icon-192.png',
  '/icon-512.png',
  '/manifest.json',
  '/offline.html',
  '/sw.js',
];

const VIDEO_ASSETS = [
  '/ARCANA-CRT.mp4',
  '/ARCANA-AD.mp4',
];

// Install: precache static + lazy cache videos
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      await cache.addAll(STATIC_ASSETS);
      // Lazy-add video assets if under 500MB
      const estimate = await navigator.storage?.estimate?.() || {};
      if ((estimate.usage || 0) < 500 * 1024 * 1024) {
        for (const url of VIDEO_ASSETS) {
          try { await cache.add(url); } catch {}
        }
      }
      try { await cache.add('/offline.html'); } catch {}
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches, claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((n) => !n.startsWith('arcana-v'))
          .map((n) => caches.delete(n))
      )
    ).then(() => self.clients.claim())
  );
});

// Stale-while-revalidate for static, Cache-First for videos
async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const fetchPromise = fetch(request).then((response) => {
    if (response && response.status === 200 && response.type === 'basic') {
      cache.put(request, response.clone());
    }
    return response;
  }).catch(() => cached);
  return cached || fetchPromise;
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response && response.status === 200) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503 });
  }
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  // Static assets: stale-while-revalidate
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // Videos: cache-first
  if (VIDEO_ASSETS.includes(url.pathname) || url.pathname.endsWith('.mp4')) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Navigation: network-first with offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.status < 400) return response;
          throw new Error('Navigation failed');
        })
        .catch(() => caches.match('/offline.html').then((r) => r || new Response('Offline', { status: 503 })))
    );
    return;
  }

  // Default: cache-first with background refresh
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) {
        fetch(request)
          .then((response) => {
            if (response && response.status === 200 && response.type === 'basic') {
              caches.open(CACHE_NAME).then((cache) => cache.put(request, response.clone()));
            }
          })
          .catch(() => {});
        return cached;
      }
      return fetch(request)
        .then((response) => {
          if (!response || response.status !== 200 || response.type !== 'basic') return response;
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => new Response('Offline', { status: 503 }));
    })
  );
});

// Background sync
try {
  self.addEventListener('sync', (event) => {
    if (event.tag === 'arcana-sync') {
      event.waitUntil(Promise.resolve());
    }
  });
} catch {}

// Periodic background sync for content updates
try {
  self.addEventListener('periodicsync', (event) => {
    if (event.tag === 'arcana-content-update') {
      event.waitUntil(
        caches.open(CACHE_NAME).then(async (cache) => {
          for (const url of STATIC_ASSETS) {
            try {
              const response = await fetch(url, { cache: 'no-store' });
              if (response.ok) cache.put(url, response.clone());
            } catch {}
          }
        })
      );
    }
  });
} catch {}

// Push notification scaffold (for future use)
self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};
  event.waitUntil(
    self.registration.showNotification(data.title || 'ARCANA.VISION', {
      body: data.body || 'New pattern detected.',
      icon: '/icon-192.png',
      badge: '/icon-192.png',
      tag: data.tag || 'arcana-default',
      requireInteraction: false,
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(self.clients.openWindow('/'));
});
