// IndexedDB cache for downloaded .onnx weights + small aux files.
//
// Keyed by (backend, file) → Uint8Array. Stored once, then reads are
// instantaneous on reload. The "fetch" workflow is:
//
//   getOrFetch(backend, file, urlOrLoader)
//     → check IDB hit; on miss, call urlOrLoader() to get bytes, write to IDB,
//       return bytes.
//
// We deliberately keep this tiny — no versioning, no eviction, no size cap.
// .onnx files are 1-20 MB; even five of them comfortably fit in default IDB
// quota on every modern browser. If a user wants a fresh copy they can clear
// site data.

const DB_NAME = 'vad-web-cache-v1';
const STORE = 'files';

let dbPromise = null;
function openDB() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
    return dbPromise;
}

function key(backend, file) { return `${backend}/${file}`; }

async function idbGet(k) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly');
        const req = tx.objectStore(STORE).get(k);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => reject(req.error);
    });
}

async function idbPut(k, value) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).put(value, k);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

// getOrFetch returns a Uint8Array. urlOrLoader is either a string (URL to fetch)
// or an async function returning Uint8Array. Hits go through IDB, misses
// fetch + persist.
export async function getOrFetch(backend, file, urlOrLoader, onProgress) {
    const k = key(backend, file);
    const cached = await idbGet(k);
    if (cached) {
        if (onProgress) onProgress({ stage: 'cache', file });
        return cached;
    }
    if (onProgress) onProgress({ stage: 'fetching', file });
    let bytes;
    if (typeof urlOrLoader === 'function') {
        bytes = await urlOrLoader();
    } else {
        const r = await fetch(urlOrLoader);
        if (!r.ok) throw new Error(`fetch ${urlOrLoader} → HTTP ${r.status}`);
        const ab = await r.arrayBuffer();
        bytes = new Uint8Array(ab);
    }
    try {
        await idbPut(k, bytes);
    } catch (err) {
        // Quota exceeded or private-mode: not fatal, just slow on next reload.
        console.warn('vad-web-cache: IDB put failed for', k, err);
    }
    if (onProgress) onProgress({ stage: 'cached', file, bytes: bytes.length });
    return bytes;
}

// clearCache wipes the store (debug helper; exposed as window.__vadClearCache).
export async function clearCache() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}
