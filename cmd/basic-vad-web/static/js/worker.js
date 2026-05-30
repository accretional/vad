// VAD inference Web Worker. One instance per (backend, session).
//
// Protocol (postMessage):
//   main → worker  { type: 'init', backend, modelBytes, aux }
//                  → response: { type: 'ready' } | { type: 'error', error }
//   main → worker  { type: 'run', id, samples }   // samples = Float32Array, transferred
//                  → response: { type: 'result', id, segments, elapsedMs }
//                            | { type: 'error',  id, error }
//
// `aux` is a { filename: Uint8Array } map for backends that need sidecar
// files (FSMN am.mvn, FireRed cmvn_*.f32). Transferred via list of buffers.
//
// ORT is loaded from jsDelivr (ESM bundle that resolves the wasm asset URL
// against the script's own location — works in workers without further
// config). If you want to vendor ORT instead, swap ORT_URL for a path under
// /static/vendor/.

const ORT_VERSION = '1.22.0';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.bundle.min.mjs`;

let ort = null;
let session = null;
let backendImpl = null;
let aux = null;

// Lazy-load the right backend's run() function based on the selected backend.
const BACKEND_LOADERS = {
    pyannote:  () => import('./backends/pyannote.js'),
    fsmn:      () => import('./backends/fsmn.js'),
    firered:   () => import('./backends/firered.js'),
    silero:    () => import('./backends/silero.js'),
    marblenet: () => import('./backends/marblenet.js'),
};

async function loadORT() {
    if (ort) return ort;
    const mod = await import(ORT_URL);
    // The bundle exports an `env` global + various session classes. The bundle
    // looks for its .wasm asset alongside the .mjs script URL — since we loaded
    // it from jsDelivr the wasm fetch is cross-origin to OUR demo but the CDN
    // serves wide CORS, so this Just Works.
    ort = mod;
    // Disable web workers inside ORT itself (we're already in a worker;
    // nesting workers is supported by ORT 1.22+ but adds latency for the
    // small models this demo uses).
    ort.env.wasm.numThreads = 1;
    // SIMD is auto-detected; leave defaults.
    return ort;
}

self.addEventListener('message', async (ev) => {
    const msg = ev.data;
    try {
        if (msg.type === 'init') {
            await loadORT();
            const loader = BACKEND_LOADERS[msg.backend];
            if (!loader) throw new Error(`unknown backend: ${msg.backend}`);
            backendImpl = await loader();
            aux = msg.aux || {};
            // ort.InferenceSession.create accepts Uint8Array directly.
            session = await ort.InferenceSession.create(msg.modelBytes, {
                executionProviders: ['wasm'],
            });
            self.postMessage({ type: 'ready' });
            return;
        }
        if (msg.type === 'run') {
            if (!session || !backendImpl) throw new Error('worker not initialized');
            const t0 = (performance || Date).now();
            const segments = await backendImpl.run(ort, session, msg.samples, aux);
            const elapsedMs = (performance || Date).now() - t0;
            self.postMessage({ type: 'result', id: msg.id, segments, elapsedMs });
            return;
        }
        throw new Error(`unknown message type: ${msg.type}`);
    } catch (err) {
        self.postMessage({
            type: 'error',
            id: msg && msg.id,
            error: err && err.message ? err.message : String(err),
        });
    }
});
