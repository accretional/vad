// VAD inference engine. Manages one worker per loaded backend; lazily
// instantiates the worker + loads the model + caches in IDB on first run.
//
// Exports an `Engine` singleton with a single `run(backend, samples)` method:
//
//   const segments = await engine.run('pyannote', float32Audio);
//
// The first call for each backend pays the model-download + worker-init
// cost; subsequent calls reuse the warm worker + session. Concurrent runs
// against the same backend are serialised (the worker session isn't
// safe for parallel execution, and ORT's wasm thread pool isn't either).

import { getOrFetch } from './cache.js';

const WORKER_URL = new URL('./worker.js', import.meta.url);

// Aux files per backend. Loaded from /aux/<dir>/<file> on the demo server,
// which proxies them from the embedded weights tree.
const BACKEND_META = {
    pyannote:  { dir: 'pyannote',    aux: [] },
    fsmn:      { dir: 'fsmn-vad',    aux: ['am.mvn'] },
    firered:   { dir: 'firered-vad', aux: ['cmvn_means.f32', 'cmvn_istd.f32'] },
    silero:    { dir: 'silero',      aux: [] },
    marblenet: { dir: 'marblenet',   aux: [] },
};

class BackendHandle {
    constructor(backend, onProgress) {
        this.backend = backend;
        this.onProgress = onProgress || (() => {});
        this.worker = null;
        this.readyPromise = null;
        this.nextRunId = 1;
        this.pending = new Map();
        // Serialise runs on the same worker — ORT sessions aren't reentrant.
        this.runQueue = Promise.resolve();
    }

    async ready() {
        if (this.readyPromise) return this.readyPromise;
        this.readyPromise = this._init();
        return this.readyPromise;
    }

    async _init() {
        const meta = BACKEND_META[this.backend];
        if (!meta) throw new Error(`unknown backend ${this.backend}`);

        this.onProgress({ stage: 'fetching-model', backend: this.backend });
        const modelBytes = await getOrFetch(meta.dir, 'model.onnx', async () => {
            // Ask the server's /fetch endpoint. If it returns JSON {"url": "..."}
            // we fetch the .onnx directly from that CDN URL (smaller round
            // trip through our server); otherwise the response IS the bytes.
            const r = await fetch(`/fetch?model=${encodeURIComponent(this.backend.toUpperCase())}`);
            if (!r.ok) throw new Error(`fetch ${this.backend} model → HTTP ${r.status}`);
            const ct = r.headers.get('content-type') || '';
            if (ct.startsWith('application/json')) {
                const { url } = await r.json();
                if (!url) throw new Error(`fetch ${this.backend}: server returned empty url`);
                this.onProgress({ stage: 'downloading-from-cdn', backend: this.backend, url });
                const r2 = await fetch(url);
                if (!r2.ok) throw new Error(`download ${url} → HTTP ${r2.status}`);
                return new Uint8Array(await r2.arrayBuffer());
            }
            return new Uint8Array(await r.arrayBuffer());
        }, this.onProgress);

        // Auxiliary files — small, no URL indirection (always streamed by
        // the demo server out of the embedded weights tree).
        const aux = {};
        for (const f of meta.aux) {
            aux[f] = await getOrFetch(meta.dir, f, `/aux/${meta.dir}/${f}`, this.onProgress);
        }

        this.onProgress({ stage: 'starting-worker', backend: this.backend });
        this.worker = new Worker(WORKER_URL, { type: 'module' });
        this.worker.addEventListener('message', (ev) => this._onMessage(ev.data));
        this.worker.addEventListener('error', (ev) => {
            // Bubble fatal worker errors to all pending runs.
            for (const { reject } of this.pending.values()) {
                reject(new Error(`worker error: ${ev.message || ev.type}`));
            }
            this.pending.clear();
        });

        await this._postAndWait({
            type: 'init',
            backend: this.backend,
            modelBytes,
            aux,
        });

        this.onProgress({ stage: 'ready', backend: this.backend });
    }

    _onMessage(msg) {
        if (msg.type === 'ready') {
            const r = this.pending.get('__init__');
            if (r) { this.pending.delete('__init__'); r.resolve(msg); }
            return;
        }
        if (msg.type === 'result' || msg.type === 'error') {
            const id = msg.id != null ? msg.id : '__init__';
            const r = this.pending.get(id);
            if (r) {
                this.pending.delete(id);
                if (msg.type === 'error') r.reject(new Error(msg.error));
                else                      r.resolve(msg);
            }
        }
    }

    _postAndWait(msg, transfer = []) {
        return new Promise((resolve, reject) => {
            const id = msg.type === 'init' ? '__init__' : (msg.id != null ? msg.id : this.nextRunId++);
            if (msg.type !== 'init') msg.id = id;
            this.pending.set(id, { resolve, reject });
            this.worker.postMessage(msg, transfer);
        });
    }

    // run serialises against other in-flight runs on this handle.
    async run(samples) {
        await this.ready();
        const next = this.runQueue.then(async () => {
            // We deliberately copy `samples` rather than transferring its
            // buffer: the caller usually still wants their PCM around (it's
            // what the waveform canvas draws from). For huge audio (>30s)
            // this is a noticeable memory cost; acceptable for a demo.
            const copy = new Float32Array(samples);
            const id = this.nextRunId++;
            return this._postAndWait({ type: 'run', id, samples: copy }, [copy.buffer]);
        });
        this.runQueue = next.catch(() => {}); // don't propagate failures to the queue
        return next;
    }

    terminate() {
        if (this.worker) this.worker.terminate();
        this.worker = null;
        this.readyPromise = null;
        this.pending.clear();
    }
}

export class Engine {
    constructor() {
        this.handles = new Map();
        this.onProgress = () => {};
    }
    setProgressHandler(fn) { this.onProgress = fn || (() => {}); }
    handle(backend) {
        let h = this.handles.get(backend);
        if (!h) {
            h = new BackendHandle(backend, this.onProgress);
            this.handles.set(backend, h);
        }
        return h;
    }
    async run(backend, samples) {
        return this.handle(backend).run(samples);
    }
}

export const engine = new Engine();
