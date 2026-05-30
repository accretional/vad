// basic-vad-web frontend. Vanilla JS as an ES module — drives the in-browser
// VAD engine (static/js/engine.js → static/js/worker.js → onnxruntime-web).
//
// All inference runs locally in the browser. The Go server is just a
// metadata + weights proxy: it serves /describe + the model.onnx (or a CDN
// URL to it via the gRPC Fetch RPC) + aux sidecars.
//
// Flow:
//   1. On load, hit /describe — render checkboxes for all known backends.
//   2. User picks audio (sample/upload/mic). We decode to 16 kHz mono
//      Float32Array via Web Audio.
//   3. "Run detection" fans out to engine.run(backend, samples) for each
//      selected backend, in parallel. Each worker downloads its model on
//      first call (cached in IndexedDB for next time).

import { engine } from '/static/js/engine.js';

const BUNDLED_SAMPLES = [
    'bestfriends.mp3',
    'sorry-dave.mp3',
    'wake-me-up.mp3',
];

const SAMPLE_RATE = 16000;

const MODEL_COLORS = {
    PYANNOTE:  '#1976d2',
    FSMN:      '#e67e22',
    FIRERED:   '#c0392b',
    MARBLENET: '#16a085',
    SILERO:    '#8e44ad',
};

const $ = (id) => document.getElementById(id);

const els = {
    serviceMeta: $('service-meta'),
    sampleSelect: $('sample-select'),
    fileInput: $('audio-file'),
    uploadText: $('upload-text'),
    recordBtn: $('record-btn'),
    recordState: $('record-state'),
    previewBlock: $('preview-block'),
    previewAudio: $('preview-audio'),
    previewMeta: $('preview-meta'),
    modelCheckboxes: $('model-checkboxes'),
    detectBtn: $('detect-btn'),
    batchStatus: $('batch-status'),
    progressLog: $('progress-log'),
    results: $('results'),
    canvas: $('timeline-canvas'),
    legend: $('legend'),
    summary: $('result-summary'),

    liveStart: $('live-start'),
    liveStop: $('live-stop'),
    liveBackendLabel: $('live-backend-label'),
    speechIndicator: $('speech-indicator'),
    liveSegments: $('live-segments'),
};

let describeData = null;
let currentAudio = null;       // { samples, source, label }
let currentResults = null;
let recordingState = null;
let liveState = null;

(async function init() {
    populateSampleDropdown();
    wireSourceHandlers();
    wireRunButton();
    wireLivePanel();
    engine.setProgressHandler(logProgress);
    try {
        await loadDescribe();
    } catch (err) {
        els.serviceMeta.innerHTML = `<span class="warn">failed to load /describe: ${escapeHtml(err.message)}</span>`;
    }
    // Debug: expose cache clear so users can wipe weights without DevTools.
    window.__vadClearCache = async () => {
        const { clearCache } = await import('/static/js/cache.js');
        await clearCache();
        console.log('vad-web: cache cleared');
    };
})();

function populateSampleDropdown() {
    for (const s of BUNDLED_SAMPLES) {
        const opt = document.createElement('option');
        opt.value = `/static/samples/${s}`;
        opt.textContent = s;
        els.sampleSelect.appendChild(opt);
    }
}

async function loadDescribe() {
    const r = await fetch('/describe');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    describeData = await r.json();

    const reflNote = describeData.reflection_note
        ? `<span class="warn"> (${escapeHtml(describeData.reflection_note)})</span>`
        : ` <span class="ok">reflection ok</span>`;
    els.serviceMeta.innerHTML =
        `service <code>${describeData.service}</code> ` +
        `&middot; methods: <code>${describeData.methods.join(', ')}</code>${reflNote}<br>` +
        `Batch inference runs <strong>in your browser</strong> via onnxruntime-web ` +
        `(weights pulled from <code>${escapeHtml(describeData.vad_addr || 'server')}</code> on first use, ` +
        `cached in IndexedDB after that). ` +
        `Live streaming hits server-side gRPC backend ` +
        `<code>${escapeHtml(describeData.default_model || 'unset')}</code> @ ` +
        `<code>${escapeHtml(describeData.vad_addr || 'unset')}</code>.`;
    if (els.liveBackendLabel) {
        els.liveBackendLabel.textContent = describeData.default_model
            ? `(server backend: ${describeData.default_model.replace('VAD_MODEL_', '')})`
            : '';
    }

    // Build the model checkboxes. Every known model is selectable — MarbleNet
    // is no longer greyed out (browser-side inference; no server-side load
    // problems to dodge).
    els.modelCheckboxes.innerHTML = '';
    for (const m of describeData.models) {
        const lbl = document.createElement('label');
        lbl.className = 'model-cb';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = m.short_name;
        cb.checked = true;
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = colorFor(m.short_name);
        const txt = document.createElement('span');
        txt.textContent = m.short_name;
        const help = document.createElement('span');
        help.style.color = '#888';
        help.style.fontSize = '0.75rem';
        help.textContent = '  ' + (m.description || '');
        lbl.append(cb, swatch, txt, help);
        els.modelCheckboxes.appendChild(lbl);
    }
}

function colorFor(shortName) { return MODEL_COLORS[shortName] || '#777'; }
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// ---------------------------------------------------------------------------
// Audio source handling
// ---------------------------------------------------------------------------

function wireSourceHandlers() {
    els.sampleSelect.addEventListener('change', async (e) => {
        const url = e.target.value;
        if (!url) return;
        await loadFromUrl(url, e.target.options[e.target.selectedIndex].text, 'sample');
    });

    els.fileInput.addEventListener('change', async (e) => {
        const f = e.target.files[0];
        if (!f) return;
        els.uploadText.textContent = f.name;
        const buf = await f.arrayBuffer();
        await loadFromBuffer(buf, f.name, 'upload');
    });

    els.recordBtn.addEventListener('click', async () => {
        if (recordingState) {
            recordingState.recorder.stop();
            recordingState.stream.getTracks().forEach(t => t.stop());
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            const chunks = [];
            recorder.ondataavailable = (ev) => { if (ev.data.size > 0) chunks.push(ev.data); };
            recorder.onstop = async () => {
                const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
                recordingState = null;
                els.recordBtn.textContent = 'Start recording';
                els.recordBtn.classList.remove('danger');
                els.recordState.textContent = '';
                const buf = await blob.arrayBuffer();
                await loadFromBuffer(buf, `recording-${Date.now()}.webm`, 'record');
            };
            recorder.start();
            recordingState = { stream, recorder, chunks };
            els.recordBtn.textContent = 'Stop recording';
            els.recordBtn.classList.add('danger');
            els.recordState.textContent = 'recording...';
        } catch (err) {
            els.recordState.textContent = 'mic error: ' + err.message;
        }
    });
}

async function loadFromUrl(url, label, source) {
    setBatchStatus('Loading ' + label + '...', '');
    const r = await fetch(url);
    if (!r.ok) {
        setBatchStatus('fetch failed: ' + r.statusText, 'error');
        return;
    }
    const buf = await r.arrayBuffer();
    await loadFromBuffer(buf, label, source);
}

async function loadFromBuffer(arrayBuffer, label, source) {
    setBatchStatus('Decoding ' + label + '...', '');
    try {
        const samples = await decodeToF32(arrayBuffer);
        currentAudio = { samples, source, label };
        showPreview(samples, label);
        setBatchStatus(`Ready: ${label} (${(samples.length / SAMPLE_RATE).toFixed(2)} s, ${samples.length.toLocaleString()} samples)`, 'success');
        els.detectBtn.disabled = false;
    } catch (err) {
        setBatchStatus('decode failed: ' + err.message, 'error');
        els.detectBtn.disabled = true;
    }
}

async function decodeToF32(arrayBuffer) {
    const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
    try {
        const decoded = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
        const targetLen = Math.ceil(decoded.duration * SAMPLE_RATE);
        const offline = new OfflineAudioContext(1, targetLen, SAMPLE_RATE);
        const src = offline.createBufferSource();
        src.buffer = decoded;
        src.connect(offline.destination);
        src.start(0);
        const rendered = await offline.startRendering();
        return new Float32Array(rendered.getChannelData(0));
    } finally {
        try { await tmpCtx.close(); } catch (_) {}
    }
}

function showPreview(samples, label) {
    const wav = samplesToWavBlob(samples);
    const url = URL.createObjectURL(wav);
    els.previewAudio.src = url;
    els.previewMeta.textContent = `${label} - ${(samples.length / SAMPLE_RATE).toFixed(2)}s`;
    els.previewBlock.classList.remove('hidden');
}

function samplesToWavBlob(samples) {
    const dataSize = samples.length * 2;
    const buf = new ArrayBuffer(44 + dataSize);
    const v = new DataView(buf);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    v.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, SAMPLE_RATE, true);
    v.setUint32(28, SAMPLE_RATE * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    writeStr(36, 'data');
    v.setUint32(40, dataSize, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
}

// ---------------------------------------------------------------------------
// Detection (browser-side inference)
// ---------------------------------------------------------------------------

function wireRunButton() {
    els.detectBtn.addEventListener('click', runDetection);
}

async function runDetection() {
    if (!currentAudio) return;
    const selected = Array.from(els.modelCheckboxes.querySelectorAll('input[type=checkbox]:checked'))
        .map(cb => cb.value.toLowerCase());
    if (selected.length === 0) {
        setBatchStatus('select at least one backend', 'error');
        return;
    }
    els.detectBtn.disabled = true;
    setBatchStatus(`Running ${selected.length} backend(s) locally...`, '');
    clearProgressLog();

    // Fan out to all selected backends in parallel. Each backend runs in its
    // own worker (warmed on first use, kept around for subsequent runs).
    const duration = currentAudio.samples.length / SAMPLE_RATE;
    const t0 = performance.now();
    const promises = selected.map(async (backend) => {
        const shortName = backend.toUpperCase();
        const t0b = performance.now();
        try {
            const { segments, elapsedMs } = await engine.run(backend, currentAudio.samples);
            return {
                model: 'VAD_MODEL_' + shortName,
                short_name: shortName,
                segments: segments || [],
                elapsed_ms: Math.round(elapsedMs != null ? elapsedMs : (performance.now() - t0b)),
            };
        } catch (err) {
            return {
                model: 'VAD_MODEL_' + shortName,
                short_name: shortName,
                segments: [],
                elapsed_ms: Math.round(performance.now() - t0b),
                error: err.message || String(err),
            };
        }
    });
    const results = await Promise.all(promises);
    const totalMs = (performance.now() - t0).toFixed(0);
    const data = { audio_duration_seconds: duration, results };
    currentResults = data;
    renderResults(data);
    setBatchStatus(`Done in ${totalMs} ms across ${results.length} model(s).`, 'success');
    els.detectBtn.disabled = false;
}

function setBatchStatus(msg, kind) {
    els.batchStatus.textContent = msg;
    els.batchStatus.className = 'status-inline' + (kind ? ' ' + kind : '');
}

function clearProgressLog() {
    if (els.progressLog) els.progressLog.innerHTML = '';
}

function logProgress(p) {
    if (!els.progressLog) return;
    const line = document.createElement('div');
    line.className = 'progress-line';
    const text = p.backend
        ? `[${p.backend}] ${p.stage}${p.file ? ' ' + p.file : ''}${p.bytes ? ` (${(p.bytes/1e6).toFixed(2)} MB)` : ''}`
        : `${p.stage}${p.file ? ' ' + p.file : ''}`;
    line.textContent = text;
    els.progressLog.appendChild(line);
    els.progressLog.scrollTop = els.progressLog.scrollHeight;
}

// ---------------------------------------------------------------------------
// Render canvas + summary
// ---------------------------------------------------------------------------

function renderResults(data) {
    els.results.classList.remove('hidden');
    const canvas = els.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 900;
    const rowHeight = 36;
    const headerHeight = 60;
    const padding = 12;
    const cssHeight = headerHeight + padding + (data.results.length * rowHeight) + padding;
    canvas.style.height = cssHeight + 'px';
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const duration = data.audio_duration_seconds;
    const xFor = (t) => (t / duration) * cssWidth;

    drawWaveform(ctx, cssWidth, headerHeight, currentAudio ? currentAudio.samples : null);

    ctx.fillStyle = '#888';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    const tickStep = chooseTickStep(duration);
    for (let t = 0; t <= duration; t += tickStep) {
        const x = xFor(t);
        ctx.strokeStyle = '#ddd';
        ctx.beginPath();
        ctx.moveTo(x, headerHeight - 10);
        ctx.lineTo(x, headerHeight);
        ctx.stroke();
        ctx.fillText(t.toFixed(1) + 's', x + 2, headerHeight - 1);
    }

    const rowY0 = headerHeight + padding;
    const rowMeta = [];

    data.results.forEach((res, i) => {
        const y = rowY0 + i * rowHeight;
        ctx.fillStyle = '#333';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(res.short_name, 4, y + 11);
        if (res.error) {
            ctx.fillStyle = '#c62828';
            ctx.font = '10px system-ui, sans-serif';
            ctx.fillText('error: ' + res.error, 80, y + 11);
            return;
        }
        ctx.strokeStyle = '#eee';
        ctx.beginPath();
        ctx.moveTo(0, y + 24);
        ctx.lineTo(cssWidth, y + 24);
        ctx.stroke();
        const color = colorFor(res.short_name);
        for (const seg of res.segments) {
            const x0 = xFor(seg.start);
            const x1 = xFor(seg.end);
            const w = Math.max(1, x1 - x0);
            const alpha = 0.4 + 0.55 * Math.max(0, Math.min(1, seg.confidence || 0.7));
            ctx.fillStyle = hexToRgba(color, alpha);
            ctx.fillRect(x0, y + 16, w, rowHeight - 18);
            rowMeta.push({ x0, x1, y0: y + 16, y1: y + rowHeight - 2, seg });
        }
    });

    els.legend.innerHTML = '';
    for (const res of data.results) {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const sw = document.createElement('span');
        sw.className = 'swatch';
        sw.style.background = colorFor(res.short_name);
        const txt = document.createElement('span');
        txt.textContent = `${res.short_name}: ${res.segments ? res.segments.length : 0} segs`;
        item.append(sw, txt);
        els.legend.appendChild(item);
    }

    els.summary.innerHTML = '';
    for (const res of data.results) {
        const card = document.createElement('div');
        card.className = 'summary-card' + (res.error ? ' error' : '');
        card.style.borderLeftColor = colorFor(res.short_name);
        const total = (res.segments || []).reduce((a, s) => a + (s.end - s.start), 0);
        const pct = duration > 0 ? (total / duration * 100).toFixed(1) : '0.0';
        const speakers = new Set((res.segments || []).map(s => s.speaker_id));
        card.innerHTML = res.error
            ? `<strong>${escapeHtml(res.short_name)}</strong>error: ${escapeHtml(res.error)}`
            : `<strong>${escapeHtml(res.short_name)}</strong>` +
              `${res.segments.length} segments &middot; ${total.toFixed(2)}s speech (${pct}%) &middot; ` +
              `${speakers.size} speaker${speakers.size === 1 ? '' : 's'} &middot; ${res.elapsed_ms} ms`;
        els.summary.appendChild(card);
    }

    canvas.onclick = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        for (const m of rowMeta) {
            if (x >= m.x0 && x <= m.x1 && y >= m.y0 && y <= m.y1) {
                playRange(m.seg.start, m.seg.end);
                return;
            }
        }
    };
}

function chooseTickStep(duration) {
    if (duration <= 5) return 0.5;
    if (duration <= 15) return 1;
    if (duration <= 60) return 5;
    return 10;
}

function drawWaveform(ctx, w, h, samples) {
    ctx.fillStyle = '#f5f7fa';
    ctx.fillRect(0, 0, w, h);
    if (!samples || samples.length === 0) return;
    const midY = h / 2;
    ctx.strokeStyle = '#7a8aa0';
    ctx.beginPath();
    const samplesPerPx = samples.length / w;
    for (let x = 0; x < w; x++) {
        const i0 = Math.floor(x * samplesPerPx);
        const i1 = Math.min(samples.length, Math.floor((x + 1) * samplesPerPx));
        let lo = 0, hi = 0;
        for (let j = i0; j < i1; j++) {
            const s = samples[j];
            if (s < lo) lo = s;
            if (s > hi) hi = s;
        }
        const y0 = midY - hi * midY * 0.9;
        const y1 = midY - lo * midY * 0.9;
        ctx.moveTo(x + 0.5, y0);
        ctx.lineTo(x + 0.5, y1);
    }
    ctx.stroke();
}

function hexToRgba(hex, a) {
    const h = hex.replace('#', '');
    const r = parseInt(h.substring(0, 2), 16);
    const g = parseInt(h.substring(2, 4), 16);
    const b = parseInt(h.substring(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
}

let playingAudio = null;
function playRange(start, end) {
    if (!currentAudio) return;
    if (playingAudio) { playingAudio.pause(); playingAudio = null; }
    const i0 = Math.floor(start * SAMPLE_RATE);
    const i1 = Math.min(currentAudio.samples.length, Math.ceil(end * SAMPLE_RATE));
    const chunk = currentAudio.samples.slice(i0, i1);
    const blob = samplesToWavBlob(chunk);
    const url = URL.createObjectURL(blob);
    playingAudio = new Audio(url);
    playingAudio.addEventListener('ended', () => URL.revokeObjectURL(url));
    playingAudio.play();
}

// ---------------------------------------------------------------------------
// Live streaming (mic → /socket → server-side DetectStream RPC)
//
// The server connects to one vad gRPC backend at startup; /socket bridges
// the WebSocket to that backend's DetectStream. There's no ?model= picker
// here anymore — the server picks the backend from its CLI flag.
// ---------------------------------------------------------------------------

function wireLivePanel() {
    els.liveStart.addEventListener('click', startLive);
    els.liveStop.addEventListener('click', stopLive);
}

async function startLive() {
    if (liveState) return;
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        appendLive('mic error: ' + err.message, 'error');
        return;
    }
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${window.location.host}/socket`);
    ws.binaryType = 'arraybuffer';

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);

    // ScriptProcessor is deprecated but works in every browser without a
    // separate worklet file. AudioWorklet would be the modern replacement.
    const bufSize = 4096;
    const processor = audioCtx.createScriptProcessor(bufSize, 1, 1);
    const srcSampleRate = audioCtx.sampleRate;
    let resampleAcc = 0;
    const sampleStride = srcSampleRate / SAMPLE_RATE;
    let pendingChunk = [];
    const chunkTargetSamples = 1600; // 100 ms at 16 kHz

    processor.onaudioprocess = (ev) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const input = ev.inputBuffer.getChannelData(0);
        // Linear resample from srcSampleRate to 16 kHz.
        for (let i = 0; i < input.length; i++) {
            resampleAcc += 1;
            if (resampleAcc >= sampleStride) {
                pendingChunk.push(input[i]);
                resampleAcc -= sampleStride;
            }
        }
        while (pendingChunk.length >= chunkTargetSamples) {
            const out = new Float32Array(pendingChunk.splice(0, chunkTargetSamples));
            ws.send(out.buffer);
        }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    ws.addEventListener('open', () => {
        appendLive('WebSocket open', 'activity-on');
    });
    ws.addEventListener('message', (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            handleLiveEvent(msg);
        } catch (err) {
            appendLive('bad message: ' + err.message, 'error');
        }
    });
    ws.addEventListener('error', () => {
        appendLive('WebSocket error', 'error');
    });
    ws.addEventListener('close', () => {
        appendLive('WebSocket closed', 'activity-off');
        teardownLive();
    });

    liveState = { ws, audioCtx, source, processor, stream };
    els.liveStart.disabled = true;
    els.liveStop.disabled = false;
    setIndicator(false);
}

function stopLive() {
    if (!liveState) return;
    try {
        if (liveState.ws.readyState === WebSocket.OPEN) {
            liveState.ws.send('stop');
        }
    } catch (_) {}
    teardownLive();
}

function teardownLive() {
    if (!liveState) {
        els.liveStart.disabled = false;
        els.liveStop.disabled = true;
        return;
    }
    try { liveState.processor.disconnect(); } catch (_) {}
    try { liveState.source.disconnect(); } catch (_) {}
    try { liveState.audioCtx.close(); } catch (_) {}
    try { liveState.stream.getTracks().forEach(t => t.stop()); } catch (_) {}
    try { liveState.ws.close(); } catch (_) {}
    liveState = null;
    els.liveStart.disabled = false;
    els.liveStop.disabled = true;
    setIndicator(false);
}

function handleLiveEvent(msg) {
    if (msg.type === 'activity') {
        setIndicator(!!msg.speech_active);
        appendLive(
            `[${msg.timestamp.toFixed(2)}s] ${msg.speech_active ? 'SPEECH ON' : 'speech off'}`,
            msg.speech_active ? 'activity-on' : 'activity-off'
        );
    } else if (msg.type === 'segment') {
        appendLive(
            `[seg ${msg.start.toFixed(2)}-${msg.end.toFixed(2)}s] spk=${msg.speaker_id} conf=${(msg.confidence || 0).toFixed(2)}`,
            'segment'
        );
    } else if (msg.type === 'error') {
        appendLive('error: ' + msg.error, 'error');
    }
}

function setIndicator(active) {
    if (active) {
        els.speechIndicator.classList.remove('dim');
        els.speechIndicator.classList.add('active');
        els.speechIndicator.querySelector('.indicator-text').textContent = 'speech';
    } else {
        els.speechIndicator.classList.add('dim');
        els.speechIndicator.classList.remove('active');
        els.speechIndicator.querySelector('.indicator-text').textContent = 'idle';
    }
}

function appendLive(text, cls) {
    const line = document.createElement('div');
    line.className = 'line ' + (cls || '');
    line.textContent = text;
    els.liveSegments.appendChild(line);
    els.liveSegments.scrollTop = els.liveSegments.scrollHeight;
}
