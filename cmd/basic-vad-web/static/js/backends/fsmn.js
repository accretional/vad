// Browser-side FunASR FSMN-VAD. Port of pkg/vad/fsmn.go.
//
// Kaldi-style fbank (Hamming window, preemph 0.97, 80 mels) → LFR m=5 stack
// → CMVN ((x + means) * rescales, means already negated in am.mvn) → ONNX
// (zero state on each call — offline inference) → silence-prob threshold +
// smoothing + hangover.

import { FbankComputer, fbankDefaults, lfrStack } from '../dsp/fbank.js';

const INPUT_DIM = 400;   // 80 mels × LFR m=5
const OUTPUT_DIM = 248;
const CACHE_CHAN = 128;
const CACHE_LEN = 19;
const CACHE_DIM = 1;
const NUM_CACHES = 4;
const LFR_M = 5, LFR_N = 1;
const FRAME_STEP_MS = 10;
const SILENCE_PROB_THRESHOLD = 0.2;
const SMOOTH_WINDOW_FRAMES = 20;
const SMOOTH_MIN_SPEECH = 15;
const SILENCE_HANGOVER_FRAMES = 80; // 800 ms

export const FSMN_AUX_FILES = ['am.mvn'];

let fbankCache = null;
function fb() {
    if (!fbankCache) {
        const opts = fbankDefaults();
        opts.windowType = 'hamming';
        opts.numMelBins = 80;
        opts.frameLengthMs = 25;
        opts.frameShiftMs = 10;
        opts.preemphCoeff = 0.97;
        opts.removeDCOffset = true;
        opts.snipEdges = true;
        fbankCache = new FbankComputer(opts);
    }
    return fbankCache;
}

export async function run(ort, session, samples, aux) {
    if (!samples || samples.length === 0) return [];
    const mvnText = new TextDecoder().decode(aux['am.mvn']);
    const { means, rescales } = parseFSMNMVN(mvnText);
    if (means.length !== INPUT_DIM || rescales.length !== INPUT_DIM) {
        throw new Error(`fsmn: mvn dims means=${means.length} rescales=${rescales.length}, want ${INPUT_DIM}`);
    }

    // 1. Scale to int16 range (FunASR WavFrontend pre-fbank).
    const scaled = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) scaled[i] = samples[i] * 32768.0;

    // 2. Fbank features [T, 80].
    const feats = fb().compute(scaled);
    const T = feats.length / 80;
    if (T === 0) return [];

    // 3. LFR stack m=5, n=1 → [tLfr, 400].
    const { stacked, tLfr } = lfrStack(feats, T, 80, LFR_M, LFR_N);
    if (tLfr === 0) return [];

    // 4. CMVN: (x + means) * rescales — means are already negated in am.mvn,
    //    so this is effectively (x - mean) * (1/std). In-place.
    for (let i = 0; i < tLfr; i++) {
        for (let j = 0; j < INPUT_DIM; j++) {
            stacked[i * INPUT_DIM + j] = (stacked[i * INPUT_DIM + j] + means[j]) * rescales[j];
        }
    }

    // 5. Zero-init the 4 caches (offline / one-shot inference).
    const cacheShape = [1, CACHE_CHAN, CACHE_LEN, CACHE_DIM];
    const cacheLen = 1 * CACHE_CHAN * CACHE_LEN * CACHE_DIM;
    const feeds = {
        speech: new ort.Tensor('float32', stacked, [1, tLfr, INPUT_DIM]),
    };
    for (let i = 0; i < NUM_CACHES; i++) {
        feeds[`in_cache${i}`] = new ort.Tensor('float32', new Float32Array(cacheLen), cacheShape);
    }
    const out = await session.run(feeds);
    const logits = out.logits.data;
    if (logits.length !== tLfr * OUTPUT_DIM) {
        throw new Error(`fsmn: unexpected logits len ${logits.length} (expected ${tLfr * OUTPUT_DIM})`);
    }
    const silenceProb = new Float32Array(tLfr);
    for (let i = 0; i < tLfr; i++) silenceProb[i] = logits[i * OUTPUT_DIM]; // index 0 = silence
    return extractSegments(silenceProb);
}

// parseFSMNMVN parses FunASR's <Nnet>...</Nnet> am.mvn format. Returns
// { means: Float32Array, rescales: Float32Array }.
function parseFSMNMVN(text) {
    let section = '';
    let means = null, rescales = null;
    for (const lineRaw of text.split('\n')) {
        const line = lineRaw.trim();
        if (line.startsWith('<AddShift>')) { section = 'means'; continue; }
        if (line.startsWith('<Rescale>')) { section = 'rescales'; continue; }
        if (line.startsWith('<LearnRateCoef>') && section) {
            const lb = line.indexOf('[');
            const rb = line.lastIndexOf(']');
            if (lb < 0 || rb < 0 || rb <= lb) continue;
            const fields = line.slice(lb + 1, rb).trim().split(/\s+/);
            const vals = new Float32Array(fields.length);
            for (let i = 0; i < fields.length; i++) vals[i] = parseFloat(fields[i]);
            if (section === 'means') means = vals;
            else                     rescales = vals;
            section = '';
        }
    }
    if (!means || !rescales) throw new Error('fsmn: am.mvn missing <AddShift> or <Rescale>');
    return { means, rescales };
}

function extractSegments(silenceProb) {
    const n = silenceProb.length;
    if (n === 0) return [];
    const raw = new Uint8Array(n);
    for (let i = 0; i < n; i++) raw[i] = silenceProb[i] <= SILENCE_PROB_THRESHOLD ? 1 : 0;
    // Sliding-window smoothing.
    const smoothed = new Uint8Array(n);
    const half = SMOOTH_WINDOW_FRAMES >> 1;
    for (let i = 0; i < n; i++) {
        let lo = i - half, hi = i + half;
        if (lo < 0) lo = 0;
        if (hi > n) hi = n;
        let count = 0;
        for (let j = lo; j < hi; j++) if (raw[j]) count++;
        smoothed[i] = count >= SMOOTH_MIN_SPEECH ? 1 : 0;
    }
    const segments = [];
    let inSpeech = false;
    let start = 0;
    let silenceRun = 0;
    for (let i = 0; i < n; i++) {
        if (smoothed[i]) {
            if (!inSpeech) { inSpeech = true; start = i; }
            silenceRun = 0;
        } else if (inSpeech) {
            silenceRun++;
            if (silenceRun >= SILENCE_HANGOVER_FRAMES) {
                segments.push({
                    start: framesToSec(start),
                    end: framesToSec(i - silenceRun + 1),
                    speaker_id: 0,
                    confidence: 1.0,
                });
                inSpeech = false;
                silenceRun = 0;
            }
        }
    }
    if (inSpeech) {
        segments.push({
            start: framesToSec(start),
            end: framesToSec(n),
            speaker_id: 0,
            confidence: 1.0,
        });
    }
    return segments;
}

function framesToSec(f) { return f * FRAME_STEP_MS / 1000.0; }
