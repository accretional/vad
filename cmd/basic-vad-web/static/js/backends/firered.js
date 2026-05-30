// Browser-side FireRedTeam DFSMN-VAD. Port of pkg/vad/firered.go.
//
// Kaldi-style fbank (Povey window, preemph 0.97) → CMVN (means/istd loaded
// from cmvn_means.f32 / cmvn_istd.f32 sidecars) → DFSMN ONNX → sigmoid per
// 10 ms frame → boxcar smooth + 4-state hysteresis machine.

import { FbankComputer, fbankDefaults } from '../dsp/fbank.js';

const INPUT_MELS = 80;
const FRAME_STEP_MS = 10;
const SMOOTH_WINDOW = 5;
const SPEECH_THRESHOLD = 0.4;
const MIN_SPEECH_FRAMES = 20;   // 200 ms
const MIN_SILENCE_FRAMES = 20;  // 200 ms
const BACKFILL_FRAMES = 5;

export const FIRERED_AUX_FILES = ['cmvn_means.f32', 'cmvn_istd.f32'];

let fbankCache = null;
function fb() {
    if (!fbankCache) {
        const opts = fbankDefaults(); // Povey window, preemph 0.97 — matches FireRed
        opts.numMelBins = INPUT_MELS;
        fbankCache = new FbankComputer(opts);
    }
    return fbankCache;
}

// run(ort, session, samples, aux) — aux is { 'cmvn_means.f32': Uint8Array, ... }
export async function run(ort, session, samples, aux) {
    if (!samples || samples.length === 0) return [];
    const means = parseFloat32LE(aux['cmvn_means.f32']);
    const istd = parseFloat32LE(aux['cmvn_istd.f32']);
    if (means.length !== INPUT_MELS || istd.length !== INPUT_MELS) {
        throw new Error(`firered: cmvn dims means=${means.length} istd=${istd.length}, want ${INPUT_MELS}`);
    }
    // Scale to int16 range (matches Go ProcessAudio + FunASR/FireRed front-ends).
    const scaled = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) scaled[i] = samples[i] * 32768.0;

    const feats = fb().compute(scaled);
    const T = feats.length / INPUT_MELS;
    if (T === 0) return [];

    // CMVN: (x - mean) * istd. In-place.
    for (let t = 0; t < T; t++) {
        for (let j = 0; j < INPUT_MELS; j++) {
            feats[t * INPUT_MELS + j] = (feats[t * INPUT_MELS + j] - means[j]) * istd[j];
        }
    }

    const featT = new ort.Tensor('float32', feats, [1, T, INPUT_MELS]);
    const out = await session.run({ feat: featT });
    const probs = out.probs.data;
    if (probs.length !== T) {
        throw new Error(`firered: probs len ${probs.length} != frames ${T}`);
    }
    return extractSegments(Array.from(probs));
}

function parseFloat32LE(u8) {
    if (!u8) throw new Error('firered: missing cmvn aux file');
    const buf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    return new Float32Array(buf);
}

function extractSegments(probs) {
    const n = probs.length;
    if (n === 0) return [];
    // 1. Boxcar smooth (cumulative mean for the first frames matching upstream).
    const smoothed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        let lo = i - SMOOTH_WINDOW + 1;
        if (lo < 0) lo = 0;
        let sum = 0;
        for (let j = lo; j <= i; j++) sum += probs[j];
        smoothed[i] = sum / (i - lo + 1);
    }
    const STATE_SILENCE = 0, STATE_POSS_SPEECH = 1, STATE_SPEECH = 2, STATE_POSS_SILENCE = 3;
    let state = STATE_SILENCE;
    const segments = [];
    let transitionFrame = 0;
    let lastSpeechStart = -1;
    let lastEndedSegEnd = 0;
    const flushSpeech = (end) => {
        if (lastSpeechStart >= 0 && end > lastSpeechStart) {
            segments.push({
                start: framesToSec(lastSpeechStart),
                end: framesToSec(end),
                speaker_id: 0,
                confidence: 1.0,
            });
        }
        lastSpeechStart = -1;
    };
    for (let i = 0; i < n; i++) {
        const isSpeech = smoothed[i] >= SPEECH_THRESHOLD;
        switch (state) {
            case STATE_SILENCE:
                if (isSpeech) { state = STATE_POSS_SPEECH; transitionFrame = i; }
                break;
            case STATE_POSS_SPEECH:
                if (!isSpeech) {
                    state = STATE_SILENCE;
                } else if (i - transitionFrame + 1 >= MIN_SPEECH_FRAMES) {
                    state = STATE_SPEECH;
                    let start = transitionFrame - BACKFILL_FRAMES;
                    if (start < 0) start = 0;
                    if (start < lastEndedSegEnd) start = lastEndedSegEnd;
                    lastSpeechStart = start;
                }
                break;
            case STATE_SPEECH:
                if (!isSpeech) { state = STATE_POSS_SILENCE; transitionFrame = i; }
                break;
            case STATE_POSS_SILENCE:
                if (isSpeech) {
                    state = STATE_SPEECH;
                } else if (i - transitionFrame + 1 >= MIN_SILENCE_FRAMES) {
                    state = STATE_SILENCE;
                    const end = transitionFrame;
                    flushSpeech(end);
                    lastEndedSegEnd = end;
                }
                break;
        }
    }
    if (state === STATE_SPEECH || state === STATE_POSS_SILENCE) {
        flushSpeech(n);
    }
    return segments;
}

function framesToSec(f) { return f * FRAME_STEP_MS / 1000.0; }
