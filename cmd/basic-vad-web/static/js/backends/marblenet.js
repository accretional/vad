// Browser-side MarbleNet VAD. Port of pkg/vad/marblenet.go.
//
// Pipeline: NeMo Slaney log-mel feats [80, T_mel] → CNN → 2-class softmax per
// 20 ms frame (encoder subsamples by 2). Onset/offset hysteresis on
// p_speech.

import { MelSpec } from '../dsp/melspec.js';

const ON_THRESHOLD = 0.5;
const OFF_THRESHOLD = 0.3;
const MIN_ON_FRAMES = 10;  // 200 ms
const MIN_OFF_FRAMES = 5;  // 100 ms
const FRAME_STEP_MS = 20;  // each output frame represents 20 ms

export const MARBLENET_AUX_FILES = []; // preprocessor.yaml not needed (NeMo defaults baked in)

let melCache = null;
function mel() {
    if (!melCache) melCache = new MelSpec();
    return melCache;
}

export async function run(ort, session, samples) {
    if (!samples || samples.length === 0) return [];
    const { feats, tMel } = mel().computeChannelsFirstFlat(samples);
    if (tMel === 0) return [];

    const inputT = new ort.Tensor('float32', feats, [1, 80, tMel]);
    const out = await session.run({ audio_signal: inputT });
    // Output: [1, T_out, 2] with T_out = T_mel / 2.
    const logits = out.outputs.data;
    const tOut = Math.floor(tMel / 2);
    if (logits.length !== tOut * 2) {
        throw new Error(`marblenet: unexpected output len ${logits.length} (expected ${tOut * 2})`);
    }
    const probs = new Float32Array(tOut);
    for (let t = 0; t < tOut; t++) {
        const a = logits[t * 2];
        const b = logits[t * 2 + 1];
        const maxL = a > b ? a : b;
        const expA = Math.exp(a - maxL);
        const expB = Math.exp(b - maxL);
        probs[t] = expB / (expA + expB);
    }
    return extractSegments(probs);
}

function extractSegments(probs) {
    const n = probs.length;
    if (n === 0) return [];
    const segments = [];
    let inSpeech = false;
    let start = 0;
    let speechRun = 0;
    let silenceRun = 0;
    for (let i = 0; i < n; i++) {
        const p = probs[i];
        if (!inSpeech) {
            if (p >= ON_THRESHOLD) {
                speechRun++;
                if (speechRun >= MIN_ON_FRAMES) {
                    inSpeech = true;
                    start = i - speechRun + 1;
                    if (start < 0) start = 0;
                }
            } else {
                speechRun = 0;
            }
        } else {
            if (p < OFF_THRESHOLD) {
                silenceRun++;
                if (silenceRun >= MIN_OFF_FRAMES) {
                    segments.push({
                        start: frameToSec(start),
                        end: frameToSec(i - silenceRun + 1),
                        speaker_id: 0,
                        confidence: 1.0,
                    });
                    inSpeech = false;
                    silenceRun = 0;
                    speechRun = 0;
                }
            } else {
                silenceRun = 0;
            }
        }
    }
    if (inSpeech) {
        segments.push({
            start: frameToSec(start),
            end: frameToSec(n),
            speaker_id: 0,
            confidence: 1.0,
        });
    }
    return segments;
}

function frameToSec(frame) { return frame * FRAME_STEP_MS / 1000.0; }
