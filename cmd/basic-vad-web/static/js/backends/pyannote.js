// Browser-side Pyannote Segmentation 3.0. Port of pkg/vad/pyannote.go.
//
// 10-second windows of raw float32 audio → log-softmax over 7 powerset
// classes per 17 ms frame (589 frames per window) → per-speaker segment
// extraction with cross-window merging.
//
// Powerset decoding maps argmax class index → set of active speakers per
// frame:
//   0: silence, 1: {A}, 2: {B}, 3: {A,B}, 4: {C}, 5: {A,C}, 6: {B,C}.
// (Max 2 of 3 speakers active per frame — pyannote 3.0 trained capacity.)

const SAMPLE_RATE = 16000;
const WINDOW_SIZE = 10 * SAMPLE_RATE; // 160_000 samples
const NUM_FRAMES = 589;
const NUM_CLASSES = 7;
const NUM_SPEAKERS = 3;

// Bit-masks: powersetMaskByClass[c] has bit `spk` set iff class c implies
// speaker `spk` is active. Faster than nested loops over the variable-length
// arrays in the Go code.
const POWERSET_MASK_BY_CLASS = new Uint8Array([
    0b000, // 0: silence
    0b001, // 1: A
    0b010, // 2: B
    0b011, // 3: A+B
    0b100, // 4: C
    0b101, // 5: A+C
    0b110, // 6: B+C
]);

export const PYANNOTE_AUX_FILES = []; // model.onnx only

export async function run(ort, session, samples) {
    if (!samples || samples.length === 0) return [];

    const allSegments = [];
    // Reuse an input buffer; ort.Tensor copies, so we can clobber it between
    // calls without invalidating earlier ones.
    const inputBuf = new Float32Array(WINDOW_SIZE);

    for (let offset = 0; offset < samples.length; offset += WINDOW_SIZE) {
        // Zero-fill, then copy whatever's available.
        inputBuf.fill(0);
        const copyEnd = Math.min(offset + WINDOW_SIZE, samples.length);
        inputBuf.set(samples.subarray(offset, copyEnd));

        const inputT = new ort.Tensor('float32', inputBuf, [1, 1, WINDOW_SIZE]);
        const out = await session.run({ input_values: inputT });
        const logits = out.logits.data;
        if (logits.length !== NUM_FRAMES * NUM_CLASSES) {
            throw new Error(`pyannote: logits len ${logits.length} != ${NUM_FRAMES * NUM_CLASSES}`);
        }
        const windowStart = offset / SAMPLE_RATE;
        const segs = extractSegments(logits, windowStart);
        for (const s of segs) allSegments.push(s);
    }
    return mergeSegments(allSegments);
}

function extractSegments(logits, windowStartSec) {
    const frameDuration = 10.0 / NUM_FRAMES;
    const segments = [];

    // Per-speaker active state. Use parallel arrays to avoid object thrash.
    const active = new Uint8Array(NUM_SPEAKERS);
    const starts = new Float64Array(NUM_SPEAKERS);
    const sumConf = new Float32Array(NUM_SPEAKERS);
    const counts = new Uint32Array(NUM_SPEAKERS);

    for (let frame = 0; frame < NUM_FRAMES; frame++) {
        const frameStart = windowStartSec + frame * frameDuration;
        const base = frame * NUM_CLASSES;

        // Argmax over the 7 log-probs.
        let maxIdx = 0;
        let maxLogP = logits[base];
        for (let c = 1; c < NUM_CLASSES; c++) {
            const v = logits[base + c];
            if (v > maxLogP) { maxLogP = v; maxIdx = c; }
        }
        const conf = Math.exp(maxLogP);
        const mask = POWERSET_MASK_BY_CLASS[maxIdx];

        for (let spk = 0; spk < NUM_SPEAKERS; spk++) {
            const isActive = (mask & (1 << spk)) !== 0;
            if (isActive) {
                if (!active[spk]) {
                    active[spk] = 1;
                    starts[spk] = frameStart;
                    sumConf[spk] = 0;
                    counts[spk] = 0;
                }
                sumConf[spk] += conf;
                counts[spk]++;
            } else if (active[spk]) {
                segments.push({
                    start: starts[spk],
                    end: frameStart,
                    speaker_id: spk,
                    confidence: sumConf[spk] / counts[spk],
                });
                active[spk] = 0;
            }
        }
    }

    const endTime = windowStartSec + 10.0;
    for (let spk = 0; spk < NUM_SPEAKERS; spk++) {
        if (active[spk]) {
            segments.push({
                start: starts[spk],
                end: endTime,
                speaker_id: spk,
                confidence: sumConf[spk] / counts[spk],
            });
        }
    }
    return segments;
}

// mergeSegments merges adjacent same-speaker segments across window boundaries
// when the gap is < 0.1 s. Mirrors mergePyannoteSegments in Go.
function mergeSegments(segments) {
    if (segments.length === 0) return [];
    const out = [];
    for (const seg of segments) {
        if (out.length > 0) {
            const last = out[out.length - 1];
            if (last.speaker_id === seg.speaker_id && seg.start - last.end < 0.1) {
                last.end = seg.end;
                continue;
            }
        }
        out.push({ ...seg });
    }
    return out;
}
