// Browser-side Silero VAD. Port of pkg/vad/silero.go.
//
// Chunked-state inference: feed the .onnx model 512-sample (32 ms @ 16 kHz)
// chunks one at a time, ping-pong the (state, context) tensors between calls.
// Output is one speech probability per chunk; post-process with a two-state
// hysteresis machine to extract segments.

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 512;
const CHUNK_MS = 32;
const STATE_DIM_0 = 2;
const STATE_DIM_1 = 128;
const CONTEXT_DIM = 64;
const SPEECH_THRESHOLD = 0.5;
const MIN_SPEECH_FRAMES = 4;   // ~128 ms
const MIN_SILENCE_FRAMES = 10; // ~320 ms

export const SILERO_AUX_FILES = []; // model.onnx only

// run(ort, session, samples) → segments[].
// `ort` is the loaded onnxruntime-web namespace; `session` is an
// ort.InferenceSession built from the model.onnx bytes.
export async function run(ort, session, samples) {
    if (!samples || samples.length === 0) return [];
    const nChunks = Math.floor(samples.length / CHUNK_SAMPLES);
    if (nChunks === 0) return [];

    // Persistent tensors reused across chunks. We re-create the input/state/
    // context per call because ort.Tensor instances aren't intended to be
    // re-bound to fresh data; allocation here is cheap relative to inference.
    const stateShape = [STATE_DIM_0, 1, STATE_DIM_1];
    const ctxShape = [1, CONTEXT_DIM];

    let stateBuf = new Float32Array(STATE_DIM_0 * 1 * STATE_DIM_1);
    let ctxBuf = new Float32Array(1 * CONTEXT_DIM);

    const probs = new Float32Array(nChunks);
    for (let i = 0; i < nChunks; i++) {
        const inputChunk = samples.subarray(i * CHUNK_SAMPLES, (i + 1) * CHUNK_SAMPLES);
        // ort.Tensor copies the typed array into its own buffer, so we can
        // safely subarray() above.
        const inputT = new ort.Tensor('float32', inputChunk, [1, CHUNK_SAMPLES]);
        const stateT = new ort.Tensor('float32', stateBuf, stateShape);
        const ctxT = new ort.Tensor('float32', ctxBuf, ctxShape);
        const out = await session.run({ input: inputT, state: stateT, context: ctxT });
        probs[i] = out.prob.data[0];
        // Ping-pong for next chunk.
        stateBuf = new Float32Array(out.stateN.data);
        ctxBuf = new Float32Array(out.contextN.data);
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
        if (probs[i] >= SPEECH_THRESHOLD) {
            if (!inSpeech) {
                speechRun++;
                if (speechRun >= MIN_SPEECH_FRAMES) {
                    inSpeech = true;
                    start = i - speechRun + 1;
                    if (start < 0) start = 0;
                }
            }
            silenceRun = 0;
        } else {
            speechRun = 0;
            if (inSpeech) {
                silenceRun++;
                if (silenceRun >= MIN_SILENCE_FRAMES) {
                    segments.push({
                        start: frameToSec(start),
                        end: frameToSec(i - silenceRun + 1),
                        speaker_id: 0,
                        confidence: 1.0,
                    });
                    inSpeech = false;
                    silenceRun = 0;
                }
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

function frameToSec(frame) {
    return frame * CHUNK_MS / 1000.0;
}
