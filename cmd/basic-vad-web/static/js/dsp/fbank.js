// Port of fbank/fbank.go, fbank/mel.go, fbank/window.go.
//
// Kaldi-compatible log-Mel-filterbank features for the FSMN-VAD and
// FireRedVAD backends. Match `compute-fbank-feats` defaults: HTK mel scale,
// Povey or Hamming window, snip_edges=true, preemph=0.97, RemoveDCOffset.
//
// Float32-throughput pipeline (doubles internally where parity matters).

import { fftRadix2, nextPow2 } from './fft.js';

export const WINDOW_HAMMING = 'hamming';
export const WINDOW_POVEY = 'povey';

// Match fbank.Defaults() in Go.
export function fbankDefaults() {
    return {
        sampleRate: 16000,
        frameLengthMs: 25,
        frameShiftMs: 10,
        numMelBins: 80,
        windowType: WINDOW_POVEY,
        lowFreqHz: 20,
        highFreqHz: 0,        // 0 ⇒ sampleRate/2
        preemphCoeff: 0.97,
        removeDCOffset: true,
        snipEdges: true,
        energyFloor: 0,
        useLog: true,
    };
}

// makeWindow returns a Float64Array of length L.
function makeWindow(length, type) {
    const w = new Float64Array(length);
    if (length === 1) { w[0] = 1; return w; }
    const denom = length - 1;
    if (type === WINDOW_HAMMING) {
        for (let i = 0; i < length; i++) {
            w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / denom);
        }
    } else {
        // Povey (default if unknown).
        for (let i = 0; i < length; i++) {
            const h = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / denom);
            w[i] = Math.pow(h, 0.85);
        }
    }
    return w;
}

// HTK mel scale, used by Kaldi.
function hzToMel(hz) { return 1127.0 * Math.log(1.0 + hz / 700.0); }
function melToHz(mel) { return 700.0 * (Math.exp(mel / 1127.0) - 1.0); }

// makeMelFilterbank returns [numBins][fftBins] (array of Float64Array rows).
function makeMelFilterbank(numBins, fftBins, sampleRate, lowFreq, highFreq) {
    const fftSize = (fftBins - 1) * 2;
    if (highFreq <= 0 || highFreq > sampleRate / 2) highFreq = sampleRate / 2;

    const melLow = hzToMel(lowFreq);
    const melHigh = hzToMel(highFreq);
    const melPoints = new Float64Array(numBins + 2);
    for (let i = 0; i < melPoints.length; i++) {
        melPoints[i] = melLow + (melHigh - melLow) * i / (numBins + 1);
    }
    const hzPoints = new Float64Array(melPoints.length);
    for (let i = 0; i < melPoints.length; i++) hzPoints[i] = melToHz(melPoints[i]);

    const binHz = new Float64Array(fftBins);
    for (let k = 0; k < fftBins; k++) binHz[k] = k * sampleRate / fftSize;

    const filt = [];
    for (let b = 0; b < numBins; b++) {
        const left = hzPoints[b], center = hzPoints[b + 1], right = hzPoints[b + 2];
        const row = new Float64Array(fftBins);
        for (let k = 0; k < fftBins; k++) {
            const f = binHz[k];
            if (f < left || f > right) continue;
            if (f <= center) row[k] = (f - left) / (center - left);
            else            row[k] = (right - f) / (right - center);
        }
        filt.push(row);
    }
    return filt;
}

// FbankComputer mirrors fbank.Fbank. Construct once, call compute() many.
// Not safe for concurrent use (scratch buffers).
export class FbankComputer {
    constructor(opts = fbankDefaults()) {
        this.opts = { ...fbankDefaults(), ...opts };
        const sr = this.opts.sampleRate;
        this.frameLength = Math.round(sr * this.opts.frameLengthMs / 1000);
        this.frameShift = Math.round(sr * this.opts.frameShiftMs / 1000);
        this.fftSize = nextPow2(this.frameLength);
        this.fftBins = (this.fftSize >> 1) + 1;
        const high = this.opts.highFreqHz > 0 ? this.opts.highFreqHz : sr / 2;
        this.window = makeWindow(this.frameLength, this.opts.windowType);
        this.mel = makeMelFilterbank(this.opts.numMelBins, this.fftBins, sr, this.opts.lowFreqHz, high);
        // Scratch.
        this._frameRe = new Float64Array(this.fftSize);
        this._frameIm = new Float64Array(this.fftSize);
        this._power = new Float64Array(this.fftBins);
    }

    numFrames(numSamples) {
        if (!this.opts.snipEdges) {
            return Math.floor((numSamples + this.frameShift / 2) / this.frameShift);
        }
        if (numSamples < this.frameLength) return 0;
        return Math.floor((numSamples - this.frameLength) / this.frameShift) + 1;
    }

    // compute returns a Float32Array of length numFrames * numMelBins, with
    // row-major layout: out[t*nbins + b]. The Go API returns [][]float32 but
    // the flat layout is friendlier for the LFR-stack + tensor copy that
    // follows in FSMN.
    compute(samples) {
        const nFrames = this.numFrames(samples.length);
        if (nFrames === 0) return new Float32Array(0);
        const nBins = this.opts.numMelBins;
        const out = new Float32Array(nFrames * nBins);
        for (let fi = 0; fi < nFrames; fi++) {
            const start = fi * this.frameShift;
            this._computeFrame(samples, start, out, fi * nBins);
        }
        return out;
    }

    _computeFrame(samples, start, out, outOff) {
        const re = this._frameRe;
        const im = this._frameIm;
        const N = this.fftSize;
        const L = this.frameLength;
        // Load frame, zero-pad tail.
        for (let i = 0; i < L; i++) re[i] = samples[start + i];
        for (let i = L; i < N; i++) re[i] = 0;
        for (let i = 0; i < N; i++) im[i] = 0;

        if (this.opts.removeDCOffset) {
            let mean = 0;
            for (let i = 0; i < L; i++) mean += re[i];
            mean /= L;
            for (let i = 0; i < L; i++) re[i] -= mean;
        }
        if (this.opts.preemphCoeff > 0) {
            const c = this.opts.preemphCoeff;
            // Kaldi: y[0] = x[0] - c*x[0]; y[i] = x[i] - c*x[i-1].
            let prev = re[0];
            re[0] = prev - c * prev;
            for (let i = 1; i < L; i++) {
                const cur = re[i];
                re[i] = cur - c * prev;
                prev = cur;
            }
        }
        for (let i = 0; i < L; i++) re[i] *= this.window[i];

        fftRadix2(re, im);

        // Power spectrum.
        const fftBins = this.fftBins;
        const power = this._power;
        for (let k = 0; k < fftBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];

        // Mel * power + log. Matches kaldi-native-fbank: floor = float32-epsilon
        // before log, so silent frames log to ~-15.94.
        const FLOOR = 1.1920928955078125e-7;
        let floor = this.opts.energyFloor;
        if (floor < FLOOR) floor = FLOOR;
        const nBins = this.opts.numMelBins;
        for (let b = 0; b < nBins; b++) {
            const row = this.mel[b];
            let sum = 0;
            for (let k = 0; k < fftBins; k++) sum += row[k] * power[k];
            if (this.opts.useLog) {
                if (sum < floor) sum = floor;
                out[outOff + b] = Math.log(sum);
            } else {
                out[outOff + b] = sum;
            }
        }
    }
}

// lfrStack groups every `m` consecutive D-dim frames into one (m*D)-dim frame,
// sliding by `n` between outputs. If t < m, the last frame is replicated to
// reach length m (matches FunASR WavFrontend). Returns a Float32Array of
// shape [tLfr * m * d] row-major, plus tLfr.
//
// `feats` is a flat Float32Array of length t*d row-major.
export function lfrStack(feats, t, d, m, n) {
    if (t === 0) return { stacked: new Float32Array(0), tLfr: 0 };
    // Pad-by-replication if t < m. This means appending (m-t) copies of the
    // last d-vector at the end of `feats`.
    let frames = feats;
    if (t < m) {
        const padded = new Float32Array(m * d);
        padded.set(feats);
        const lastOff = (t - 1) * d;
        for (let i = t; i < m; i++) {
            for (let j = 0; j < d; j++) padded[i * d + j] = feats[lastOff + j];
        }
        frames = padded;
        t = m;
    }
    const tLfr = Math.floor((t - m) / n) + 1;
    const stacked = new Float32Array(tLfr * m * d);
    for (let i = 0; i < tLfr; i++) {
        for (let j = 0; j < m; j++) {
            const srcOff = (i * n + j) * d;
            const dstOff = i * (m * d) + j * d;
            for (let k = 0; k < d; k++) stacked[dstOff + k] = frames[srcOff + k];
        }
    }
    return { stacked, tLfr };
}
