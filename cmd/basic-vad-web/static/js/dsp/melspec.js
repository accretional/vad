// Port of melspec/melspec.go + melspec/mel.go.
//
// NeMo / torchaudio MelSpectrogram: Hann periodic window, Slaney mel scale,
// STFT center=True with reflect padding, whole-signal preemph, log(mel + 2^-24).
// Used by the MarbleNet VAD backend.

import { fftRadix2 } from './fft.js';

export function nemoDefaults() {
    return {
        sampleRate: 16000,
        winLenSamples: 400,   // 25 ms
        hopLenSamples: 160,   // 10 ms
        nFFT: 512,
        numMelBins: 80,
        fMinHz: 0,
        fMaxHz: 0,            // 0 ⇒ sampleRate/2
        preemphCoeff: 0.97,
        logOffset: Math.pow(2, -24),
        padToMultiple: 2,
    };
}

// Slaney piecewise mel scale (librosa htk=False).
const SLANEY_F_SP = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ = 1000.0;
const SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
const SLANEY_LOG_STEP = Math.log(6.4) / 27.0;

function hzToMelSlaney(hz) {
    if (hz < SLANEY_MIN_LOG_HZ) return hz / SLANEY_F_SP;
    return SLANEY_MIN_LOG_MEL + Math.log(hz / SLANEY_MIN_LOG_HZ) / SLANEY_LOG_STEP;
}
function melToHzSlaney(mel) {
    if (mel < SLANEY_MIN_LOG_MEL) return mel * SLANEY_F_SP;
    return SLANEY_MIN_LOG_HZ * Math.exp(SLANEY_LOG_STEP * (mel - SLANEY_MIN_LOG_MEL));
}

function makeMelFilterbankSlaney(numBins, fftBins, sampleRate, lowFreq, highFreq) {
    const fftSize = (fftBins - 1) * 2;
    if (highFreq <= 0 || highFreq > sampleRate / 2) highFreq = sampleRate / 2;
    const melLow = hzToMelSlaney(lowFreq);
    const melHigh = hzToMelSlaney(highFreq);
    const melPoints = new Float64Array(numBins + 2);
    for (let i = 0; i < melPoints.length; i++) {
        melPoints[i] = melLow + (melHigh - melLow) * i / (numBins + 1);
    }
    const hzPoints = new Float64Array(melPoints.length);
    for (let i = 0; i < melPoints.length; i++) hzPoints[i] = melToHzSlaney(melPoints[i]);
    const binHz = new Float64Array(fftBins);
    for (let k = 0; k < fftBins; k++) binHz[k] = k * sampleRate / fftSize;
    const filt = [];
    for (let b = 0; b < numBins; b++) {
        const left = hzPoints[b], center = hzPoints[b + 1], right = hzPoints[b + 2];
        // Slaney area normalization: enorm = 2 / (right - left).
        const enorm = 2.0 / (right - left);
        const row = new Float64Array(fftBins);
        for (let k = 0; k < fftBins; k++) {
            const f = binHz[k];
            if (f < left || f > right) continue;
            if (f <= center) row[k] = (f - left) / (center - left) * enorm;
            else            row[k] = (right - f) / (right - center) * enorm;
        }
        filt.push(row);
    }
    return filt;
}

// Hann periodic: w[i] = 0.5 - 0.5*cos(2π*i/L). torchaudio default.
function hannPeriodic(length) {
    const w = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / length);
    }
    return w;
}

export class MelSpec {
    constructor(opts = nemoDefaults()) {
        this.opts = { ...nemoDefaults(), ...opts };
        const o = this.opts;
        if (o.nFFT < o.winLenSamples || (o.nFFT & (o.nFFT - 1)) !== 0) {
            throw new Error('melspec: nFFT must be a power of 2 >= winLenSamples');
        }
        this.fftBins = (o.nFFT >> 1) + 1;
        const fmax = o.fMaxHz > 0 ? o.fMaxHz : o.sampleRate / 2;
        this.window = hannPeriodic(o.winLenSamples);
        this.mel = makeMelFilterbankSlaney(o.numMelBins, this.fftBins, o.sampleRate, o.fMinHz, fmax);
        this._re = new Float64Array(o.nFFT);
        this._im = new Float64Array(o.nFFT);
        this._power = new Float64Array(this.fftBins);
    }

    // computeChannelsFirstFlat returns { feats, tMel } where feats is a
    // Float32Array of length numMelBins * tMel, laid out as [b * tMel + t]
    // (matching Go's ComputeChannelsFirstFlat). Ready to feed a NeMo ONNX
    // model with input shape [1, numMelBins, tMel].
    computeChannelsFirstFlat(samples) {
        const o = this.opts;
        const n = samples.length;
        if (n === 0) return { feats: null, tMel: 0 };

        // 1. Whole-signal preemphasis.
        const preemphed = new Float64Array(n);
        preemphed[0] = samples[0];
        if (o.preemphCoeff !== 0) {
            for (let i = 1; i < n; i++) {
                preemphed[i] = samples[i] - o.preemphCoeff * samples[i - 1];
            }
        } else {
            for (let i = 1; i < n; i++) preemphed[i] = samples[i];
        }

        // 2. Reflection pad by nFFT/2 on each side (torch.stft pad_mode=reflect).
        const pad = o.nFFT >> 1;
        const padded = new Float64Array(n + 2 * pad);
        for (let i = 0; i < pad; i++) padded[i] = preemphed[pad - i];
        padded.set(preemphed, pad);
        for (let i = 0; i < pad; i++) {
            let idx = n - 2 - i;
            if (idx < 0) idx = 0;
            padded[pad + n + i] = preemphed[idx];
        }

        // 3. Frame count (matches NeMo get_seq_len: no +1 vs torch.stft).
        const tMel = Math.floor((padded.length - o.nFFT) / o.hopLenSamples);
        if (tMel <= 0) return { feats: null, tMel: 0 };
        let finalT = tMel;
        if (o.padToMultiple > 0) {
            const rem = finalT % o.padToMultiple;
            if (rem !== 0) finalT += o.padToMultiple - rem;
        }

        const nBins = o.numMelBins;
        const feats = new Float32Array(nBins * finalT);
        const winOffset = (o.nFFT - o.winLenSamples) >> 1;
        for (let t = 0; t < tMel; t++) {
            this._computeFrame(padded, t * o.hopLenSamples, winOffset, feats, t, finalT);
        }
        // pad_to right-pads with literal zeros (log-of-1 region); leave as 0.
        return { feats, tMel: finalT };
    }

    _computeFrame(padded, srcOff, winOffset, out, t, tMel) {
        const o = this.opts;
        const re = this._re;
        const im = this._im;
        for (let i = 0; i < o.nFFT; i++) { re[i] = 0; im[i] = 0; }
        for (let i = 0; i < o.winLenSamples; i++) {
            re[winOffset + i] = padded[srcOff + winOffset + i] * this.window[i];
        }
        fftRadix2(re, im);
        const fftBins = this.fftBins;
        const power = this._power;
        for (let k = 0; k < fftBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        const nBins = o.numMelBins;
        const offs = o.logOffset;
        for (let b = 0; b < nBins; b++) {
            const row = this.mel[b];
            let sum = 0;
            for (let k = 0; k < fftBins; k++) sum += row[k] * power[k];
            out[b * tMel + t] = Math.log(sum + offs);
        }
    }
}
