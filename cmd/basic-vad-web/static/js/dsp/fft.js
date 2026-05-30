// Port of fbank/fft.go and melspec/fft.go.
//
// Radix-2 iterative Cooley-Tukey, in-place, using two parallel arrays for the
// real and imaginary parts (avoids a per-bin object allocation). Sign
// convention exp(-2*pi*i/N) — same as the Go code, NumPy, and torch.fft.
//
// nextPow2(n) returns the smallest power of 2 >= n; >= 1 for n <= 1.

export function nextPow2(n) {
    if (n <= 1) return 1;
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

// fftRadix2 transforms (re, im) in place. Length must be a power of 2.
export function fftRadix2(re, im) {
    const n = re.length;
    if (n !== im.length) throw new Error('fft: re/im length mismatch');
    if ((n & (n - 1)) !== 0) throw new Error('fft: length must be a power of 2');
    if (n <= 1) return;

    // Bit-reversal permutation.
    for (let i = 0, j = 0; i < n; i++) {
        if (i < j) {
            const tr = re[i]; re[i] = re[j]; re[j] = tr;
            const ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        let m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterflies, transform sizes 2, 4, 8, ..., n.
    for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1;
        // Primitive size-th root of unity: wn = cos(-2π/size) + i*sin(-2π/size)
        const theta = -2 * Math.PI / size;
        const wnR = Math.cos(theta);
        const wnI = Math.sin(theta);
        for (let k = 0; k < n; k += size) {
            let wR = 1, wI = 0;
            for (let j = 0; j < half; j++) {
                // t = w * x[k+j+half]
                const xR = re[k + j + half];
                const xI = im[k + j + half];
                const tR = wR * xR - wI * xI;
                const tI = wR * xI + wI * xR;
                // x[k+j+half] = x[k+j] - t
                re[k + j + half] = re[k + j] - tR;
                im[k + j + half] = im[k + j] - tI;
                // x[k+j] = x[k+j] + t
                re[k + j] += tR;
                im[k + j] += tI;
                // w *= wn
                const nR = wR * wnR - wI * wnI;
                const nI = wR * wnI + wI * wnR;
                wR = nR; wI = nI;
            }
        }
    }
}
