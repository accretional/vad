package fbank

import (
	"math"
	"math/cmplx"
)

// fftRadix2 computes the FFT of x in-place using iterative Cooley-Tukey
// radix-2 decimation-in-time. len(x) must be a power of 2.
func fftRadix2(x []complex128) {
	n := len(x)
	if n&(n-1) != 0 {
		panic("fbank.fft: length must be a power of 2")
	}
	if n <= 1 {
		return
	}

	// Bit-reversal permutation.
	for i, j := 0, 0; i < n; i++ {
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
		m := n >> 1
		for m >= 1 && j >= m {
			j -= m
			m >>= 1
		}
		j += m
	}

	// Butterflies, increasing transform sizes 2, 4, 8, ... n.
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		// Primitive size-th root of unity (using DFT sign convention: exp(-2πi/N)).
		wn := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for k := 0; k < n; k += size {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				t := w * x[k+j+half]
				x[k+j+half] = x[k+j] - t
				x[k+j] = x[k+j] + t
				w *= wn
			}
		}
	}
}

// nextPow2 returns the smallest power of 2 >= n. Returns 1 for n <= 1.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}
