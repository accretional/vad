package melspec

import (
	"math"
	"math/cmplx"
)

// fftRadix2 computes the FFT of x in-place using iterative Cooley-Tukey
// radix-2 decimation-in-time. len(x) must be a power of 2.
//
// Forward DFT, no normalization (matches numpy / torch / kaldi forward
// FFTs — divide by N if you need the inverse normalized).
func fftRadix2(x []complex128) {
	n := len(x)
	if n&(n-1) != 0 {
		panic("melspec.fft: length must be a power of 2")
	}
	if n <= 1 {
		return
	}
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
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
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
