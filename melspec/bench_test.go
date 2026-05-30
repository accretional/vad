package melspec_test

import (
	"testing"
	"time"

	"github.com/accretional/vad/melspec"
)

func BenchmarkCompute_10s(b *testing.B) {
	full := loadAudioF32(&testing.T{})
	if len(full) > 10*16000 {
		full = full[:10*16000]
	}
	m, _ := melspec.New(melspec.NeMoDefaults())
	_ = m.Compute(full)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Compute(full)
	}
	b.ReportMetric(float64(b.N)*10/(float64(b.Elapsed())/float64(time.Second)), "rt_x")
}
