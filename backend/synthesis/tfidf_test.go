package synthesis

import "testing"

func TestComputeFrequencyPenalty_FewDocs(t *testing.T) {
	if p := ComputeFrequencyPenalty(nil); p != 0 {
		t.Errorf("nil docs penalty = %v, want 0", p)
	}
	if p := ComputeFrequencyPenalty([]string{"only one document"}); p != 0 {
		t.Errorf("single doc penalty = %v, want 0", p)
	}
}

func TestComputeFrequencyPenalty_Range(t *testing.T) {
	docs := []string{"attention transformer model", "bert pretraining language", "lora adaptation tuning"}
	p := ComputeFrequencyPenalty(docs)
	if p < 0 || p > 2.0 {
		t.Errorf("penalty %v out of range [0,2]", p)
	}
}

// TestComputeFrequencyPenalty_RedundancyOrdering checks the central property:
// highly redundant context (one dominant singular direction) yields a larger
// penalty than diverse context.
func TestComputeFrequencyPenalty_RedundancyOrdering(t *testing.T) {
	redundant := []string{"attention attention", "attention attention", "attention attention"}
	diverse := []string{"alpha beta", "gamma delta", "epsilon zeta"}
	pr := ComputeFrequencyPenalty(redundant)
	pd := ComputeFrequencyPenalty(diverse)
	if pr <= pd {
		t.Errorf("expected redundant penalty (%v) > diverse penalty (%v)", pr, pd)
	}
}

func TestTFIDFMatrix_Shape(t *testing.T) {
	docs := []string{"attention model", "attention transformer model"}
	m := TFIDFMatrix(docs)
	if len(m) == 0 {
		t.Fatal("expected non-empty TF-IDF matrix")
	}
	for term, vec := range m {
		if len(vec) != len(docs) {
			t.Errorf("term %q vector length %d, want %d", term, len(vec), len(docs))
		}
	}
}

func TestPenaltyInstruction(t *testing.T) {
	if PenaltyInstruction(0.2) != "" {
		t.Error("low penalty should produce no instruction")
	}
	if PenaltyInstruction(1.0) == "" {
		t.Error("medium penalty should produce an instruction")
	}
	if PenaltyInstruction(1.8) == "" {
		t.Error("high penalty should produce an instruction")
	}
}

func BenchmarkComputeFrequencyPenalty(b *testing.B) {
	docs := []string{
		"[paper] FlashAttention: Fast and Memory-Efficient Exact Attention (2022)",
		"[paper] Attention Is All You Need (2017)",
		"[topic] Efficient Attention",
		"[author] Tri Dao",
		"[topic] Transformer Architecture",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ComputeFrequencyPenalty(docs)
	}
}
