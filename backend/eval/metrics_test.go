package eval

import (
	"math"
	"testing"
)

func TestClassifyEntity(t *testing.T) {
	cases := []struct {
		answer, entity string
		want           EntityHit
	}{
		{"FlashAttention is fast.", "FlashAttention", EntityExact},
		{"Flash Attention is fast.", "FlashAttention", EntityNearMiss}, // spacing
		{"flashattention is fast.", "FlashAttention", EntityNearMiss},  // casing
		{"FlashAttension is fast.", "FlashAttention", EntityNearMiss},  // typo (edit dist 1)
		{"It uses softmax attention.", "FlashAttention", EntityAbsent}, // absent
		{"BERT is bidirectional.", "BERT", EntityExact},
		{"Bert is bidirectional.", "BERT", EntityNearMiss},
		{"The LoRA method.", "LoRA", EntityExact},
		{"The Lora method.", "LoRA", EntityNearMiss},
		{"A transformer model.", "Transformer", EntityNearMiss}, // lowercase
	}
	for _, c := range cases {
		if got := ClassifyEntity(c.answer, c.entity); got != c.want {
			t.Errorf("ClassifyEntity(%q,%q) = %d, want %d", c.answer, c.entity, got, c.want)
		}
	}
}

func TestDistinct2(t *testing.T) {
	// Fully repetitive text -> low distinct-2.
	rep := Distinct2("the cat the cat the cat the cat")
	diverse := Distinct2("the quick brown fox jumps over a lazy dog today")
	if !(diverse > rep) {
		t.Errorf("expected diverse distinct2 (%v) > repetitive (%v)", diverse, rep)
	}
	if Distinct2("one") != 1.0 {
		t.Errorf("single-token distinct2 should be 1.0")
	}
}

func TestGroundedness(t *testing.T) {
	vocab := []string{"FlashAttention", "BERT", "Transformer"}
	// Both mentioned entities are in context -> 1.0.
	g, ok := Groundedness("FlashAttention extends the Transformer.", "context: FlashAttention, Transformer architecture", vocab)
	if !ok || math.Abs(g-1.0) > 1e-9 {
		t.Errorf("grounded = %v ok=%v, want 1.0 true", g, ok)
	}
	// One of two mentioned is absent from context -> 0.5.
	g2, _ := Groundedness("FlashAttention and BERT.", "context mentions FlashAttention only", vocab)
	if math.Abs(g2-0.5) > 1e-9 {
		t.Errorf("grounded = %v, want 0.5", g2)
	}
	// No known entities mentioned -> undefined.
	if _, ok := Groundedness("a generic sentence", "ctx", vocab); ok {
		t.Errorf("expected undefined groundedness when no entities mentioned")
	}
}

func TestLevenshtein(t *testing.T) {
	if levenshtein("flashattension", "flashattention") != 1 {
		t.Errorf("expected edit distance 1")
	}
}
