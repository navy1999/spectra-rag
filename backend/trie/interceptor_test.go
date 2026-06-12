package trie

import (
	"strings"
	"testing"
	"time"
)

func testInterceptor() *StreamInterceptor {
	tr := New()
	for _, name := range []string{
		"FlashAttention: Fast and Memory-Efficient Exact Attention",
		"BERT: Pre-training of Deep Bidirectional Transformers",
		"Attention Is All You Need",
		"Transformer Architecture",
		"Tri Dao",
	} {
		tr.Insert(name)
	}
	return NewInterceptor(tr)
}

// runStream feeds tokens through the interceptor and returns the fully emitted
// text (including the flushed tail) plus the correction count.
func runStream(si *StreamInterceptor, tokens []string) (string, int) {
	var out strings.Builder
	for _, tk := range tokens {
		emit, _ := si.ProcessToken(tk)
		out.WriteString(emit)
	}
	out.WriteString(si.Flush())
	return out.String(), si.Count()
}

func TestInterceptor_PassThroughNormalText(t *testing.T) {
	si := testInterceptor()
	in := []string{"The ", "quick ", "brown ", "fox ", "jumps ", "over"}
	got, n := runStream(si, in)
	if want := "The quick brown fox jumps over"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if n != 0 {
		t.Errorf("expected 0 corrections on normal text, got %d", n)
	}
}

func TestInterceptor_CorrectsNearMissEntity(t *testing.T) {
	si := testInterceptor()
	// "FlashAttenton" drops the 'i' -> edit distance 1 from "FlashAttention".
	got, n := runStream(si, []string{"FlashAttenton ", "is ", "fast"})
	if !strings.Contains(got, "FlashAttention") {
		t.Errorf("expected correction to FlashAttention, got %q", got)
	}
	if n != 1 {
		t.Errorf("expected 1 correction, got %d", n)
	}
}

func TestInterceptor_NormalizesCasing(t *testing.T) {
	si := testInterceptor()
	got, n := runStream(si, []string{"Bert ", "uses ", "transformers"})
	if !strings.HasPrefix(got, "BERT ") {
		t.Errorf("expected BERT casing normalization, got %q", got)
	}
	if n != 1 {
		t.Errorf("expected 1 correction, got %d", n)
	}
}

func TestInterceptor_ReassemblesWordAcrossTokens(t *testing.T) {
	si := testInterceptor()
	// The entity arrives split across three tokens.
	got, n := runStream(si, []string{"Flash", "Atten", "ton "})
	if strings.TrimSpace(got) != "FlashAttention" {
		t.Errorf("got %q, want FlashAttention", got)
	}
	if n != 1 {
		t.Errorf("expected 1 correction, got %d", n)
	}
}

func TestInterceptor_FlushReturnsBufferedWord(t *testing.T) {
	si := testInterceptor()
	// No trailing whitespace: the word stays buffered until Flush.
	if emit, _ := si.ProcessToken("Bert"); emit != "" {
		t.Errorf("expected empty emit while mid-word, got %q", emit)
	}
	if tail := si.Flush(); tail != "BERT" {
		t.Errorf("Flush = %q, want BERT", tail)
	}
}

// TestInterceptor_DoesNotSleep guards against the original implementation, which
// slept 1 second in the streaming hot path on every correction.
func TestInterceptor_DoesNotSleep(t *testing.T) {
	si := testInterceptor()
	start := time.Now()
	si.ProcessToken("FlashAttenton ") // triggers a correction
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Errorf("ProcessToken took %v — interceptor must not block the stream", elapsed)
	}
}

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"abc", "abc", 0},
		{"abc", "abd", 1},
		{"flashattenton", "flashattention", 1},
		{"kitten", "sitting", 3},
	}
	for _, c := range cases {
		if got := levenshtein(c.a, c.b); got != c.want {
			t.Errorf("levenshtein(%q,%q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

func BenchmarkInterceptorProcess(b *testing.B) {
	tokens := strings.Fields(
		"The FlashAttenton paper by Tri Dao improves Transfomer attention with " +
			"a memory efficient exact softmax that scales to long sequences and " +
			"reduces peak memory from quadratic to linear in the sequence length")
	for i := range tokens {
		tokens[i] += " "
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		si := testInterceptor()
		runStream(si, tokens)
	}
}
