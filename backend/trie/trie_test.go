package trie

import "testing"

func TestTrieSearchPrefix(t *testing.T) {
	tr := New()
	tr.Insert("FlashAttention")
	cases := map[string]bool{
		"Flash":            true,
		"flashatten":       true, // case-insensitive
		"FlashAttention":   true,
		"Flush":            false,
		"FlashAttentionXY": false,
	}
	for prefix, want := range cases {
		if got := tr.SearchPrefix(prefix); got != want {
			t.Errorf("SearchPrefix(%q) = %v, want %v", prefix, got, want)
		}
	}
}

func TestTrieValidateAndComplete(t *testing.T) {
	tr := New()
	tr.Insert("FlashAttention")
	if got, ok := tr.ValidateAndComplete("FlashAttent"); !ok || got != "FlashAttention" {
		t.Errorf("ValidateAndComplete partial = %q,%v; want FlashAttention,true", got, ok)
	}
	if got, ok := tr.ValidateAndComplete("FlashAttention"); !ok || got != "FlashAttention" {
		t.Errorf("ValidateAndComplete exact = %q,%v; want FlashAttention,true", got, ok)
	}
	if _, ok := tr.ValidateAndComplete("Zzz"); ok {
		t.Error("ValidateAndComplete of unknown prefix should be false")
	}
}

func TestTrieAllWords(t *testing.T) {
	tr := New()
	words := []string{"FlashAttention", "BERT", "Attention Is All You Need"}
	for _, w := range words {
		tr.Insert(w)
	}
	got := map[string]bool{}
	for _, w := range tr.AllWords() {
		got[w] = true
	}
	for _, w := range words {
		if !got[w] {
			t.Errorf("AllWords missing %q", w)
		}
	}
}
