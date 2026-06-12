package trie

import (
	"strings"
	"sync/atomic"
)

// HallucinationInterceptions counts entity tokens the interceptor has corrected
// since process startup. Surfaced via the X-Hallucination-Interceptions header.
var HallucinationInterceptions atomic.Int64

// StreamInterceptor performs token-stream entity correction. It buffers incoming
// tokens until a word boundary, then checks each completed word against a
// vocabulary of entity words built from the knowledge-graph node names. A
// capitalized word that is a near-miss (small Levenshtein distance) of a known
// entity is rewritten to its canonical spelling — correcting entity
// hallucinations token-by-token, the capability free providers deny us by
// withholding logit_bias.
//
// Scope/limitation: correction is per-word and ASCII-oriented (the graph names
// are ASCII); multi-word entity phrases are not realigned.
type StreamInterceptor struct {
	vocab   map[string]string // lowercase word -> canonical spelling
	words   []string          // canonical entity words, for edit-distance search
	pending strings.Builder   // current partial word (no boundary seen yet)
	count   int               // corrections made by this interceptor instance
}

// Count returns the number of corrections this interceptor has made. Tokens are
// processed sequentially within a single request, so no synchronization is
// needed for the per-instance counter.
func (si *StreamInterceptor) Count() int { return si.count }

func NewInterceptor(t *Trie) *StreamInterceptor {
	si := &StreamInterceptor{vocab: map[string]string{}}
	for _, name := range t.AllWords() {
		for _, w := range splitWords(name) {
			if len(w) < 3 {
				continue
			}
			lw := strings.ToLower(w)
			if _, ok := si.vocab[lw]; !ok {
				si.vocab[lw] = w
				si.words = append(si.words, w)
			}
		}
	}
	return si
}

// ProcessToken consumes one streamed token and returns the text now safe to emit
// (possibly empty while a word is still mid-stream) and whether a correction
// occurred within that emitted text.
func (si *StreamInterceptor) ProcessToken(token string) (string, bool) {
	si.pending.WriteString(token)
	s := si.pending.String()

	cut := strings.LastIndexAny(s, " \t\n\r")
	if cut < 0 {
		return "", false // no word boundary yet — keep buffering
	}
	ready := s[:cut+1]
	rest := s[cut+1:]
	si.pending.Reset()
	si.pending.WriteString(rest)

	return si.correctText(ready)
}

// Flush returns and clears any buffered partial word, correcting it if it is a
// near-miss entity. Call once at end of stream.
func (si *StreamInterceptor) Flush() string {
	s := si.pending.String()
	si.pending.Reset()
	if s == "" {
		return ""
	}
	out, _ := si.correctText(s)
	return out
}

// correctText rewrites each word in s that is a near-miss of a known entity,
// preserving all original whitespace and punctuation.
func (si *StreamInterceptor) correctText(s string) (string, bool) {
	var out, word strings.Builder
	changed := false

	flushWord := func() {
		if word.Len() == 0 {
			return
		}
		corrected, ok := si.correctWord(word.String())
		word.Reset()
		out.WriteString(corrected)
		if ok {
			changed = true
		}
	}

	for _, r := range s {
		if isWordRune(r) {
			word.WriteRune(r)
		} else {
			flushWord()
			out.WriteRune(r)
		}
	}
	flushWord()
	return out.String(), changed
}

// correctWord returns the canonical spelling for w when w looks like an entity
// (capitalized, length >= 4) and either matches a known entity word (casing
// normalized) or is within a tight edit distance of one. Otherwise w is
// returned unchanged.
func (si *StreamInterceptor) correctWord(w string) (string, bool) {
	if len(w) < 4 || !isUpper(rune(w[0])) {
		return w, false
	}
	lw := strings.ToLower(w)

	if canon, ok := si.vocab[lw]; ok {
		if canon != w {
			HallucinationInterceptions.Add(1)
			si.count++
			return canon, true
		}
		return w, false
	}

	maxDist := 1
	if len(w) >= 8 {
		maxDist = 2
	}
	best := ""
	bestDist := maxDist + 1
	for _, cand := range si.words {
		lc := strings.ToLower(cand)
		if len(cand) < 4 || lw[0] != lc[0] || abs(len(cand)-len(w)) > maxDist {
			continue
		}
		if d := levenshtein(lw, lc); d < bestDist {
			bestDist = d
			best = cand
		}
	}
	if best != "" && bestDist <= maxDist {
		HallucinationInterceptions.Add(1)
		si.count++
		return best, true
	}
	return w, false
}

func splitWords(s string) []string {
	var words []string
	var b strings.Builder
	for _, r := range s {
		if isWordRune(r) {
			b.WriteRune(r)
		} else if b.Len() > 0 {
			words = append(words, b.String())
			b.Reset()
		}
	}
	if b.Len() > 0 {
		words = append(words, b.String())
	}
	return words
}

func isWordRune(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
}

func isUpper(r rune) bool { return r >= 'A' && r <= 'Z' }

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// levenshtein computes the edit distance between two ASCII strings.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min3(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
