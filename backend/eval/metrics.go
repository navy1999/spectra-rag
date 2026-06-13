package eval

import "strings"

// EntityHit classifies how an expected entity appears in an answer.
type EntityHit int

const (
	EntityAbsent   EntityHit = iota // not present in any form
	EntityNearMiss                  // present but mis-cased, mis-spaced, or a small typo
	EntityExact                     // present with the exact canonical spelling
)

// ClassifyEntity decides whether `entity` appears in `answer` exactly, as a
// near-miss, or not at all. Exactness is case- and spacing-sensitive: "BERT"
// counts as exact, "Bert" / "B.E.R.T" / "FlashAttension" as near-misses. This
// is the measurable target of the trie entity guard (A2), which rewrites
// near-misses to the canonical form.
func ClassifyEntity(answer, entity string) EntityHit {
	if entity == "" {
		return EntityAbsent
	}
	if strings.Contains(answer, entity) {
		return EntityExact
	}
	na := normalizeAlnum(strings.ToLower(answer))
	ne := normalizeAlnum(strings.ToLower(entity))
	if ne == "" {
		return EntityAbsent
	}
	// Case/spacing/punctuation differences only.
	if strings.Contains(na, ne) {
		return EntityNearMiss
	}
	// Small typo: slide a window of comparable length and look for a close match.
	if len(ne) >= 4 {
		maxDist := 1
		if len(ne) >= 8 {
			maxDist = 2
		}
		for w := len(ne); w <= len(ne)+maxDist; w++ {
			for i := 0; i+w <= len(na); i++ {
				if levenshtein(na[i:i+w], ne) <= maxDist {
					return EntityNearMiss
				}
			}
		}
	}
	return EntityAbsent
}

// Distinct2 is the ratio of unique bigrams to total bigrams in the answer.
// Higher means less repetition; the SVD redundancy penalty (A3) aims to raise
// it on redundant contexts. Returns 1.0 for answers too short to have bigrams.
func Distinct2(text string) float64 {
	toks := tokens(text)
	if len(toks) < 2 {
		return 1.0
	}
	seen := make(map[string]struct{})
	total := 0
	for i := 0; i+1 < len(toks); i++ {
		seen[toks[i]+" "+toks[i+1]] = struct{}{}
		total++
	}
	return float64(len(seen)) / float64(total)
}

// EntitiesMentioned returns the vocabulary entries that appear (case-insensitively)
// in the answer.
func EntitiesMentioned(answer string, vocab []string) []string {
	la := strings.ToLower(answer)
	var out []string
	for _, v := range vocab {
		if v != "" && strings.Contains(la, strings.ToLower(v)) {
			out = append(out, v)
		}
	}
	return out
}

// Groundedness is the fraction of graph entities mentioned in the answer that
// also appear in the retrieved context: a cheap hallucination proxy. The bool
// is false when the answer mentions no known entities (metric undefined).
func Groundedness(answer, context string, vocab []string) (float64, bool) {
	mentioned := EntitiesMentioned(answer, vocab)
	if len(mentioned) == 0 {
		return 0, false
	}
	lc := strings.ToLower(context)
	grounded := 0
	for _, e := range mentioned {
		if strings.Contains(lc, strings.ToLower(e)) {
			grounded++
		}
	}
	return float64(grounded) / float64(len(mentioned)), true
}

func normalizeAlnum(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func tokens(s string) []string {
	var out []string
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else if b.Len() > 0 {
			out = append(out, b.String())
			b.Reset()
		}
	}
	if b.Len() > 0 {
		out = append(out, b.String())
	}
	return out
}

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
