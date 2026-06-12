package synthesis

import (
	"math"
	"strings"
)

// TFIDFMatrix computes a TF-IDF matrix for the given documents.
// Returns a map from term → per-doc weight.
func TFIDFMatrix(docs []string) map[string][]float64 {
	N := len(docs)
	if N == 0 {
		return nil
	}

	// Tokenize
	tokenized := make([][]string, N)
	for i, doc := range docs {
		tokenized[i] = tokenize(doc)
	}

	// DF
	df := make(map[string]int)
	for _, tokens := range tokenized {
		seen := make(map[string]bool)
		for _, t := range tokens {
			if !seen[t] {
				df[t]++
				seen[t] = true
			}
		}
	}

	// Build per-doc TF-IDF vectors (sublinear TF, smoothed IDF) and L2-normalize.
	result := make(map[string][]float64)
	for docIdx, tokens := range tokenized {
		tf := make(map[string]int)
		for _, t := range tokens {
			tf[t]++
		}
		var norm float64
		vals := make(map[string]float64)
		for term, count := range tf {
			subTF := 1.0 + math.Log(float64(count))
			idf := math.Log(float64(N+1)/float64(df[term]+1)) + 1.0
			v := subTF * idf
			vals[term] = v
			norm += v * v
		}
		if norm > 0 {
			norm = math.Sqrt(norm)
			for term, v := range vals {
				if result[term] == nil {
					result[term] = make([]float64, N)
				}
				result[term][docIdx] = v / norm
			}
		}
	}
	return result
}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	var tokens []string
	buf := strings.Builder{}
	for _, c := range s {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			buf.WriteRune(c)
		} else if buf.Len() > 0 {
			tokens = append(tokens, buf.String())
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		tokens = append(tokens, buf.String())
	}
	return tokens
}
