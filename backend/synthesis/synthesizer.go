package synthesis

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// ComputeFrequencyPenalty uses SVD on the TF-IDF matrix to compute a
// variance-ratio-weighted frequency penalty, capped at 2.0.
// This is Algorithm 3: PCA variance-weighted frequency penalty.
func ComputeFrequencyPenalty(docs []string) float64 {
	if len(docs) < 2 {
		return 0.0
	}

	tfidf := TFIDFMatrix(docs)
	if len(tfidf) == 0 {
		return 0.0
	}

	// Build dense matrix (terms × docs)
	terms := make([]string, 0, len(tfidf))
	for t := range tfidf {
		terms = append(terms, t)
	}
	sort.Strings(terms)

	nTerms := len(terms)
	nDocs := len(docs)
	data := make([]float64, nTerms*nDocs)
	for i, term := range terms {
		for j, v := range tfidf[term] {
			data[i*nDocs+j] = v
		}
	}

	m := mat.NewDense(nTerms, nDocs, data)
	var svd mat.SVD
	if ok := svd.Factorize(m, mat.SVDThin); !ok {
		return 0.0
	}

	vals := svd.Values(nil)
	var totalVar float64
	for _, v := range vals {
		totalVar += v * v
	}
	if totalVar == 0 {
		return 0.0
	}
	// First component variance ratio
	varianceRatio := (vals[0] * vals[0]) / totalVar

	// frequency_penalty = min(2.0, ratio * 3.5)
	penalty := math.Min(2.0, varianceRatio*3.5)
	return penalty
}

// PenaltyInstruction translates a frequency penalty (0..2) into a synthesis
// directive injected into the system prompt. The higher the penalty — i.e. the
// more redundant / low-rank the retrieved context, as measured by the dominance
// of its first singular value — the stronger the instruction to avoid
// repetition. This is our stand-in for the frequency_penalty API parameter that
// free-tier providers omit: we cannot bias logits, so we steer the model in
// natural language instead. Returns "" when the context is diverse enough to
// need no nudging.
func PenaltyInstruction(penalty float64) string {
	switch {
	case penalty >= 1.5:
		return "The retrieved context is highly repetitive. Be concise and state each fact only once."
	case penalty >= 0.8:
		return "Some retrieved context overlaps. Avoid redundant phrasing and consolidate related points."
	default:
		return ""
	}
}
