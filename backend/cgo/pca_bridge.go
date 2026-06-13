//go:build !cgo_pca

package cgo

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// pcaModel is the fitted projection produced by scripts/fit_pca.py or
// scripts/fit_lda.py: components is (n_components x n_features), mean is
// (n_features). The projection is the same matvec regardless of how the
// components were fitted (unsupervised PCA or supervised PCA->LDA), so a single
// loader serves both; `Method` records which fitter produced it (for reporting).
type pcaModel struct {
	Components [][]float64 `json:"components"`
	Mean       []float64   `json:"mean"`
	Method     string      `json:"method"`
}

// loadedModel is set once at startup by LoadModel and read-only thereafter.
var loadedModel *pcaModel

// LoadModel loads a fitted PCA model from JSON so the pure-Go build can do REAL
// PCA — the projection is just components·(x-mean), a matrix-vector multiply, so
// it needs no cgo/Eigen. The C++ engine (build tag cgo_pca) remains an optional
// fast path; this is what lets a CGO_ENABLED=0 deployment route on a true signal.
// Returns an error (and leaves projection on the dev fallback) if the model is
// missing or malformed.
func LoadModel(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("pca model not loaded: %w", err)
	}
	var m pcaModel
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("parse pca model: %w", err)
	}
	if len(m.Components) < 2 || len(m.Mean) == 0 {
		return fmt.Errorf("pca model needs >=2 components and a mean")
	}
	for i, c := range m.Components[:2] {
		if len(c) != len(m.Mean) {
			return fmt.Errorf("pca component %d has %d features, mean has %d", i, len(c), len(m.Mean))
		}
	}
	loadedModel = &m
	return nil
}

// LoadedMethod returns the fitter that produced the loaded model (e.g. "pca",
// "pca16_lda"), or "" if no model is loaded. Used to label the eval report so
// it reflects the actual projection in use rather than a hardcoded name.
func LoadedMethod() string {
	if loadedModel == nil {
		return ""
	}
	return loadedModel.Method
}

// ProjectToPCA projects an embedding to 2D. With a fitted model loaded whose
// feature count matches the embedding, it does the real PCA projection.
// Otherwise it falls back to a random-projection sketch (dev/offline only).
func ProjectToPCA(embedding []float32) ([2]float64, error) {
	if len(embedding) == 0 {
		return [2]float64{}, nil
	}
	if m := loadedModel; m != nil && len(m.Mean) == len(embedding) {
		var out [2]float64
		for c := 0; c < 2; c++ {
			comp := m.Components[c]
			var sum float64
			for i, v := range embedding {
				sum += comp[i] * (float64(v) - m.Mean[i])
			}
			out[c] = sum
		}
		return out, nil
	}
	return rademacherProj(embedding), nil
}

// rademacherProj is a deterministic random-projection sketch used only when no
// fitted model matches the embedding (offline/dev). It is NOT PCA; it just
// spreads distinct inputs across roughly [-1,1]^2 so the router and UI can be
// exercised without a real model.
func rademacherProj(embedding []float32) [2]float64 {
	var d0, d1 float64
	for i, v := range embedding {
		f := float64(v)
		if rademacher(uint32(i), 0) {
			d0 += f
		} else {
			d0 -= f
		}
		if rademacher(uint32(i), 1) {
			d1 += f
		} else {
			d1 -= f
		}
	}
	scale := 2.0 / math.Sqrt(float64(len(embedding)))
	return [2]float64{math.Tanh(d0 * scale), math.Tanh(d1 * scale)}
}

func rademacher(i, axis uint32) bool {
	h := i*2654435761 + axis*0x9e3779b9
	h ^= h >> 16
	h *= 2246822519
	h ^= h >> 13
	return h&1 == 0
}
