//go:build !cgo_pca

package cgo

import (
	"errors"
	"math"
)

// LoadModel is a no-op in the pure-Go build: ProjectToPCA always uses the
// sigmoid fallback regardless of any model file. It returns a sentinel error so
// startup can log that the real Eigen-backed PCA engine is not active (build
// with -tags cgo_pca and the compiled libpca to enable it).
func LoadModel(path string) error {
	return errors.New("built without cgo_pca; using pure-Go fallback projection")
}

// ProjectToPCA projects a float32 embedding to 2D when the C++ PCA engine is
// not linked (build tag cgo_pca absent). It is a dev/test stub, not PCA: the
// embedding is dotted with two fixed pseudo-random sign axes and squashed with
// tanh. The axes are decorrelated and the gain is chosen so distinct inputs
// spread across roughly [-1,1]^2 — enough for the router's centroids, paths,
// and the UI's routing map to exercise all outcomes without the real engine.
// The earlier mean-based version collapsed every realistic embedding to ~(0,0),
// which pinned the router to a single regime/path in fallback mode.
func ProjectToPCA(embedding []float32) ([2]float64, error) {
	if len(embedding) == 0 {
		return [2]float64{}, nil
	}
	var d0, d1 float64
	for i, v := range embedding {
		f := float64(v)
		// Two independent Rademacher (±1) axes — a tiny random-projection
		// sketch. The signs come from a hash mix rather than modular index
		// arithmetic: simple multiplier patterns have low bits that cycle in
		// lockstep, which made the two axes perfectly anti-correlated and
		// collapsed the projection onto a 1-D line.
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
	return [2]float64{math.Tanh(d0 * scale), math.Tanh(d1 * scale)}, nil
}

// rademacher returns a deterministic pseudo-random sign for (index, axis),
// using an xxhash-style avalanche so the two axes are decorrelated.
func rademacher(i, axis uint32) bool {
	h := i*2654435761 + axis*0x9e3779b9
	h ^= h >> 16
	h *= 2246822519
	h ^= h >> 13
	return h&1 == 0
}
