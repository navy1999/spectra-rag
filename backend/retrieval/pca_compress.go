package retrieval

import "gonum.org/v1/gonum/mat"

// pcaBasis is a fitted PCA projection. project(x) = components·(x − mean), giving
// a k-dim compressed vector. Used to shrink the node-embedding retrieval index
// for large ingested corpora (the memory↔recall tradeoff measured in
// data/compression_curve.md: ~8× at K=128 for ~0.85 recall@10).
type pcaBasis struct {
	components [][]float64 // k × d, row j is the j-th principal direction
	mean       []float64   // d
	k          int
}

// fitPCABasis fits a k-component PCA over vecs (one row per node). Returns nil
// when compression is not applicable (k out of range, or too few samples), so
// callers fall back to a full-dim index.
func fitPCABasis(vecs [][]float32, k int) *pcaBasis {
	n := len(vecs)
	if n == 0 {
		return nil
	}
	d := len(vecs[0])
	if k <= 0 || k >= d {
		return nil // 0/negative = off; k>=d = no compression
	}
	if k >= n {
		k = n - 1
	}
	if k <= 0 {
		return nil
	}

	mean := make([]float64, d)
	for _, v := range vecs {
		for i, val := range v {
			mean[i] += float64(val)
		}
	}
	for i := range mean {
		mean[i] /= float64(n)
	}

	data := make([]float64, n*d)
	for r, v := range vecs {
		for i, val := range v {
			data[r*d+i] = float64(val) - mean[i]
		}
	}
	var svd mat.SVD
	if !svd.Factorize(mat.NewDense(n, d, data), mat.SVDThin) {
		return nil
	}
	var V mat.Dense
	svd.VTo(&V) // d × min(n,d); columns are right singular vectors, descending
	rows, cols := V.Dims()
	if rows != d || cols < k {
		return nil
	}
	comps := make([][]float64, k)
	for j := 0; j < k; j++ {
		row := make([]float64, d)
		for i := 0; i < d; i++ {
			row[i] = V.At(i, j)
		}
		comps[j] = row
	}
	return &pcaBasis{components: comps, mean: mean, k: k}
}

// project maps a full-dim vector into the k-dim PCA space.
func (b *pcaBasis) project(x []float32) []float32 {
	if len(x) != len(b.mean) {
		return nil
	}
	z := make([]float32, b.k)
	for j := 0; j < b.k; j++ {
		comp := b.components[j]
		var s float64
		for i, val := range x {
			s += comp[i] * (float64(val) - b.mean[i])
		}
		z[j] = float32(s)
	}
	return z
}
