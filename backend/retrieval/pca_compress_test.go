package retrieval

import (
	"math"
	"testing"
)

// makeVecs builds n deterministic d-dim vectors with structure (a few dominant
// directions + noise) so PCA has something to compress.
func makeVecs(n, d int) [][]float32 {
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, d)
		a := float64(i%7) - 3
		b := float64((i*3)%5) - 2
		for j := 0; j < d; j++ {
			noise := float64((i*31+j*17)%11-5) / 50.0
			v[j] = float32(a*math.Sin(float64(j)) + b*math.Cos(float64(j)) + noise)
		}
		vecs[i] = v
	}
	return vecs
}

func TestFitPCABasis_DimGuards(t *testing.T) {
	vecs := makeVecs(20, 64)
	if fitPCABasis(vecs, 0) != nil {
		t.Error("k=0 should disable compression (nil)")
	}
	if fitPCABasis(vecs, 64) != nil {
		t.Error("k>=d should be nil (no compression)")
	}
	b := fitPCABasis(vecs, 8)
	if b == nil || b.k != 8 || len(b.components) != 8 || len(b.mean) != 64 {
		t.Fatalf("unexpected basis: %+v", b)
	}
	if z := b.project(vecs[0]); len(z) != 8 {
		t.Errorf("project gave %d dims, want 8", len(z))
	}
}

func TestCompressedNodeIndex_SelfRetrieval(t *testing.T) {
	n, d := 40, 128
	vecs := makeVecs(n, d)
	ids := make([]string, n)
	for i := range ids {
		ids[i] = string(rune('A'+i%26)) + string(rune('0'+i/26))
	}
	ix := NewCompressedNodeIndex(ids, vecs, 16)
	if !ix.Compressed() || ix.Dim() != 16 {
		t.Fatalf("index not compressed: compressed=%v dim=%d", ix.Compressed(), ix.Dim())
	}
	// A full-dim query equal to node i's vector must be projected and retrieve
	// node i as the top neighbour.
	for _, i := range []int{0, 5, 17, 39} {
		got := ix.Nearest(vecs[i], 1)
		if len(got) != 1 || got[0] != ids[i] {
			t.Errorf("self-retrieval for node %d: got %v, want [%s]", i, got, ids[i])
		}
	}
	// A mismatched-dim query (already projected size) returns nil, not a panic.
	if ix.Nearest(make([]float32, 16), 1) != nil {
		t.Error("query of wrong (non-full) dim should return nil")
	}
}
