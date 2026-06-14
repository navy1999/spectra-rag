package retrieval

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// NodeIndex holds precomputed node embeddings for semantic seed retrieval. It is
// optional: when absent the agent loop seeds lexically only. The embeddings are
// produced offline by scripts/embed_nodes.py with the SAME provider/model the
// backend uses at query time, so the query vector and node vectors share a space.
type NodeIndex struct {
	ids   []string
	vecs  [][]float32
	norms []float64
	dim   int
}

// LoadNodeIndex reads data/node_embeddings.json:
// {"model":..., "dim":N, "embeddings": {nodeID: [floats]}}.
func LoadNodeIndex(path string) (*NodeIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Dim        int                  `json:"dim"`
		Embeddings map[string][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse node embeddings: %w", err)
	}
	if len(doc.Embeddings) == 0 {
		return nil, fmt.Errorf("node embeddings file has no vectors")
	}
	ix := &NodeIndex{dim: doc.Dim}
	for id, v := range doc.Embeddings {
		if ix.dim == 0 {
			ix.dim = len(v)
		}
		ix.ids = append(ix.ids, id)
		ix.vecs = append(ix.vecs, v)
		ix.norms = append(ix.norms, l2norm(v))
	}
	return ix, nil
}

// Len reports the number of indexed nodes (0 for a nil index).
func (ix *NodeIndex) Len() int {
	if ix == nil {
		return 0
	}
	return len(ix.ids)
}

// Nearest returns up to k node IDs most cosine-similar to q, best first. Returns
// nil when the index is empty or the query dimension doesn't match (so callers
// fall back to lexical seeding rather than routing on a mismatched space).
func (ix *NodeIndex) Nearest(q []float32, k int) []string {
	if ix == nil || len(q) == 0 || (ix.dim != 0 && len(q) != ix.dim) {
		return nil
	}
	qn := l2norm(q)
	if qn == 0 {
		return nil
	}
	type scored struct {
		id  string
		sim float64
	}
	all := make([]scored, 0, len(ix.ids))
	for i, v := range ix.vecs {
		if len(v) != len(q) {
			continue
		}
		all = append(all, scored{ix.ids[i], dot(q, v) / (qn*ix.norms[i] + 1e-12)})
	}
	sort.Slice(all, func(a, b int) bool { return all[a].sim > all[b].sim })
	if k > len(all) {
		k = len(all)
	}
	out := make([]string, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].id
	}
	return out
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func l2norm(a []float32) float64 {
	var s float64
	for _, v := range a {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s)
}
