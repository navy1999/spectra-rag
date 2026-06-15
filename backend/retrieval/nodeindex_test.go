package retrieval

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNodeIndex_Nearest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "node_embeddings.json")
	// Three orthogonal-ish 3-d vectors.
	body := `{"dim":3,"embeddings":{
		"a":[1.0,0.0,0.0],
		"b":[0.0,1.0,0.0],
		"c":[0.9,0.1,0.0]
	}}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	ix, err := LoadNodeIndex(path)
	if err != nil {
		t.Fatalf("LoadNodeIndex: %v", err)
	}
	if ix.Len() != 3 {
		t.Fatalf("Len = %d, want 3", ix.Len())
	}

	// A query along x should rank "a" then "c" (both x-ish) above "b".
	got := ix.Nearest([]float32{1, 0, 0}, 2)
	if len(got) != 2 || got[0] != "a" {
		t.Fatalf("Nearest = %v, want [a c]", got)
	}
	if got[1] != "c" {
		t.Errorf("second nearest = %q, want c", got[1])
	}
}

func TestNodeIndex_DimMismatchFallsBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "n.json")
	os.WriteFile(path, []byte(`{"dim":3,"embeddings":{"a":[1,0,0]}}`), 0o644)
	ix, err := LoadNodeIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	// Wrong-dimension query -> nil (caller uses lexical seeding instead).
	if got := ix.Nearest([]float32{1, 0}, 1); got != nil {
		t.Errorf("dim mismatch should return nil, got %v", got)
	}
}

func TestNodeIndex_NilSafe(t *testing.T) {
	var ix *NodeIndex
	if ix.Len() != 0 || ix.Nearest([]float32{1}, 1) != nil {
		t.Error("nil NodeIndex should be safe")
	}
}
