package retrieval

import (
	"os"
	"path/filepath"
	"testing"
)

func testGraph(t testing.TB) *Graph {
	t.Helper()
	path := filepath.Join(t.TempDir(), "graph.json")
	content := `{"nodes":[
		{"id":"p5","type":"paper","name":"FlashAttention: Fast and Memory-Efficient Exact Attention","props":{"year":2022}},
		{"id":"p1","type":"paper","name":"Attention Is All You Need","props":{"year":2017}},
		{"id":"a5","type":"author","name":"Tri Dao"},
		{"id":"t4","type":"topic","name":"Efficient Attention"}
	],"edges":[
		{"from":"a5","to":"p5","rel":"authored"},
		{"from":"p5","to":"t4","rel":"introduces"},
		{"from":"p5","to":"p1","rel":"cites"}
	]}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write graph: %v", err)
	}
	g, err := LoadGraph(path)
	if err != nil {
		t.Fatalf("LoadGraph: %v", err)
	}
	return g
}

func ids(nodes []*Node) map[string]bool {
	m := map[string]bool{}
	for _, n := range nodes {
		m[n.ID] = true
	}
	return m
}

// TestFindNodesByQuery_NaturalLanguage is the regression test for the Tier 0
// fix: a natural-language question is not a substring of any node name, so the
// old strings.Contains seeded nothing. Keyword matching must still find the
// FlashAttention paper.
func TestFindNodesByQuery_NaturalLanguage(t *testing.T) {
	g := testGraph(t)
	got := g.FindNodesByQuery("What is FlashAttention?")
	if len(got) == 0 {
		t.Fatal("expected a seed match for natural-language query, got none")
	}
	if got[0].ID != "p5" {
		t.Errorf("expected top match p5 (FlashAttention), got %s", got[0].ID)
	}
}

func TestFindNodesByQuery_Ranking(t *testing.T) {
	g := testGraph(t)
	got := ids(g.FindNodesByQuery("efficient attention"))
	for _, want := range []string{"p5", "t4", "p1"} {
		if !got[want] {
			t.Errorf("expected %s in results for 'efficient attention'", want)
		}
	}
}

func TestFindNodesByQuery_NoMatch(t *testing.T) {
	g := testGraph(t)
	if got := g.FindNodesByQuery("quantum chromodynamics lattice"); len(got) != 0 {
		t.Errorf("expected no matches, got %d", len(got))
	}
}

func TestFindNodesByQuery_StopwordsOnly(t *testing.T) {
	g := testGraph(t)
	if got := g.FindNodesByQuery("what is the of and"); len(got) != 0 {
		t.Errorf("expected no matches for stopword-only query, got %d", len(got))
	}
}

func TestGetNeighbors(t *testing.T) {
	g := testGraph(t)
	got := ids(g.GetNeighbors("p5"))
	for _, want := range []string{"a5", "t4", "p1"} {
		if !got[want] {
			t.Errorf("expected p5 neighbor %s", want)
		}
	}
}

func TestBFS(t *testing.T) {
	g := testGraph(t)
	got := ids(g.BFS("a5", 2))
	// a5 -> p5 (hop 1) -> {t4, p1} (hop 2)
	for _, want := range []string{"p5", "t4", "p1"} {
		if !got[want] {
			t.Errorf("expected BFS to reach %s within 2 hops", want)
		}
	}
}

func TestLoadGraph_MissingFileIsEmpty(t *testing.T) {
	g, err := LoadGraph(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("missing file should not error, got %v", err)
	}
	if g.NodeCount() != 0 {
		t.Errorf("expected empty graph, got %d nodes", g.NodeCount())
	}
}

func TestSignificantTokens(t *testing.T) {
	got := significantTokens("What is the FlashAttention model?")
	want := map[string]bool{"flashattention": true, "model": true}
	if len(got) != len(want) {
		t.Fatalf("got %v, want keys %v", got, want)
	}
	for _, tok := range got {
		if !want[tok] {
			t.Errorf("unexpected token %q", tok)
		}
	}
}

func BenchmarkFindNodesByQuery(b *testing.B) {
	g := testGraph(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.FindNodesByQuery("How does efficient attention work in FlashAttention?")
	}
}

func BenchmarkBFS(b *testing.B) {
	g := testGraph(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = g.BFS("a5", 3)
	}
}
