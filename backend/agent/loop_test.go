package agent

import (
	"strings"
	"testing"

	"github.com/navy1999/spectra-rag/backend/retrieval"
)

func TestNodeToChunk_RendersAbstract(t *testing.T) {
	g, err := retrieval.ParseGraph([]byte(`{"nodes":[
		{"id":"p1","type":"paper","name":"FlashAttention","props":{"year":2022,"abstract":"A fast exact attention algorithm."}},
		{"id":"a1","type":"author","name":"Tri Dao"}
	],"edges":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	paper, _ := g.Node("p1")
	chunk := nodeToChunk(paper)
	if !strings.Contains(chunk, "FlashAttention") || !strings.Contains(chunk, "(2022)") || !strings.Contains(chunk, "fast exact attention") {
		t.Errorf("paper chunk missing name/year/abstract: %q", chunk)
	}
	// A node without an abstract renders just its label.
	author, _ := g.Node("a1")
	if c := nodeToChunk(author); c != "[author] Tri Dao" {
		t.Errorf("author chunk = %q, want [author] Tri Dao", c)
	}
}

func TestSeedNodes_LexicalFallbackWhenNoIndex(t *testing.T) {
	g, err := retrieval.ParseGraph([]byte(`{"nodes":[
		{"id":"p1","type":"paper","name":"FlashAttention: Fast and Memory-Efficient Exact Attention"},
		{"id":"p2","type":"paper","name":"Attention Is All You Need"}
	],"edges":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	// nil node index + nil query embedding -> lexical seeding only, still works.
	loop := NewAgentLoop(EvaluatorConfig{}, g, nil, 3)
	seeds := loop.seedNodes("What is FlashAttention?", nil)
	if len(seeds) == 0 || seeds[0].ID != "p1" {
		t.Errorf("expected lexical seed p1 first, got %v", seeds)
	}
}
