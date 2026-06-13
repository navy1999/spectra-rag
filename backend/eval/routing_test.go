package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/router"
)

func mustRouter(t *testing.T) *router.PCARouter {
	t.Helper()
	r, err := router.NewPCARouter("") // missing path -> hardcoded default centroids
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestLengthRoute(t *testing.T) {
	if LengthRoute("What is BERT?", 8) != "chat" {
		t.Error("short query should route chat")
	}
	if LengthRoute("Which institutions are connected to the authors of FlashAttention today?", 8) != "agentic" {
		t.Error("long query should route agentic")
	}
}

func TestHitCountRoute(t *testing.T) {
	path := filepath.Join(t.TempDir(), "graph.json")
	os.WriteFile(path, []byte(`{"nodes":[
		{"id":"p5","type":"paper","name":"FlashAttention: Fast and Memory-Efficient Exact Attention"},
		{"id":"p1","type":"paper","name":"Attention Is All You Need"},
		{"id":"t1","type":"topic","name":"Transformer Architecture"}
	],"edges":[]}`), 0o644)
	g, err := retrieval.LoadGraph(path)
	if err != nil {
		t.Fatal(err)
	}
	// Single entity -> chat.
	if HitCountRoute("What is FlashAttention?", g, 2) != "chat" {
		t.Error("single-entity query should route chat")
	}
	// Spans multiple entities -> agentic.
	if HitCountRoute("How does the Transformer relate to FlashAttention and Attention?", g, 2) != "agentic" {
		t.Error("multi-entity query should route agentic")
	}
}

func TestRunRouting_Baselines(t *testing.T) {
	// No embedder needed for the non-PCA routers; verify always_* accuracy equals
	// the label balance on a tiny set.
	path := filepath.Join(t.TempDir(), "g.json")
	os.WriteFile(path, []byte(`{"nodes":[{"id":"n","type":"paper","name":"X"}],"edges":[]}`), 0o644)
	g, _ := retrieval.LoadGraph(path)
	ds := &RoutingDataset{Questions: []RoutingQuestion{
		{ID: "a", Text: "short one", Path: "chat"},
		{ID: "b", Text: "another short", Path: "agentic"},
	}}
	// Use a nil embedder but the PCA router won't be exercised because we only
	// read the always_* results here.
	results, err := RunRouting(ds, retrieval.NewEmbedder("", "", ""), mustRouter(t), g, 8, 2)
	if err != nil {
		t.Fatalf("RunRouting: %v", err)
	}
	got := map[string]RoutingResult{}
	for _, r := range results {
		got[r.Router] = r
	}
	if got["always_agentic"].Accuracy != 0.5 || got["always_chat"].Accuracy != 0.5 {
		t.Errorf("always_* accuracy should equal label balance 0.5, got %v / %v",
			got["always_agentic"].Accuracy, got["always_chat"].Accuracy)
	}
	if got["always_agentic"].AgenticRate != 1.0 || got["always_chat"].AgenticRate != 0.0 {
		t.Errorf("always_* agentic-rate wrong: %v / %v", got["always_agentic"].AgenticRate, got["always_chat"].AgenticRate)
	}
}
