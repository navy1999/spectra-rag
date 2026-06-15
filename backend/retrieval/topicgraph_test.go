package retrieval

import "testing"

func TestBuildGraphFromPapers(t *testing.T) {
	papers := []ArxivPaper{
		{ID: "2501.01", Title: "Retrieval Augmented Transformers", Authors: []string{"Ada Lovelace", "Alan Turing"},
			Summary: "We use retrieval and attention.", Year: 2025, Categories: []string{"cs.CL"}},
		{ID: "2412.02", Title: "Sparse Mixture of Experts", Authors: []string{"Ada Lovelace"},
			Summary: "A mixture of experts model.", Year: 2024, Categories: []string{"cs.LG"}},
	}
	g := BuildGraphFromPapers(papers)

	nodes, edges, byType := g.Stats()
	if byType["paper"] != 2 {
		t.Errorf("want 2 papers, got %d", byType["paper"])
	}
	// Ada appears on both papers but must be a single deduped author node.
	if byType["author"] != 2 {
		t.Errorf("want 2 authors (deduped), got %d", byType["author"])
	}
	if byType["topic"] == 0 {
		t.Error("expected topic nodes from keywords/categories")
	}
	if nodes == 0 || edges == 0 {
		t.Error("empty graph")
	}

	// The paper node must carry its abstract for retrieval.
	pid := "p_2501_01"
	n, ok := g.Node(pid)
	if !ok {
		t.Fatalf("paper node %s missing", pid)
	}
	if abs, _ := n.Props["abstract"].(string); abs == "" {
		t.Error("paper node missing abstract prop")
	}

	// NodeTexts should yield one text per node, embeddable.
	ids, texts := g.NodeTexts()
	if len(ids) != nodes || len(texts) != nodes {
		t.Errorf("NodeTexts length mismatch: %d ids, %d texts, %d nodes", len(ids), len(texts), nodes)
	}
}

func TestBuildGraphFromPapers_SkipsDupesAndEmpty(t *testing.T) {
	papers := []ArxivPaper{
		{ID: "x1", Title: "Real", Authors: []string{"A"}, Summary: "s"},
		{ID: "x1", Title: "Duplicate id", Authors: []string{"B"}}, // dup id -> skipped
		{ID: "", Title: "No id"},                                  // skipped
		{ID: "x2", Title: ""},                                     // empty title -> skipped
	}
	g := BuildGraphFromPapers(papers)
	if _, _, byType := g.Stats(); byType["paper"] != 1 {
		t.Errorf("want 1 paper after dedupe/skip, got %d", byType["paper"])
	}
}
