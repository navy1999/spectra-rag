package retrieval

import "testing"

func TestParseGraph_Valid(t *testing.T) {
	data := []byte(`{"nodes":[{"id":"n1","type":"paper","name":"A"},{"id":"n2","type":"topic","name":"B"}],"edges":[{"from":"n1","to":"n2","rel":"about"}]}`)
	g, err := ParseGraph(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if g.NodeCount() != 2 {
		t.Errorf("nodes = %d, want 2", g.NodeCount())
	}
}

func TestParseGraph_Invalid(t *testing.T) {
	cases := map[string]string{
		"malformed json": `{not json`,
		"no nodes":       `{"nodes":[],"edges":[]}`,
		"empty id":       `{"nodes":[{"id":"","name":"A"}]}`,
		"empty name":     `{"nodes":[{"id":"n1","name":""}]}`,
		"duplicate id":   `{"nodes":[{"id":"n1","name":"A"},{"id":"n1","name":"B"}]}`,
		"dangling edge":  `{"nodes":[{"id":"n1","name":"A"}],"edges":[{"from":"n1","to":"ghost"}]}`,
	}
	for name, doc := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseGraph([]byte(doc)); err == nil {
				t.Errorf("expected an error for %q", name)
			}
		})
	}
}

// TestParseGraph_FreeformType confirms node types are not restricted to the
// built-in four, so users can model their own domains.
func TestParseGraph_FreeformType(t *testing.T) {
	data := []byte(`{"nodes":[{"id":"n1","type":"recipe","name":"Carbonara"}],"edges":[]}`)
	if _, err := ParseGraph(data); err != nil {
		t.Errorf("freeform node type should be allowed, got %v", err)
	}
}

func TestStats(t *testing.T) {
	g := testGraph(t) // 2 papers, 1 author, 1 topic; 3 edges
	nodes, edges, byType := g.Stats()
	if nodes != 4 {
		t.Errorf("nodes = %d, want 4", nodes)
	}
	if edges != 3 {
		t.Errorf("edges = %d, want 3", edges)
	}
	if byType["paper"] != 2 {
		t.Errorf("paper count = %d, want 2", byType["paper"])
	}
}
