package eval

import (
	"fmt"

	"github.com/navy1999/spectra-rag/backend/retrieval"
)

// retrieveContext reproduces the framework's retrieval (keyword seed + BFS hop
// expansion) without the vote evaluator, so both RAG conditions see the same
// context and the comparison isolates the synthesis penalty (A3) and trie guard
// (A2) rather than retrieval variance. Returns the chunk strings in the same
// "[type] name (year)" form the agent loop produces.
// RetrieveContext is the exported form of the framework retrieval, used by the
// pairwise judge harness so ON/OFF answers and the judge all see identical context.
func RetrieveContext(g *retrieval.Graph, query string, maxHops int) []string {
	return retrieveContext(g, query, maxHops)
}

func retrieveContext(g *retrieval.Graph, query string, maxHops int) []string {
	seed := g.FindNodesByQuery(query)
	if len(seed) == 0 {
		return nil
	}
	visited := make(map[string]bool)
	var chunks []string
	frontier := make([]*retrieval.Node, 0, len(seed))
	for _, n := range seed {
		if !visited[n.ID] {
			visited[n.ID] = true
			chunks = append(chunks, chunkOf(n))
			frontier = append(frontier, n)
		}
	}
	for h := 0; h < maxHops; h++ {
		var next []*retrieval.Node
		for _, n := range frontier {
			for _, nb := range g.GetNeighbors(n.ID) {
				if !visited[nb.ID] {
					visited[nb.ID] = true
					chunks = append(chunks, chunkOf(nb))
					next = append(next, nb)
				}
			}
		}
		if len(next) == 0 {
			break
		}
		frontier = next
	}
	return chunks
}

func chunkOf(n *retrieval.Node) string {
	c := fmt.Sprintf("[%s] %s", n.Type, n.Name)
	if year, ok := n.Props["year"]; ok {
		c += fmt.Sprintf(" (%v)", year)
	}
	if abs, ok := n.Props["abstract"].(string); ok && abs != "" {
		c += ": " + abs
	}
	return c
}
