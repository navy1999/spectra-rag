package retrieval

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

type NodeType string

const (
	NodePaper       NodeType = "paper"
	NodeAuthor      NodeType = "author"
	NodeTopic       NodeType = "topic"
	NodeInstitution NodeType = "institution"
)

type Node struct {
	ID    string                 `json:"id"`
	Type  NodeType               `json:"type"`
	Name  string                 `json:"name"`
	Props map[string]interface{} `json:"props,omitempty"`
}

type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Rel  string `json:"rel"`
}

type Graph struct {
	nodes map[string]*Node
	edges []Edge
	adj   map[string][]string // node ID -> neighbor IDs
}

type graphDoc struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// LoadGraph reads a graph from disk. A missing file is tolerated (returns an
// empty graph) so the server still boots with the hardcoded fallbacks; a present
// but malformed/invalid file is a hard error.
func LoadGraph(path string) (*Graph, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return emptyGraph(), nil
	}
	return ParseGraph(data)
}

// ParseGraph parses and validates a graph JSON document. Unlike LoadGraph it is
// strict — malformed JSON or a structurally invalid graph returns an error. This
// is the entry point for user-supplied graphs via the /ingest endpoint.
func ParseGraph(data []byte) (*Graph, error) {
	var raw graphDoc
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse graph: %w", err)
	}
	if err := validateGraphData(raw.Nodes, raw.Edges); err != nil {
		return nil, err
	}
	return buildGraph(raw.Nodes, raw.Edges), nil
}

// validateGraphData enforces the structural invariants every graph must satisfy:
// at least one node, unique non-empty ids, non-empty names, and edges that only
// reference declared nodes. Node `type` is intentionally freeform so users can
// model their own domains.
func validateGraphData(nodes []Node, edges []Edge) error {
	if len(nodes) == 0 {
		return fmt.Errorf("graph has no nodes")
	}
	seen := make(map[string]bool, len(nodes))
	for i, n := range nodes {
		if n.ID == "" {
			return fmt.Errorf("node[%d] has an empty id", i)
		}
		if seen[n.ID] {
			return fmt.Errorf("duplicate node id %q", n.ID)
		}
		seen[n.ID] = true
		if n.Name == "" {
			return fmt.Errorf("node %q has an empty name", n.ID)
		}
	}
	for i, e := range edges {
		if !seen[e.From] {
			return fmt.Errorf("edge[%d] references unknown node %q (from)", i, e.From)
		}
		if !seen[e.To] {
			return fmt.Errorf("edge[%d] references unknown node %q (to)", i, e.To)
		}
	}
	return nil
}

func buildGraph(nodes []Node, edges []Edge) *Graph {
	g := &Graph{
		nodes: make(map[string]*Node, len(nodes)),
		edges: edges,
		adj:   make(map[string][]string),
	}
	for i := range nodes {
		g.nodes[nodes[i].ID] = &nodes[i]
	}
	for _, e := range edges {
		g.adj[e.From] = append(g.adj[e.From], e.To)
		g.adj[e.To] = append(g.adj[e.To], e.From)
	}
	return g
}

func emptyGraph() *Graph {
	return &Graph{nodes: make(map[string]*Node), adj: make(map[string][]string)}
}

// Stats returns node/edge counts plus a per-type node breakdown, for the
// /graph inspection endpoint.
func (g *Graph) Stats() (nodes int, edges int, byType map[string]int) {
	byType = map[string]int{}
	for _, n := range g.nodes {
		byType[string(n.Type)]++
	}
	return len(g.nodes), len(g.edges), byType
}

func (g *Graph) NodeCount() int { return len(g.nodes) }

func (g *Graph) AllNodeNames() []string {
	names := make([]string, 0, len(g.nodes))
	for _, n := range g.nodes {
		names = append(names, n.Name)
	}
	return names
}

func (g *Graph) GetNeighbors(id string) []*Node {
	var out []*Node
	for _, nid := range g.adj[id] {
		if n, ok := g.nodes[nid]; ok {
			out = append(out, n)
		}
	}
	return out
}

// BFS returns all nodes reachable within maxHops from startID.
func (g *Graph) BFS(startID string, maxHops int) []*Node {
	visited := map[string]bool{startID: true}
	queue := []string{startID}
	var result []*Node
	depth := 0

	for len(queue) > 0 && depth < maxHops {
		next := []string{}
		for _, id := range queue {
			for _, nid := range g.adj[id] {
				if !visited[nid] {
					visited[nid] = true
					next = append(next, nid)
					if n, ok := g.nodes[nid]; ok {
						result = append(result, n)
					}
				}
			}
		}
		queue = next
		depth++
	}
	return result
}

// FindNodesByQuery returns nodes whose name overlaps with the query's
// significant keywords, ranked by overlap strength (best match first).
//
// A naive strings.Contains(name, query) fails for natural-language questions:
// "What is FlashAttention?" is not a substring of any node name, so it would
// return nothing and starve the agent loop of seed context. Instead we tokenize
// the query, drop stopwords, and score each node by (a) exact token matches
// against the node's own name tokens and (b) keyword-substring hits, returning
// every node with a non-zero score sorted by score.
func (g *Graph) FindNodesByQuery(q string) []*Node {
	qTokens := significantTokens(q)
	if len(qTokens) == 0 {
		return nil
	}
	qSet := make(map[string]bool, len(qTokens))
	for _, t := range qTokens {
		qSet[t] = true
	}

	type scored struct {
		node  *Node
		score int
	}
	var matches []scored
	for _, n := range g.nodes {
		nameLower := strings.ToLower(n.Name)
		score := 0
		for _, nt := range significantTokens(n.Name) {
			if qSet[nt] {
				score += 2 // whole-token match is the strongest signal
			}
		}
		for _, qt := range qTokens {
			if len(qt) >= 4 && strings.Contains(nameLower, qt) {
				score++ // keyword appears inside the name (e.g. "attention" in "FlashAttention")
			}
		}
		if score > 0 {
			matches = append(matches, scored{n, score})
		}
	}

	sort.SliceStable(matches, func(i, j int) bool {
		return matches[i].score > matches[j].score
	})
	out := make([]*Node, len(matches))
	for i, m := range matches {
		out[i] = m.node
	}
	return out
}

// stopwords are common question/filler words stripped before keyword matching.
var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "is": true, "are": true, "was": true,
	"were": true, "what": true, "which": true, "who": true, "whom": true,
	"how": true, "why": true, "when": true, "where": true, "does": true,
	"do": true, "did": true, "of": true, "in": true, "on": true, "to": true,
	"for": true, "and": true, "or": true, "with": true, "about": true,
	"tell": true, "me": true, "explain": true, "describe": true, "can": true,
	"you": true, "this": true, "that": true, "it": true, "as": true, "by": true,
	"from": true, "into": true, "at": true, "be": true, "i": true, "give": true,
}

// significantTokens lowercases s, splits on non-alphanumeric runs, and drops
// stopwords and 1-character tokens.
func significantTokens(s string) []string {
	var out []string
	for _, t := range tokenizeWords(strings.ToLower(s)) {
		if len(t) < 2 || stopwords[t] {
			continue
		}
		out = append(out, t)
	}
	return out
}

func tokenizeWords(s string) []string {
	var tokens []string
	var buf strings.Builder
	for _, c := range s {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			buf.WriteRune(c)
		} else if buf.Len() > 0 {
			tokens = append(tokens, buf.String())
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		tokens = append(tokens, buf.String())
	}
	return tokens
}
