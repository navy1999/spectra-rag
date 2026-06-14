package agent

import (
	"fmt"
	"strings"

	"github.com/navy1999/spectra-rag/backend/retrieval"
)

type AgentMetrics struct {
	HopsUsed int
	Verdict  Verdict
}

type AgentLoop struct {
	cfg       EvaluatorConfig
	graph     *retrieval.Graph
	nodeIndex *retrieval.NodeIndex // optional: enables semantic seed retrieval
	maxHops   int
}

func NewAgentLoop(cfg EvaluatorConfig, graph *retrieval.Graph, nodeIndex *retrieval.NodeIndex, maxHops int) *AgentLoop {
	return &AgentLoop{cfg: cfg, graph: graph, nodeIndex: nodeIndex, maxHops: maxHops}
}

// maxSeeds caps the seed set so context stays bounded now that chunks carry
// abstracts (rendering every lexical hit on a few-hundred-node graph would blow
// up the prompt).
const maxSeeds = 8

// Run seeds relevant nodes, then BFS-expands hop-by-hop gated by the evaluator,
// returning the accumulated context chunks. queryEmb is the query's embedding
// (already computed for routing); when a node index is present it drives
// semantic nearest-neighbor seeding, unioned with lexical keyword matches. A nil
// queryEmb or absent index degrades to lexical-only seeding.
func (a *AgentLoop) Run(query string, queryEmb []float32) ([]string, AgentMetrics) {
	metrics := AgentMetrics{}

	seed := a.seedNodes(query, queryEmb)
	if len(seed) == 0 {
		return nil, metrics
	}

	visited := map[string]bool{}
	var chunks []string
	for _, n := range seed {
		visited[n.ID] = true
		chunks = append(chunks, nodeToChunk(n))
	}

	for hop := 0; hop < a.maxHops; hop++ {
		metrics.HopsUsed = hop + 1
		context := strings.Join(chunks, "\n\n")

		verdict, err := RunEvaluator(a.cfg, query, context)
		if err != nil {
			verdict = VerdictInsufficient
		}
		metrics.Verdict = verdict
		if verdict == VerdictSufficient {
			break
		}

		// Expand: BFS one more hop from all visited nodes.
		var newChunks []string
		for id := range visited {
			for _, neighbor := range a.graph.GetNeighbors(id) {
				if !visited[neighbor.ID] {
					visited[neighbor.ID] = true
					newChunks = append(newChunks, nodeToChunk(neighbor))
				}
			}
		}
		if len(newChunks) == 0 {
			break
		}
		chunks = append(chunks, newChunks...)
	}

	return chunks, metrics
}

// seedNodes merges semantic (cosine-NN on the query embedding) and lexical
// (keyword-scored) seeds, deduped and capped. Semantic seeds go first because
// they are more precise on a larger graph where queries use different words than
// node names; lexical seeds are always included so retrieval still works with no
// embeddings or no node index.
func (a *AgentLoop) seedNodes(query string, queryEmb []float32) []*retrieval.Node {
	seen := map[string]bool{}
	var seeds []*retrieval.Node
	add := func(n *retrieval.Node, ok bool) {
		if ok && n != nil && !seen[n.ID] && len(seeds) < maxSeeds {
			seen[n.ID] = true
			seeds = append(seeds, n)
		}
	}

	if a.nodeIndex.Len() > 0 && len(queryEmb) > 0 {
		for _, id := range a.nodeIndex.Nearest(queryEmb, maxSeeds) {
			add(a.graph.Node(id))
		}
	}
	for _, n := range a.graph.FindNodesByQuery(query) {
		add(n, true)
	}
	return seeds
}

func nodeToChunk(n *retrieval.Node) string {
	chunk := fmt.Sprintf("[%s] %s", n.Type, n.Name)
	if year, ok := n.Props["year"]; ok {
		chunk += fmt.Sprintf(" (%v)", year)
	}
	if abs, ok := n.Props["abstract"].(string); ok && abs != "" {
		chunk += ": " + abs
	}
	return chunk
}
