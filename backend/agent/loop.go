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
	cfg     EvaluatorConfig
	graph   *retrieval.Graph
	maxHops int
}

func NewAgentLoop(cfg EvaluatorConfig, graph *retrieval.Graph, maxHops int) *AgentLoop {
	return &AgentLoop{cfg: cfg, graph: graph, maxHops: maxHops}
}

// Run executes BFS hops + evaluator gate until sufficient or maxHops reached.
// Returns accumulated context chunks and metrics.
func (a *AgentLoop) Run(query string) ([]string, AgentMetrics) {
	var chunks []string
	metrics := AgentMetrics{}

	// Seed: find relevant nodes
	seed := a.graph.FindNodesByQuery(query)
	if len(seed) == 0 {
		return nil, metrics
	}

	visited := map[string]bool{}
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

		// Expand: BFS one more hop from all visited nodes
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

func nodeToChunk(n *retrieval.Node) string {
	chunk := fmt.Sprintf("[%s] %s", n.Type, n.Name)
	if year, ok := n.Props["year"]; ok {
		chunk += fmt.Sprintf(" (%v)", year)
	}
	return chunk
}
