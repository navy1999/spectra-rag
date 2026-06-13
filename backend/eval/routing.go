package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/router"
)

// Routing evaluation (A1): does the PCA router's chat-vs-agentic decision agree
// with human-judged labels better than trivial heuristics? Routing needs no chat
// LLM — only one embedding per question for the PCA path — so it is cheap and
// judge-free. The PCA path is only meaningful with real embeddings + a fitted
// model loaded; otherwise it routes on the dev sketch and the numbers are noise.

type RoutingQuestion struct {
	ID   string `json:"id"`
	Text string `json:"text"`
	Path string `json:"path"` // "chat" | "agentic"
}

type RoutingDataset struct {
	Questions []RoutingQuestion `json:"questions"`
}

func LoadRoutingDataset(path string) (*RoutingDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read routing questions: %w", err)
	}
	var d RoutingDataset
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("parse routing questions: %w", err)
	}
	if len(d.Questions) == 0 {
		return nil, fmt.Errorf("routing set %q is empty", path)
	}
	return &d, nil
}

// LengthRoute: a one-line baseline — longer queries tend to be multi-hop.
func LengthRoute(query string, threshold int) string {
	if len(strings.Fields(query)) > threshold {
		return "agentic"
	}
	return "chat"
}

// HitCountRoute: route agentic when the query keyword-matches several graph
// nodes (i.e. it spans multiple entities, likely needing traversal).
func HitCountRoute(query string, g *retrieval.Graph, threshold int) string {
	if len(g.FindNodesByQuery(query)) >= threshold {
		return "agentic"
	}
	return "chat"
}

type RoutingResult struct {
	Router      string
	Accuracy    float64 // fraction of questions whose route matched the label
	AgenticRate float64 // fraction routed agentic (cost proxy: agentic = retrieval + votes)
}

// RunRouting evaluates the PCA router against the baselines over the dataset.
// The embedder/pcaRouter are used only for the PCA path; embErr surfaces if
// embeddings fail (e.g. no key) so the caller can flag that the PCA row is not
// meaningful.
func RunRouting(ds *RoutingDataset, emb *retrieval.Embedder, pcaRouter *router.PCARouter, g *retrieval.Graph, lenThreshold, hitThreshold int) ([]RoutingResult, error) {
	type router2 struct {
		name   string
		decide func(q string) (string, error)
	}
	routers := []router2{
		{"pca", func(q string) (string, error) {
			v, err := emb.Embed(q)
			if err != nil {
				return "", err
			}
			d, _ := pcaRouter.Route(v)
			return string(d.Path), nil
		}},
		{"length", func(q string) (string, error) { return LengthRoute(q, lenThreshold), nil }},
		{"hit_count", func(q string) (string, error) { return HitCountRoute(q, g, hitThreshold), nil }},
		{"always_agentic", func(q string) (string, error) { return "agentic", nil }},
		{"always_chat", func(q string) (string, error) { return "chat", nil }},
	}

	n := len(ds.Questions)
	results := make([]RoutingResult, 0, len(routers))
	for _, rt := range routers {
		correct, agentic := 0, 0
		for _, q := range ds.Questions {
			got, err := rt.decide(q.Text)
			if err != nil {
				return nil, fmt.Errorf("%s router on %s: %w", rt.name, q.ID, err)
			}
			if got == q.Path {
				correct++
			}
			if got == "agentic" {
				agentic++
			}
		}
		results = append(results, RoutingResult{
			Router:      rt.name,
			Accuracy:    float64(correct) / float64(n),
			AgenticRate: float64(agentic) / float64(n),
		})
	}
	return results, nil
}

// RoutingReport renders the comparison as markdown.
func RoutingReport(results []RoutingResult, ds *RoutingDataset, pcaReal bool) string {
	agenticLabels := 0
	for _, q := range ds.Questions {
		if q.Path == "agentic" {
			agenticLabels++
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "## A1 routing evaluation\n\n")
	fmt.Fprintf(&b, "%d questions · %d labeled agentic / %d chat. Routing accuracy = agreement with the label; agentic-rate is a cost proxy (agentic triggers retrieval + the vote ensemble).\n\n",
		len(ds.Questions), agenticLabels, len(ds.Questions)-agenticLabels)
	if !pcaReal {
		fmt.Fprintf(&b, "> WARNING: no real embeddings (no EMBEDDINGS_API_KEY or dimension mismatch) — the `pca` row routed on the dev sketch and is NOT meaningful.\n\n")
	}
	fmt.Fprintf(&b, "| Router | Routing accuracy | Agentic-rate |\n|---|---|---|\n")
	for _, r := range results {
		fmt.Fprintf(&b, "| %s | %.0f%% | %.0f%% |\n", r.Router, r.Accuracy*100, r.AgenticRate*100)
	}
	fmt.Fprintf(&b, "\nA router earns its complexity only if it beats `length`/`hit_count` and the `always_*` baselines. If a one-liner ties `pca`, that is a real finding (simplify).\n")
	return b.String()
}
