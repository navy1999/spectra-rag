// Command routeeval runs the A1 routing evaluation: it compares the PCA router's
// chat-vs-agentic decisions against trivial heuristics over a labeled set. Needs
// a real embeddings key (EMBEDDINGS_API_KEY) and a fitted pca_model.json for the
// PCA row to be meaningful; the heuristic rows work regardless.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	pcacgo "github.com/navy1999/spectra-rag/backend/cgo"
	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/eval"
	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/router"
)

func resolveData(name string) string {
	for _, p := range []string{filepath.Join("data", name), filepath.Join("..", "data", name)} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return filepath.Join("data", name)
}

func main() {
	questionsPath := flag.String("questions", "", "routing question set (default data/routing_questions.json)")
	lenThreshold := flag.Int("length-threshold", 8, "length-baseline: words above which a query routes agentic")
	hitThreshold := flag.Int("hits-threshold", 2, "hit-count baseline: graph keyword hits at/above which a query routes agentic")
	out := flag.String("out", "", "markdown report output path (default data/routing_results.md)")
	flag.Parse()

	cfg := config.Load()

	graph, err := retrieval.LoadGraph(cfg.GraphPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load graph: %v\n", err)
		os.Exit(1)
	}
	if err := pcacgo.LoadModel(cfg.PCAModelPath); err != nil {
		fmt.Fprintf(os.Stderr, "[routeeval] PCA model not loaded (%v) — PCA row will use the dev sketch\n", err)
	}
	emb := retrieval.NewEmbedderWithTask(cfg.EmbeddingsAPIKey, cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel, cfg.EmbeddingsTask)
	pcaReal := !emb.Mock()
	if !pcaReal {
		fmt.Fprintln(os.Stderr, "[routeeval] WARNING: no EMBEDDINGS_API_KEY — the pca row is NOT meaningful (mock embeddings). Set it to evaluate the real router.")
	}
	pcaRouter, err := router.NewPCARouter(cfg.PCACentroidsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build router: %v\n", err)
		os.Exit(1)
	}

	qPath := *questionsPath
	if qPath == "" {
		qPath = resolveData("routing_questions.json")
	}
	ds, err := eval.LoadRoutingDataset(qPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dataset: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "[routeeval] %d questions · embeddings=%s · pcaReal=%v\n", len(ds.Questions), cfg.EmbeddingsModel, pcaReal)
	// Label the learned-router row by the fitter that produced the loaded model
	// (e.g. "pca" or "pca16_lda"), so the report reflects the real projection.
	routerName := pcacgo.LoadedMethod()
	results, err := eval.RunRouting(ds, emb, pcaRouter, graph, *lenThreshold, *hitThreshold, routerName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run routing eval: %v\n", err)
		os.Exit(1)
	}

	report := eval.RoutingReport(results, ds, pcaReal)
	fmt.Println()
	fmt.Println(report)

	outPath := *out
	if outPath == "" {
		outPath = filepath.Join(filepath.Dir(qPath), "routing_results.md")
	}
	if err := os.WriteFile(outPath, []byte(report), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[routeeval] report written to %s\n", outPath)
	}
}
