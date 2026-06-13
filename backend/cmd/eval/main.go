// Command eval runs the Phase 1 evaluation: one small model under three
// conditions (raw / plain RAG / full spectra pipeline) over a graph-grounded
// question set, scored with judge-free metrics. Requires OPENROUTER_API_KEY.
//
// Usage (from repo root or backend/):
//
//	OPENROUTER_API_KEY=sk-or-... go run ./cmd/eval
//	OPENROUTER_API_KEY=sk-or-... go run ./cmd/eval -limit 5 -model meta-llama/llama-3.2-3b-instruct:free
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/eval"
	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/trie"
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
	questionsPath := flag.String("questions", "", "path to question set JSON (default: data/eval_questions.json)")
	modelOverride := flag.String("model", "", "model id override (default: from dataset)")
	hops := flag.Int("hops", 1, "BFS hops for retrieval")
	temp := flag.Float64("temp", 0.3, "sampling temperature (fixed across conditions for fairness)")
	maxTokens := flag.Int("max-tokens", 400, "max tokens per answer")
	limit := flag.Int("limit", 0, "evaluate only the first N questions (0 = all)")
	out := flag.String("out", "", "markdown report output path (default: data/eval_results.md)")
	cachePath := flag.String("cache", "eval_cache.json", "LLM response cache path")
	flag.Parse()

	cfg := config.Load()
	if cfg.OpenRouterAPIKey == "" {
		fmt.Fprintln(os.Stderr, "eval needs a live model. Set OPENROUTER_API_KEY and re-run:")
		fmt.Fprintln(os.Stderr, "  OPENROUTER_API_KEY=sk-or-... go run ./cmd/eval")
		os.Exit(1)
	}

	qPath := *questionsPath
	if qPath == "" {
		qPath = resolveData("eval_questions.json")
	}
	ds, err := eval.LoadDataset(qPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load dataset: %v\n", err)
		os.Exit(1)
	}

	model := *modelOverride
	if model == "" {
		model = ds.Model
	}
	if model == "" {
		model = cfg.DefaultModel
	}

	graph, err := retrieval.LoadGraph(cfg.GraphPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load graph: %v\n", err)
		os.Exit(1)
	}
	t := trie.New()
	for _, name := range graph.AllNodeNames() {
		t.Insert(name)
	}

	questions := ds.Questions
	if *limit > 0 && *limit < len(questions) {
		questions = questions[:*limit]
	}

	fmt.Fprintf(os.Stderr, "[eval] model=%s questions=%d graph=%d nodes cache=%s\n", model, len(questions), graph.NodeCount(), *cachePath)

	client := eval.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL, model, *cachePath)
	runner := eval.NewRunner(graph, t, client, *hops, *temp, *maxTokens)
	results, aggs := runner.RunAll(questions)

	report := eval.Report(model, *temp, *hops, len(questions), aggs)
	fmt.Println()
	fmt.Println(report)

	outPath := *out
	if outPath == "" {
		outPath = resolveData("eval_results.md")
	}
	if err := os.WriteFile(outPath, []byte(report), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[eval] report written to %s\n", outPath)
	}

	rawPath := filepath.Join(filepath.Dir(outPath), "eval_results.json")
	if data, err := json.MarshalIndent(results, "", "  "); err == nil {
		_ = os.WriteFile(rawPath, data, 0o644)
		fmt.Fprintf(os.Stderr, "[eval] raw answers written to %s\n", rawPath)
	}
}
