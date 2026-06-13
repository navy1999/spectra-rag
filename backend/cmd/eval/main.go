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
	"strings"

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

// fallbackModels are preflighted (in order) when no model is pinned and the
// dataset model is unavailable. Free rosters churn, so this is best-effort and
// overridable with -model / -models.
var fallbackModels = []string{
	"openai/gpt-oss-20b:free",
	"nvidia/nemotron-nano-9b-v2:free",
	"google/gemma-4-31b-it:free",
	"openai/gpt-oss-120b:free",
}

func buildCandidates(override, list, datasetModel, cfgModel string) []string {
	var raw []string
	switch {
	case override != "":
		raw = []string{override}
	case list != "":
		for _, m := range strings.Split(list, ",") {
			if m = strings.TrimSpace(m); m != "" {
				raw = append(raw, m)
			}
		}
	default:
		if datasetModel != "" {
			raw = append(raw, datasetModel)
		} else if cfgModel != "" {
			raw = append(raw, cfgModel)
		}
		raw = append(raw, fallbackModels...)
	}
	seen := map[string]bool{}
	var out []string
	for _, m := range raw {
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	return out
}

// selectModel preflights candidates in order and returns the first that serves.
func selectModel(cfg *config.Config, candidates []string, cachePath string) (*eval.Client, string) {
	for _, m := range candidates {
		c := eval.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL, m, cachePath)
		fmt.Fprintf(os.Stderr, "[eval] preflight %s ... ", m)
		if err := c.Preflight(); err != nil {
			fmt.Fprintf(os.Stderr, "unavailable (%v)\n", err)
			continue
		}
		fmt.Fprintln(os.Stderr, "ok")
		return c, m
	}
	return nil, ""
}

func main() {
	questionsPath := flag.String("questions", "", "path to question set JSON (default: data/eval_questions.json)")
	modelOverride := flag.String("model", "", "pin a single model id (skips fallback)")
	modelList := flag.String("models", "", "comma-separated model fallback list; first one serving is used")
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
		fmt.Fprintln(os.Stderr, "  PowerShell:  $env:OPENROUTER_API_KEY=\"sk-or-...\"; go run ./cmd/eval")
		fmt.Fprintln(os.Stderr, "  bash:        OPENROUTER_API_KEY=sk-or-... go run ./cmd/eval")
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

	// Pick a model that is actually serving. Free models are frequently
	// rate-limited upstream, so we preflight an ordered candidate list and use
	// the first that responds. An explicit -model/-models pins the candidates.
	candidates := buildCandidates(*modelOverride, *modelList, ds.Model, cfg.DefaultModel)
	client, model := selectModel(cfg, candidates, *cachePath)
	if client == nil {
		fmt.Fprintf(os.Stderr, "no candidate model is currently serving (tried: %s).\nRetry later or pass -model <id>.\n", strings.Join(candidates, ", "))
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "[eval] model=%s questions=%d graph=%d nodes cache=%s\n", model, len(questions), graph.NodeCount(), *cachePath)

	runner := eval.NewRunner(graph, t, client, *hops, *temp, *maxTokens)
	results, aggs := runner.RunAll(questions)

	report := eval.Report(model, *temp, *hops, len(questions), aggs)
	fmt.Println()
	fmt.Println(report)

	// Write outputs next to the question set (resolveData found it under the
	// real data dir, which may be ../data when running from backend/).
	outPath := *out
	if outPath == "" {
		outPath = filepath.Join(filepath.Dir(qPath), "eval_results.md")
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create output dir: %v\n", err)
	}
	if err := os.WriteFile(outPath, []byte(report), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[eval] report written to %s\n", outPath)
	}

	rawPath := filepath.Join(filepath.Dir(outPath), "eval_results.json")
	if data, err := json.MarshalIndent(results, "", "  "); err != nil {
		fmt.Fprintf(os.Stderr, "marshal raw: %v\n", err)
	} else if err := os.WriteFile(rawPath, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write raw: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[eval] raw answers written to %s\n", rawPath)
	}
}
