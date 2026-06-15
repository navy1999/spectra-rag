// Command qualityeval runs Phase 2: a blind, position-bias-controlled pairwise
// LLM-as-judge comparison of the full control-surface pipeline (ON) against a
// plain-RAG baseline (OFF), to test whether the spectra layers actually improve
// end-to-end answer quality — the project's headline claim, which the judge-free
// Phase 1 metrics do not measure. Requires OPENROUTER_API_KEY.
//
// Usage (from repo root or backend/):
//
//	OPENROUTER_API_KEY=sk-or-... go run ./cmd/qualityeval
//	OPENROUTER_API_KEY=sk-or-... go run ./cmd/qualityeval -limit 5
//	OPENROUTER_API_KEY=sk-or-... go run ./cmd/qualityeval -on rag_spectra -off raw
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/eval"
	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/synthesis"
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

func validCond(c string) bool {
	switch eval.CondName(c) {
	case eval.CondRaw, eval.CondRAGPlain, eval.CondRAGSpectra:
		return true
	}
	return false
}

type pairRec struct {
	QuestionID string   `json:"question_id"`
	Verdict    string   `json:"verdict"`
	OnAnswer   string   `json:"on_answer"`
	OffAnswer  string   `json:"off_answer"`
	JudgeRaw   []string `json:"judge_raw"`
}

func main() {
	questionsPath := flag.String("questions", "", "question set JSON (default: data/eval_questions.json)")
	genModel := flag.String("model", "", "generator model under test (default: dataset model)")
	judgeModel := flag.String("judge", "openai/gpt-oss-120b:free", "judge model (must differ from generator)")
	onCond := flag.String("on", string(eval.CondRAGSpectra), "ON condition: raw|rag_plain|rag_spectra")
	offCond := flag.String("off", string(eval.CondRAGPlain), "OFF condition: raw|rag_plain|rag_spectra")
	hops := flag.Int("hops", 1, "BFS hops for retrieval")
	temp := flag.Float64("temp", 0.3, "generator temperature (fixed across conditions)")
	maxTokens := flag.Int("max-tokens", 400, "max tokens per generated answer")
	judgeMaxTokens := flag.Int("judge-max-tokens", 16, "max tokens for the judge reply")
	limit := flag.Int("limit", 0, "evaluate only the first N questions (0 = all)")
	singleOrder := flag.Bool("single-order", false, "use one randomized ordering per pair instead of both (faster, less rigorous)")
	seed := flag.Int64("seed", 1, "RNG seed for single-order randomization")
	out := flag.String("out", "", "markdown report path (default: data/quality_eval_results.md)")
	genCache := flag.String("cache", "eval_cache.json", "generator response cache (shared with cmd/eval to reuse answers)")
	judgeCache := flag.String("judge-cache", "quality_judge_cache.json", "judge response cache")
	flag.Parse()

	cfg := config.Load()
	if cfg.OpenRouterAPIKey == "" {
		fmt.Fprintln(os.Stderr, "qualityeval needs live models. Set OPENROUTER_API_KEY and re-run:")
		fmt.Fprintln(os.Stderr, "  PowerShell:  $env:OPENROUTER_API_KEY=\"sk-or-...\"; go run ./cmd/qualityeval")
		fmt.Fprintln(os.Stderr, "  bash:        OPENROUTER_API_KEY=sk-or-... go run ./cmd/qualityeval")
		os.Exit(1)
	}

	if !validCond(*onCond) || !validCond(*offCond) {
		fmt.Fprintf(os.Stderr, "invalid -on/-off condition (use raw|rag_plain|rag_spectra)\n")
		os.Exit(1)
	}
	if *onCond == *offCond {
		fmt.Fprintf(os.Stderr, "-on and -off must differ (got %q for both)\n", *onCond)
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

	gm := *genModel
	if gm == "" {
		gm = ds.Model
	}
	if gm == "" {
		gm = cfg.DefaultModel
	}
	if gm == *judgeModel {
		fmt.Fprintf(os.Stderr, "judge (%s) must differ from generator (%s) — no self-grading\n", *judgeModel, gm)
		os.Exit(1)
	}

	genClient := eval.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL, gm, *genCache)
	judgeClient := eval.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL, *judgeModel, *judgeCache)

	fmt.Fprintf(os.Stderr, "[qualityeval] preflight generator %s ... ", gm)
	if err := genClient.Preflight(); err != nil {
		fmt.Fprintf(os.Stderr, "unavailable (%v)\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "ok")
	fmt.Fprintf(os.Stderr, "[qualityeval] preflight judge %s ... ", *judgeModel)
	if err := judgeClient.Preflight(); err != nil {
		fmt.Fprintf(os.Stderr, "unavailable (%v)\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "ok")

	runner := eval.NewRunner(graph, t, genClient, *hops, *temp, *maxTokens)
	judge := eval.NewJudge(judgeClient, *judgeMaxTokens, !*singleOrder)
	rng := rand.New(rand.NewSource(*seed))
	onC, offC := eval.CondName(*onCond), eval.CondName(*offCond)

	fmt.Fprintf(os.Stderr, "[qualityeval] gen=%s judge=%s on=%s off=%s questions=%d bothOrders=%v\n",
		gm, *judgeModel, *onCond, *offCond, len(questions), !*singleOrder)

	var agg eval.PairAgg
	var recs []pairRec
	for i, q := range questions {
		chunks := eval.RetrieveContext(graph, q.Text, *hops)
		freq := synthesis.ComputeFrequencyPenalty(chunks)
		ctxBlock := eval.ContextBlock(chunks)

		fmt.Fprintf(os.Stderr, "[qualityeval] %d/%d %s · generating\n", i+1, len(questions), q.ID)
		onAns, _, _, err := runner.Answer(q, onC, chunks, freq)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip (on: %v)\n", err)
			continue
		}
		offAns, _, _, err := runner.Answer(q, offC, chunks, freq)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip (off: %v)\n", err)
			continue
		}

		fmt.Fprintf(os.Stderr, "[qualityeval] %d/%d %s · judging\n", i+1, len(questions), q.ID)
		verdict, raws, err := judge.JudgePair(q.Text, ctxBlock, onAns, offAns, rng)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip (judge: %v)\n", err)
			continue
		}
		agg.Add(verdict)
		recs = append(recs, pairRec{QuestionID: q.ID, Verdict: verdict.String(), OnAnswer: onAns, OffAnswer: offAns, JudgeRaw: raws})
		fmt.Fprintf(os.Stderr, "  -> %s\n", verdict.String())
	}

	report := eval.PairReport(gm, *judgeModel, *onCond, *offCond, len(questions), !*singleOrder, &agg)
	fmt.Println()
	fmt.Println(report)

	outPath := *out
	if outPath == "" {
		outPath = filepath.Join(filepath.Dir(qPath), "quality_eval_results.md")
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create output dir: %v\n", err)
	}
	if err := os.WriteFile(outPath, []byte(report), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write report: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[qualityeval] report written to %s\n", outPath)
	}

	rawPath := filepath.Join(filepath.Dir(outPath), "quality_eval_results.json")
	if data, err := json.MarshalIndent(recs, "", "  "); err != nil {
		fmt.Fprintf(os.Stderr, "marshal raw: %v\n", err)
	} else if err := os.WriteFile(rawPath, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write raw: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "[qualityeval] raw answers written to %s\n", rawPath)
	}
}
