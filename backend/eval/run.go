package eval

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/synthesis"
	"github.com/navy1999/spectra-rag/backend/trie"
)

type CondName string

const (
	CondRaw        CondName = "raw"
	CondRAGPlain   CondName = "rag_plain"
	CondRAGSpectra CondName = "rag_spectra"
)

// Conditions is the fixed evaluation order. All three use the same model; the
// only variables are retrieved context (raw has none) and the spectra synthesis
// + trie layers (rag_spectra only).
var Conditions = []CondName{CondRaw, CondRAGPlain, CondRAGSpectra}

const basePrompt = "You are a helpful research assistant with access to an academic knowledge graph.\n\n"

// AnswerResult is one model answer under one condition, with its scores.
type AnswerResult struct {
	QuestionID  string   `json:"question_id"`
	Condition   CondName `json:"condition"`
	Answer      string   `json:"answer"`
	LatencyMs   int64    `json:"latency_ms"`
	Corrections int      `json:"corrections"`
	FreqPenalty float64  `json:"freq_penalty"`
	Score       scoreRec `json:"score"`
}

type scoreRec struct {
	ExactRate    float64 `json:"exact_rate"`
	NearMissRate float64 `json:"near_miss_rate"`
	Recall       float64 `json:"recall"`
	Distinct2    float64 `json:"distinct2"`
	Grounded     float64 `json:"grounded"`
	HasGrounded  bool    `json:"has_grounded"`
}

type Runner struct {
	Graph     *retrieval.Graph
	Trie      *trie.Trie
	LLM       *Client
	Hops      int
	Temp      float64
	MaxTokens int
	vocab     []string
}

func NewRunner(g *retrieval.Graph, t *trie.Trie, llm *Client, hops int, temp float64, maxTokens int) *Runner {
	return &Runner{Graph: g, Trie: t, LLM: llm, Hops: hops, Temp: temp, MaxTokens: maxTokens, vocab: g.AllNodeNames()}
}

func contextBlock(chunks []string) string {
	if len(chunks) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Relevant context:\n")
	for _, c := range chunks {
		b.WriteString("- " + c + "\n")
	}
	b.WriteString("\n")
	return b.String()
}

// RunAll evaluates every question under every condition and returns the raw
// per-answer results plus per-condition aggregates. Progress goes to stderr so
// long, rate-limited runs are observable.
func (r *Runner) RunAll(questions []Question) ([]AnswerResult, map[CondName]*condAgg) {
	var results []AnswerResult
	aggs := map[CondName]*condAgg{}
	for _, c := range Conditions {
		aggs[c] = &condAgg{}
	}

	for i, q := range questions {
		chunks := retrieveContext(r.Graph, q.Text, r.Hops)
		freq := synthesis.ComputeFrequencyPenalty(chunks)
		ctx := contextBlock(chunks)

		for _, cond := range Conditions {
			fmt.Fprintf(os.Stderr, "[eval] %d/%d %s · %s\n", i+1, len(questions), q.ID, cond)

			prompt := basePrompt
			useCtx := ""
			if cond == CondRAGSpectra {
				if instr := synthesis.PenaltyInstruction(freq); instr != "" {
					prompt += instr + "\n\n"
				}
			}
			if cond != CondRaw {
				useCtx = ctx
			}
			prompt += useCtx + "User question: " + q.Text

			start := time.Now()
			answer, err := r.LLM.Complete(prompt, r.Temp, r.MaxTokens)
			latency := time.Since(start).Milliseconds()
			if err != nil {
				fmt.Fprintf(os.Stderr, "[eval]   skipped (%v)\n", err)
				continue
			}

			corrections := 0
			if cond == CondRAGSpectra {
				si := trie.NewInterceptor(r.Trie)
				corrected, _ := si.ProcessToken(answer)
				corrected += si.Flush()
				answer = corrected
				corrections = si.Count()
			}

			scoreCtx := ""
			if cond != CondRaw {
				scoreCtx = ctx
			}
			rec := r.score(q, answer, scoreCtx)
			results = append(results, AnswerResult{
				QuestionID:  q.ID,
				Condition:   cond,
				Answer:      answer,
				LatencyMs:   latency,
				Corrections: corrections,
				FreqPenalty: freq,
				Score:       rec,
			})
			aggs[cond].add(rec, latency, corrections)
		}
	}
	return results, aggs
}

func (r *Runner) score(q Question, answer, ctx string) scoreRec {
	var exact, near, recall int
	for _, e := range q.Entities {
		switch ClassifyEntity(answer, e) {
		case EntityExact:
			exact++
			recall++
		case EntityNearMiss:
			near++
			recall++
		}
	}
	var rec scoreRec
	if n := len(q.Entities); n > 0 {
		rec.ExactRate = float64(exact) / float64(n)
		rec.NearMissRate = float64(near) / float64(n)
		rec.Recall = float64(recall) / float64(n)
	}
	rec.Distinct2 = Distinct2(answer)
	if ctx != "" {
		if g, ok := Groundedness(answer, ctx, r.vocab); ok {
			rec.Grounded = g
			rec.HasGrounded = true
		}
	}
	return rec
}

type condAgg struct {
	n                                       int
	exact, near, recall, distinct2, latency float64
	groundSum                               float64
	groundN                                 int
	corrections                             int
}

func (a *condAgg) add(rec scoreRec, latency int64, corrections int) {
	a.n++
	a.exact += rec.ExactRate
	a.near += rec.NearMissRate
	a.recall += rec.Recall
	a.distinct2 += rec.Distinct2
	a.latency += float64(latency)
	a.corrections += corrections
	if rec.HasGrounded {
		a.groundSum += rec.Grounded
		a.groundN++
	}
}

func (a *condAgg) mean(v float64) float64 {
	if a.n == 0 {
		return 0
	}
	return v / float64(a.n)
}

// Report renders the aggregates as a Markdown table suitable for the README.
func Report(model string, temp float64, hops, n int, aggs map[CondName]*condAgg) string {
	pct := func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) }
	var b strings.Builder
	nRaw, nPlain, nSpec := aggs[CondRaw].n, aggs[CondRAGPlain].n, aggs[CondRAGSpectra].n
	scored := fmt.Sprintf("%d", nRaw)
	if !(nRaw == nPlain && nPlain == nSpec) {
		scored = fmt.Sprintf("raw %d / plain %d / spectra %d", nRaw, nPlain, nSpec)
	}
	fmt.Fprintf(&b, "## Phase 1 evaluation\n\n")
	fmt.Fprintf(&b, "Model `%s` · scored on %s of %d questions", model, scored, n)
	if nRaw < n {
		fmt.Fprintf(&b, " (remainder skipped: provider rate-limited)")
	}
	fmt.Fprintf(&b, " · same model across conditions · temperature %.2f · retrieval held fixed across both RAG conditions (seed + %d-hop BFS).\n\n", temp, hops)
	fmt.Fprintf(&b, "Generated %s.\n\n", time.Now().Format("2006-01-02"))

	fmt.Fprintf(&b, "| Metric | raw | rag_plain | rag_spectra |\n")
	fmt.Fprintf(&b, "|---|---|---|---|\n")

	raw, plain, spectra := aggs[CondRaw], aggs[CondRAGPlain], aggs[CondRAGSpectra]
	row := func(label string, f func(a *condAgg) string) {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", label, f(raw), f(plain), f(spectra))
	}

	row("Entity exact-spelling rate (higher better)", func(a *condAgg) string { return pct(a.mean(a.exact)) })
	row("Entity near-miss rate (lower better)", func(a *condAgg) string { return pct(a.mean(a.near)) })
	row("Entity recall, any form (higher better)", func(a *condAgg) string { return pct(a.mean(a.recall)) })
	row("Repetition: distinct-2 (higher better)", func(a *condAgg) string { return fmt.Sprintf("%.3f", a.mean(a.distinct2)) })
	row("Groundedness, RAG only (higher better)", func(a *condAgg) string {
		if a.groundN == 0 {
			return "—"
		}
		return pct(a.groundSum / float64(a.groundN))
	})
	row("Trie corrections (total)", func(a *condAgg) string {
		if a.corrections == 0 {
			return "—"
		}
		return fmt.Sprintf("%d", a.corrections)
	})
	row("Mean latency (ms)", func(a *condAgg) string { return fmt.Sprintf("%.0f", a.mean(a.latency)) })

	fmt.Fprintf(&b, "\n_Entity fidelity is the trie guard (A2): the spectra column should raise exact-spelling and cut near-misses. Repetition (distinct-2) is the SVD penalty (A3). Metrics are judge-free string measures; routing (A1) and the vote evaluator (A4) are reported separately._\n")
	return b.String()
}
