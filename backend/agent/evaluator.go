package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
)

// Verdict represents the evaluator's confidence vote.
type Verdict string

const (
	VerdictSufficient   Verdict = "A"
	VerdictInsufficient Verdict = "B"
	VerdictUncertain    Verdict = "C"
)

type EvaluatorConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	MockLLM bool
}

// voter defines one member of the evaluation ensemble. Giving each voter a
// distinct persona and temperature is what makes the 3-way vote meaningful:
// with identical prompts at temperature 0, all three calls return the same
// token, so the "ensemble" carries no more signal than a single call. The
// diversity here lets borderline contexts produce split votes (e.g. A/A/B)
// that the majority rule then resolves — the closest we can get to a
// confidence estimate on providers that expose no logprobs on free models.
type voter struct {
	persona     string
	temperature float64
}

var voters = []voter{
	{persona: "You are a STRICT reviewer: demand comprehensive, directly relevant evidence before judging the context sufficient.", temperature: 0.1},
	{persona: "You are a BALANCED reviewer: judge whether the context reasonably covers the question.", temperature: 0.5},
	{persona: "You are a LENIENT reviewer: accept the context if it contains partial but on-topic evidence.", temperature: 0.9},
}

// RunEvaluator asks an ensemble of differently-tempered LLM personas to vote
// A/B/C (max_tokens=1) on whether the retrieved context answers the query, then
// resolves the votes by majority via tallyVotes.
func RunEvaluator(cfg EvaluatorConfig, query, context string) (Verdict, error) {
	if cfg.MockLLM {
		return VerdictSufficient, nil
	}

	ctx := context
	if len(ctx) > 500 {
		ctx = ctx[:500]
	}

	votes := make([]Verdict, len(voters))
	var wg sync.WaitGroup
	for i, v := range voters {
		wg.Add(1)
		go func(i int, v voter) {
			defer wg.Done()
			prompt := fmt.Sprintf(
				"%s\n\n"+
					"Question: %s\n"+
					"Retrieved context:\n%s\n\n"+
					"Does the context sufficiently answer the question? Reply with ONE letter:\n"+
					"A = Sufficient\nB = Need more context\nC = Uncertain\n\nAnswer:",
				v.persona, query, ctx)
			votes[i] = callEvaluatorLLM(cfg, prompt, v.temperature)
		}(i, v)
	}
	wg.Wait()

	return tallyVotes(votes), nil
}

// tallyVotes resolves the ensemble by majority. Sufficient (A) wins only when it
// strictly out-votes Insufficient and is at least as common as Uncertain;
// otherwise we return Insufficient so the agent keeps expanding rather than
// stopping early on a weak signal.
func tallyVotes(votes []Verdict) Verdict {
	counts := map[Verdict]int{}
	for _, v := range votes {
		counts[v]++
	}
	if counts[VerdictSufficient] > counts[VerdictInsufficient] &&
		counts[VerdictSufficient] >= counts[VerdictUncertain] {
		return VerdictSufficient
	}
	return VerdictInsufficient
}

func callEvaluatorLLM(cfg EvaluatorConfig, prompt string, temperature float64) Verdict {
	body, _ := json.Marshal(map[string]interface{}{
		"model": cfg.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens":  1,
		"temperature": temperature,
	})

	req, _ := http.NewRequest("POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return VerdictUncertain
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil || len(result.Choices) == 0 {
		return VerdictUncertain
	}

	content := strings.TrimSpace(result.Choices[0].Message.Content)
	switch {
	case strings.HasPrefix(content, "A"):
		return VerdictSufficient
	case strings.HasPrefix(content, "B"):
		return VerdictInsufficient
	default:
		return VerdictUncertain
	}
}
