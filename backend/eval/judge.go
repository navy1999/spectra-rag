package eval

import (
	"math/rand"
	"strings"
)

// PairVerdict is the outcome of a blind pairwise comparison between the ON
// (control-surfaces) answer and the OFF (baseline) answer.
type PairVerdict int

const (
	PairTie PairVerdict = iota
	OnWins
	OffWins
)

func (v PairVerdict) String() string {
	switch v {
	case OnWins:
		return "on"
	case OffWins:
		return "off"
	default:
		return "tie"
	}
}

// Judge wraps a (stronger, different) model that scores answers. The completion
// function is injectable so the pairing and parse logic are unit-testable
// without a network call.
type Judge struct {
	complete  func(prompt string, temp float64, maxTokens int) (string, error)
	MaxTokens int
	// BothOrders runs each pair in both A/B orderings and only declares a winner
	// when the judge is consistent across orderings, which cancels position bias.
	BothOrders bool
}

// NewJudge builds a Judge backed by a live client. The judge must be a different
// (ideally stronger) model than the one under test — no self-grading.
func NewJudge(c *Client, maxTokens int, bothOrders bool) *Judge {
	return &Judge{complete: c.Complete, MaxTokens: maxTokens, BothOrders: bothOrders}
}

func buildJudgePrompt(question, context, first, second string) string {
	var b strings.Builder
	b.WriteString("You are a strict evaluator comparing two answers to a research question. ")
	b.WriteString("Judge ONLY on accuracy, how well each answer is grounded in the provided context, and completeness.\n\n")
	b.WriteString("Question:\n" + strings.TrimSpace(question) + "\n\n")
	if strings.TrimSpace(context) == "" {
		b.WriteString("Context:\n(none provided)\n\n")
	} else {
		b.WriteString("Context:\n" + strings.TrimSpace(context) + "\n\n")
	}
	b.WriteString("Answer 1:\n" + strings.TrimSpace(first) + "\n\n")
	b.WriteString("Answer 2:\n" + strings.TrimSpace(second) + "\n\n")
	b.WriteString(`Which answer is better? Reply with EXACTLY one token: "1" if Answer 1 is better, "2" if Answer 2 is better, or "TIE" if they are equally good. Do not explain.`)
	return b.String()
}

// parseVerdict maps a judge reply to 1 (first answer), 2 (second answer), or 0
// (tie/unknown). It is lenient: an explicit "tie" wins, otherwise the first
// 1/2 token seen — robust to markdown, leading whitespace, or a short rationale.
func parseVerdict(raw string) int {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" || strings.Contains(s, "tie") {
		return 0
	}
	for _, r := range s {
		switch r {
		case '1':
			return 1
		case '2':
			return 2
		}
	}
	return 0
}

// judgeOnce asks the judge a single ordering and returns 1/2/0 plus the raw reply.
func (j *Judge) judgeOnce(question, context, first, second string) (int, string, error) {
	raw, err := j.complete(buildJudgePrompt(question, context, first, second), 0.0, j.MaxTokens)
	if err != nil {
		return 0, "", err
	}
	return parseVerdict(raw), raw, nil
}

// JudgePair compares the ON answer against the OFF answer blind. With BothOrders
// it runs both orderings and credits a side only if it wins consistently
// (disagreement => tie); otherwise it runs one randomized ordering. Returns the
// verdict and the raw judge reply(s) for logging.
func (j *Judge) JudgePair(question, context, onAns, offAns string, rng *rand.Rand) (PairVerdict, []string, error) {
	if j.BothOrders {
		cA, rawA, err := j.judgeOnce(question, context, onAns, offAns) // ON is Answer 1
		if err != nil {
			return PairTie, nil, err
		}
		cB, rawB, err := j.judgeOnce(question, context, offAns, onAns) // OFF is Answer 1
		if err != nil {
			return PairTie, nil, err
		}
		raws := []string{rawA, rawB}
		switch {
		case cA == 1 && cB == 2: // ON preferred in both orderings
			return OnWins, raws, nil
		case cA == 2 && cB == 1: // OFF preferred in both orderings
			return OffWins, raws, nil
		default: // tie or position-inconsistent
			return PairTie, raws, nil
		}
	}

	onFirst := rng.Intn(2) == 0
	first, second := onAns, offAns
	if !onFirst {
		first, second = offAns, onAns
	}
	c, raw, err := j.judgeOnce(question, context, first, second)
	if err != nil {
		return PairTie, nil, err
	}
	raws := []string{raw}
	switch c {
	case 1:
		if onFirst {
			return OnWins, raws, nil
		}
		return OffWins, raws, nil
	case 2:
		if onFirst {
			return OffWins, raws, nil
		}
		return OnWins, raws, nil
	default:
		return PairTie, raws, nil
	}
}
