package eval

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// PairAgg tallies pairwise judge outcomes across a question set.
type PairAgg struct {
	OnWins  int
	OffWins int
	Ties    int
}

func (a *PairAgg) Add(v PairVerdict) {
	switch v {
	case OnWins:
		a.OnWins++
	case OffWins:
		a.OffWins++
	default:
		a.Ties++
	}
}

// Decisive is the number of comparisons with a clear winner (ties excluded).
func (a *PairAgg) Decisive() int { return a.OnWins + a.OffWins }

// Total is every scored comparison, ties included.
func (a *PairAgg) Total() int { return a.OnWins + a.OffWins + a.Ties }

// OnWinRate is the ON win share among decisive (non-tie) comparisons.
func (a *PairAgg) OnWinRate() float64 {
	d := a.Decisive()
	if d == 0 {
		return 0
	}
	return float64(a.OnWins) / float64(d)
}

// Wilson95 is the Wilson score 95% interval for c successes in n trials — honest
// at the small N a free-tier judge eval runs at, unlike the normal approximation
// which can run past [0,1].
func Wilson95(c, n int) (float64, float64) {
	if n == 0 {
		return 0, 0
	}
	const z = 1.96
	nn := float64(n)
	p := float64(c) / nn
	denom := 1 + z*z/nn
	center := (p + z*z/(2*nn)) / denom
	half := (z / denom) * math.Sqrt(p*(1-p)/nn+z*z/(4*nn*nn))
	lo, hi := center-half, center+half
	return math.Max(0, lo), math.Min(1, hi)
}

// PairReport renders the pairwise judge outcome as a Markdown report.
func PairReport(genModel, judgeModel, onCond, offCond string, n int, bothOrders bool, a *PairAgg) string {
	pct := func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) }
	var b strings.Builder

	fmt.Fprintf(&b, "## Phase 2 — end-to-end answer quality (LLM-as-judge)\n\n")
	order := "single randomized A/B ordering per pair"
	if bothOrders {
		order = "both A/B orderings per pair (a side scores only if it wins consistently, cancelling position bias)"
	}
	fmt.Fprintf(&b, "Generator `%s` · judge `%s` · scored on %d of %d questions.\n", genModel, judgeModel, a.Total(), n)
	fmt.Fprintf(&b, "Blind pairwise: **ON = `%s`** vs **OFF = `%s`**, %s.\n\n", onCond, offCond, order)
	fmt.Fprintf(&b, "Generated %s.\n\n", time.Now().Format("2006-01-02"))

	fmt.Fprintf(&b, "| Outcome | count |\n|---|---|\n")
	fmt.Fprintf(&b, "| ON (`%s`) preferred | %d |\n", onCond, a.OnWins)
	fmt.Fprintf(&b, "| OFF (`%s`) preferred | %d |\n", offCond, a.OffWins)
	if bothOrders {
		fmt.Fprintf(&b, "| Tie / inconsistent across orderings | %d |\n", a.Ties)
	} else {
		fmt.Fprintf(&b, "| Tie | %d |\n", a.Ties)
	}

	d := a.Decisive()
	lo, hi := Wilson95(a.OnWins, d)
	fmt.Fprintf(&b, "\n**ON win-rate among decisive comparisons: %d/%d = %s** (95%% CI %s–%s).\n\n",
		a.OnWins, d, pct(a.OnWinRate()), pct(lo), pct(hi))

	switch {
	case d == 0:
		fmt.Fprintf(&b, "Every comparison tied — the judge saw no decisive quality difference between ON and OFF.\n\n")
	case lo > 0.5:
		fmt.Fprintf(&b, "**ON (`%s`) is preferred over OFF (`%s`):** the lower 95%% bound stays above 50%%. Attribute the gain to whatever ON adds over OFF (here: %s vs %s), not to the pipeline as a whole.\n\n", onCond, offCond, onCond, offCond)
	case hi < 0.5:
		fmt.Fprintf(&b, "**OFF (`%s`) is preferred over ON (`%s`):** the upper 95%% bound is below 50%%.\n\n", offCond, onCond)
	default:
		fmt.Fprintf(&b, "**Inconclusive at this N:** the 95%% CI spans 50%%, so this run does not establish a quality difference between `%s` and `%s`.\n\n", onCond, offCond)
	}

	fmt.Fprintf(&b, "_LLM-as-judge is a proxy for human preference, not ground truth. The judge is a different, larger model than the one under test (no self-grading), comparisons are blind, and position bias is controlled. N is small, so treat this as directional, not definitive. Any measured difference is attributable only to what ON (`%s`) adds over OFF (`%s`)._\n", onCond, offCond)
	return b.String()
}
