package eval

import (
	"math/rand"
	"strings"
	"testing"
)

func TestParseVerdict(t *testing.T) {
	cases := map[string]int{
		"1":                        1,
		"2":                        2,
		" 1 ":                      1,
		"**2**":                    2,
		"Answer 1":                 1,
		"Answer 2 is better":       2,
		"TIE":                      0,
		"tie":                      0,
		"It's a tie, both are 1":   0, // explicit tie wins over the digit
		"":                         0,
		"neither is good":          0,
		"I prefer the first (1).":  1,
		"The second one, 2, wins.": 2,
	}
	for in, want := range cases {
		if got := parseVerdict(in); got != want {
			t.Errorf("parseVerdict(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestBuildJudgePrompt(t *testing.T) {
	p := buildJudgePrompt("Who wrote X?", "Relevant context:\n- [paper] X", "alpha", "beta")
	for _, sub := range []string{"Who wrote X?", "alpha", "beta", "Answer 1:", "Answer 2:", "TIE", "[paper] X"} {
		if !strings.Contains(p, sub) {
			t.Errorf("prompt missing %q\n%s", sub, p)
		}
	}
	empty := buildJudgePrompt("q", "", "a", "b")
	if !strings.Contains(empty, "(none provided)") {
		t.Errorf("empty-context prompt should note none provided:\n%s", empty)
	}
}

// prefersGood is a stub judge that always picks whichever answer contains "GOOD".
func prefersGood(prompt string, _ float64, _ int) (string, error) {
	i1 := strings.Index(prompt, "Answer 1:")
	i2 := strings.Index(prompt, "Answer 2:")
	if i1 < 0 || i2 < 0 || i2 < i1 {
		return "TIE", nil
	}
	if strings.Contains(prompt[i1:i2], "GOOD") {
		return "1", nil
	}
	return "2", nil
}

func TestJudgePair_BothOrders_ConsistentWinner(t *testing.T) {
	j := &Judge{complete: prefersGood, MaxTokens: 8, BothOrders: true}
	rng := rand.New(rand.NewSource(1))
	v, raws, err := j.JudgePair("q", "ctx", "the GOOD answer", "a weak answer", rng)
	if err != nil {
		t.Fatal(err)
	}
	if v != OnWins {
		t.Errorf("verdict = %v, want OnWins", v)
	}
	if len(raws) != 2 {
		t.Errorf("both-orders should record 2 raw replies, got %d", len(raws))
	}
}

func TestJudgePair_BothOrders_PositionBiasIsTie(t *testing.T) {
	// A judge that always picks Answer 1 is purely position-biased; both orderings
	// disagree on the underlying side, so the verdict must collapse to a tie.
	alwaysFirst := func(string, float64, int) (string, error) { return "1", nil }
	j := &Judge{complete: alwaysFirst, MaxTokens: 8, BothOrders: true}
	rng := rand.New(rand.NewSource(1))
	v, _, err := j.JudgePair("q", "ctx", "on", "off", rng)
	if err != nil {
		t.Fatal(err)
	}
	if v != PairTie {
		t.Errorf("position-biased judge should yield PairTie, got %v", v)
	}
}

func TestJudgePair_SingleOrder_TracksOnAnswer(t *testing.T) {
	j := &Judge{complete: prefersGood, MaxTokens: 8, BothOrders: false}
	// Regardless of the randomized ordering, the ON answer carries "GOOD", so the
	// verdict must map back to OnWins.
	for seed := int64(0); seed < 6; seed++ {
		rng := rand.New(rand.NewSource(seed))
		v, _, err := j.JudgePair("q", "ctx", "the GOOD answer", "weak", rng)
		if err != nil {
			t.Fatal(err)
		}
		if v != OnWins {
			t.Errorf("seed %d: verdict = %v, want OnWins", seed, v)
		}
	}
}

func TestJudgePair_Tie(t *testing.T) {
	tie := func(string, float64, int) (string, error) { return "TIE", nil }
	j := &Judge{complete: tie, MaxTokens: 8, BothOrders: true}
	v, _, err := j.JudgePair("q", "ctx", "on", "off", rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if v != PairTie {
		t.Errorf("verdict = %v, want PairTie", v)
	}
}
