package agent

import "testing"

func TestTallyVotes(t *testing.T) {
	cases := []struct {
		name  string
		votes []Verdict
		want  Verdict
	}{
		{"unanimous sufficient", []Verdict{"A", "A", "A"}, VerdictSufficient},
		{"majority sufficient over insufficient", []Verdict{"A", "A", "B"}, VerdictSufficient},
		{"majority sufficient over uncertain", []Verdict{"A", "A", "C"}, VerdictSufficient},
		{"insufficient majority", []Verdict{"A", "B", "B"}, VerdictInsufficient},
		{"uncertain outweighs lone A", []Verdict{"A", "C", "C"}, VerdictInsufficient},
		{"all insufficient", []Verdict{"B", "B", "B"}, VerdictInsufficient},
		{"empty", nil, VerdictInsufficient},
		{"tie A vs B", []Verdict{"A", "B"}, VerdictInsufficient},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tallyVotes(c.votes); got != c.want {
				t.Errorf("tallyVotes(%v) = %s, want %s", c.votes, got, c.want)
			}
		})
	}
}

func TestRunEvaluator_MockShortCircuits(t *testing.T) {
	cfg := EvaluatorConfig{MockLLM: true}
	got, err := RunEvaluator(cfg, "any query", "any context")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != VerdictSufficient {
		t.Errorf("mock evaluator = %s, want A", got)
	}
}

func TestVotersAreDiverse(t *testing.T) {
	// The ensemble only adds signal if the voters actually differ; identical
	// temperatures would make the three calls collapse to one.
	if len(voters) < 2 {
		t.Fatal("expected multiple voters")
	}
	temps := map[float64]bool{}
	for _, v := range voters {
		temps[v.temperature] = true
		if v.persona == "" {
			t.Error("voter missing persona")
		}
	}
	if len(temps) != len(voters) {
		t.Errorf("expected distinct temperatures, got %d unique across %d voters", len(temps), len(voters))
	}
}
