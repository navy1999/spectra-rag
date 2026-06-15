package eval

import (
	"math"
	"testing"
)

func TestPairAggCounts(t *testing.T) {
	var a PairAgg
	seq := []PairVerdict{OnWins, OnWins, OnWins, OffWins, PairTie, PairTie}
	for _, v := range seq {
		a.Add(v)
	}
	if a.OnWins != 3 || a.OffWins != 1 || a.Ties != 2 {
		t.Fatalf("counts = on %d off %d tie %d", a.OnWins, a.OffWins, a.Ties)
	}
	if a.Decisive() != 4 {
		t.Errorf("Decisive() = %d, want 4", a.Decisive())
	}
	if a.Total() != 6 {
		t.Errorf("Total() = %d, want 6", a.Total())
	}
	if math.Abs(a.OnWinRate()-0.75) > 1e-9 {
		t.Errorf("OnWinRate() = %v, want 0.75", a.OnWinRate())
	}
}

func TestPairAggOnWinRateNoDecisive(t *testing.T) {
	var a PairAgg
	a.Add(PairTie)
	if a.OnWinRate() != 0 {
		t.Errorf("OnWinRate() with no decisive pairs = %v, want 0", a.OnWinRate())
	}
}

func TestWilson95(t *testing.T) {
	if lo, hi := Wilson95(0, 0); lo != 0 || hi != 0 {
		t.Errorf("Wilson95(0,0) = (%v,%v), want (0,0)", lo, hi)
	}
	// Unanimous wins: lower bound well above 0.5, upper bound capped at 1.
	lo, hi := Wilson95(10, 10)
	if lo <= 0.6 || hi != 1.0 {
		t.Errorf("Wilson95(10,10) = (%v,%v); want lo>0.6, hi==1", lo, hi)
	}
	// Even split: interval straddles 0.5 and stays inside [0,1].
	lo, hi = Wilson95(5, 10)
	if lo <= 0 || hi >= 1 || !(lo < 0.5 && hi > 0.5) {
		t.Errorf("Wilson95(5,10) = (%v,%v); want straddling 0.5 within (0,1)", lo, hi)
	}
}
