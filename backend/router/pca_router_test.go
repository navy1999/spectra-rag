package router

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// defaultRouter returns a router with the hardcoded logic/creative centroids
// (logic at (0.42,-0.18), creative at (-0.31,0.29)).
func defaultRouter(t testing.TB) *PCARouter {
	t.Helper()
	r, err := NewPCARouter(filepath.Join(t.TempDir(), "missing.json"))
	if err != nil {
		t.Fatalf("NewPCARouter: %v", err)
	}
	return r
}

func TestNoveltyFromDist(t *testing.T) {
	cases := []struct {
		dist, want float64
	}{
		{0.0, 0.0},  // clamped low
		{0.3, 0.0},  // lower knee
		{0.65, 0.5}, // midpoint -> the agentic boundary
		{1.0, 1.0},  // upper knee
		{2.0, 1.0},  // clamped high
	}
	for _, c := range cases {
		if got := noveltyFromDist(c.dist); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("noveltyFromDist(%v) = %v, want %v", c.dist, got, c.want)
		}
	}
}

// TestDecide_RegimeIsArgmin checks the core of the dual-signal design: the
// regime label comes from WHICH centroid is nearest, not just how far it is.
func TestDecide_RegimeIsArgmin(t *testing.T) {
	r := defaultRouter(t)

	atLogic := r.decide([2]float64{0.42, -0.18})
	if atLogic.Regime != "logic" {
		t.Errorf("regime at logic centroid = %q, want logic", atLogic.Regime)
	}
	if atLogic.Confidence != 1.0 {
		t.Errorf("confidence at centroid = %v, want 1", atLogic.Confidence)
	}
	if atLogic.Temperature != 0.1 {
		t.Errorf("logic base temp = %v, want 0.1", atLogic.Temperature)
	}
	if atLogic.Path != PathChat {
		t.Errorf("path at centroid = %s, want chat", atLogic.Path)
	}

	atCreative := r.decide([2]float64{-0.31, 0.29})
	if atCreative.Regime != "creative" {
		t.Errorf("regime at creative centroid = %q, want creative", atCreative.Regime)
	}
	if atCreative.Temperature != 0.6 {
		t.Errorf("creative base temp = %v, want 0.6", atCreative.Temperature)
	}
}

// TestDecide_DualSignalIndependence pins the two scenarios a scalar-only router
// cannot express: familiar-but-creative (high temp, no retrieval) and
// novel-but-logical (low-ish temp, agentic retrieval).
func TestDecide_DualSignalIndependence(t *testing.T) {
	r := defaultRouter(t)

	// Exactly at the creative centroid: loose sampling, but no retrieval need.
	familiar := r.decide([2]float64{-0.31, 0.29})
	if familiar.Path != PathChat || familiar.Temperature < 0.5 {
		t.Errorf("familiar-creative: path=%s temp=%v, want chat with temp >= 0.5", familiar.Path, familiar.Temperature)
	}

	// Far out along the logic side: regime logic, but novel → agentic.
	novel := r.decide([2]float64{1.5, -0.6})
	if novel.Regime != "logic" {
		t.Errorf("novel point regime = %q, want logic", novel.Regime)
	}
	if novel.Path != PathAgentic {
		t.Errorf("novel point path = %s, want agentic", novel.Path)
	}
	if novel.Confidence != 0 {
		t.Errorf("novel point confidence = %v, want 0", novel.Confidence)
	}
	// temp = logic base 0.1 + full novelty boost 0.3
	if math.Abs(novel.Temperature-0.4) > 1e-9 {
		t.Errorf("novel-logic temp = %v, want 0.4", novel.Temperature)
	}
}

// TestRouteInvariants verifies policy invariants over a spread of embeddings,
// whichever projection backend produced the points.
func TestRouteInvariants(t *testing.T) {
	r := defaultRouter(t)
	for i := 0; i < 50; i++ {
		emb := make([]float32, 384)
		for j := range emb {
			emb[j] = float32(math.Sin(float64(i*7 + j))) // deterministic spread
		}
		d, err := r.Route(emb)
		if err != nil {
			t.Fatalf("Route: %v", err)
		}
		if d.Temperature < 0 || d.Temperature > maxTemp {
			t.Errorf("temperature %v out of range [0,%v]", d.Temperature, maxTemp)
		}
		if d.Confidence < 0 || d.Confidence > 1 {
			t.Errorf("confidence %v out of range [0,1]", d.Confidence)
		}
		if d.Regime != "logic" && d.Regime != "creative" {
			t.Errorf("regime %q not a known centroid", d.Regime)
		}
		wantAgentic := d.Confidence < agenticConfidenceCutoff
		if (d.Path == PathAgentic) != wantAgentic {
			t.Errorf("path %s but confidence %v (cutoff %v)", d.Path, d.Confidence, agenticConfidenceCutoff)
		}
	}
}

func TestNewPCARouter_LoadsCentroidFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "centroids.json")
	if err := os.WriteFile(path, []byte(`{"logic":[0.42,-0.18],"creative":[-0.31,0.29]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := NewPCARouter(path)
	if err != nil {
		t.Fatalf("NewPCARouter: %v", err)
	}
	if len(r.Centroids()) != 2 {
		t.Errorf("centroids = %d, want 2", len(r.Centroids()))
	}
	if _, err := r.Route(make([]float32, 384)); err != nil {
		t.Errorf("Route after load: %v", err)
	}
}

func BenchmarkRoute(b *testing.B) {
	r := defaultRouter(b)
	emb := make([]float32, 384)
	for j := range emb {
		emb[j] = float32(j%17) / 17.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Route(emb)
	}
}
