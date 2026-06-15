package router

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLDADecide(t *testing.T) {
	// score = x0; > 0 => agentic
	l := &ldaRouter{w: []float64{1, 0}, b: 0, dim: 2, posIsAgentic: true}
	if p, ok := l.decide([]float32{1, 0}); !ok || p != PathAgentic {
		t.Errorf("positive score should route agentic, got %v ok=%v", p, ok)
	}
	if p, ok := l.decide([]float32{-1, 0}); !ok || p != PathChat {
		t.Errorf("negative score should route chat, got %v ok=%v", p, ok)
	}
}

func TestLDADecide_Orientation(t *testing.T) {
	// posIsAgentic=false: > 0 => chat
	l := &ldaRouter{w: []float64{1}, b: 0, dim: 1, posIsAgentic: false}
	if p, _ := l.decide([]float32{1}); p != PathChat {
		t.Errorf("posIsAgentic=false: positive score should be chat, got %v", p)
	}
}

func TestLDADecide_GuardsMismatch(t *testing.T) {
	l := &ldaRouter{w: []float64{1, 0}, b: 0, dim: 2, posIsAgentic: true}
	if _, ok := l.decide([]float32{1}); ok {
		t.Error("dimension mismatch must return ok=false")
	}
	var nilL *ldaRouter
	if _, ok := nilL.decide([]float32{1, 2}); ok {
		t.Error("nil router must return ok=false")
	}
}

func TestLoadLDA(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "lda.json")
	if err := os.WriteFile(good, []byte(`{"dim":2,"w":[0.5,-0.5],"b":0.1,"positive_class":"agentic"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	l, err := loadLDA(good)
	if err != nil {
		t.Fatal(err)
	}
	if l.dim != 2 || !l.posIsAgentic {
		t.Errorf("parsed wrong: dim=%d posIsAgentic=%v", l.dim, l.posIsAgentic)
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"dim":3,"w":[1,2],"b":0}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadLDA(bad); err == nil {
		t.Error("dim/weight-count mismatch should error")
	}
}
