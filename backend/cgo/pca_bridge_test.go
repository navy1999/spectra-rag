//go:build !cgo_pca

package cgo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProjectRealPCA(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.json")
	// 2 components over 3 features, mean [1,1,1]: projection is comp·(x-mean).
	if err := os.WriteFile(p, []byte(`{"components":[[1,0,0],[0,1,0]],"mean":[1,1,1]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := LoadModel(p); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	got, err := ProjectToPCA([]float32{2, 3, 9})
	if err != nil {
		t.Fatal(err)
	}
	// out0 = 1*(2-1) = 1 ; out1 = 1*(3-1) = 2
	if got[0] != 1 || got[1] != 2 {
		t.Errorf("real PCA projection = %v, want [1 2]", got)
	}

	// Dimension mismatch falls back to the dev sketch (tanh-bounded), no panic.
	fb, err := ProjectToPCA([]float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if fb[0] <= -1 || fb[0] >= 1 || fb[1] <= -1 || fb[1] >= 1 {
		t.Errorf("expected tanh-bounded fallback for mismatched dims, got %v", fb)
	}
}

func TestLoadModelErrors(t *testing.T) {
	loadedModel = nil
	if err := LoadModel(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("expected error for missing model file")
	}
	if got, _ := ProjectToPCA(nil); got != [2]float64{} {
		t.Errorf("empty embedding should project to origin, got %v", got)
	}
}
