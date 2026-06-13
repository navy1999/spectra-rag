//go:build cgo_pca

package cgo

/*
#cgo LDFLAGS: -L../pca_engine/build -lpca -lstdc++
#include "../pca_engine/include/pca.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"os"
	"unsafe"
)

// loadedMethod records the fitter that produced the loaded model (read from the
// JSON's "method" field), so the eval report can label the router row correctly
// even on the C++ fast path.
var loadedMethod string

// LoadModel loads the PCA/LDA model JSON (produced by scripts/fit_pca.py or
// scripts/fit_lda.py) into the C++ engine. Must be called once at startup before
// ProjectToPCA, or every projection will report the model as not loaded.
func LoadModel(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if ret := C.pca_load_model(cpath); ret != 0 {
		return fmt.Errorf("pca_load_model(%q) returned %d", path, int(ret))
	}
	// Best-effort read of the method label for reporting; projection itself is
	// handled by the C++ engine and does not depend on this.
	loadedMethod = ""
	if data, err := os.ReadFile(path); err == nil {
		var meta struct {
			Method string `json:"method"`
		}
		if json.Unmarshal(data, &meta) == nil {
			loadedMethod = meta.Method
		}
	}
	return nil
}

// LoadedMethod returns the fitter that produced the loaded model (e.g. "pca",
// "pca16_lda"), or "" if unknown.
func LoadedMethod() string { return loadedMethod }

func ProjectToPCA(embedding []float32) ([2]float64, error) {
	if C.pca_is_loaded() == 0 {
		return [2]float64{}, fmt.Errorf("PCA model not loaded")
	}
	if len(embedding) == 0 {
		return [2]float64{}, fmt.Errorf("empty embedding")
	}
	var outX, outY C.double
	ptr := (*C.float)(unsafe.Pointer(&embedding[0]))
	ret := C.pca_project(ptr, C.int(len(embedding)), &outX, &outY)
	if ret != 0 {
		return [2]float64{}, fmt.Errorf("pca_project failed: %d", int(ret))
	}
	return [2]float64{float64(outX), float64(outY)}, nil
}
