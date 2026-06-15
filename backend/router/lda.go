package router

import (
	"encoding/json"
	"fmt"
	"os"
)

// ldaRouter is the supervised chat-vs-agentic decision: a single linear boundary
// over the RAW query embedding, w·x + b. Fitted offline by scripts/fit_lda.py
// (shrinkage-LDA with Ledoit-Wolf covariance, no PCA pre-reduction — the config
// that scored 97.5% leave-one-out, permutation-confirmed at p=0.024). When
// loaded it OVERRIDES the unsupervised PCA-2D path decision; the 2D projection
// still drives temperature and the visualization. It is a full-dimension linear
// model, which is why it cannot be expressed as the router's 2D centroid scheme.
type ldaRouter struct {
	w            []float64
	b            float64
	dim          int
	posIsAgentic bool // true if w·x + b > 0 means the agentic class
}

// decide returns the routed path for a raw embedding, or ok=false when the model
// is absent or the embedding dimension does not match (caller keeps the PCA path).
func (l *ldaRouter) decide(emb []float32) (RoutePath, bool) {
	if l == nil || l.dim == 0 || len(emb) != l.dim {
		return "", false
	}
	s := l.b
	for i, v := range emb {
		s += l.w[i] * float64(v)
	}
	if (s > 0) == l.posIsAgentic {
		return PathAgentic, true
	}
	return PathChat, true
}

func loadLDA(path string) (*ldaRouter, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Dim           int       `json:"dim"`
		W             []float64 `json:"w"`
		B             float64   `json:"b"`
		PositiveClass string    `json:"positive_class"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse lda router: %w", err)
	}
	if doc.Dim == 0 || len(doc.W) != doc.Dim {
		return nil, fmt.Errorf("lda router: dim %d but %d weights", doc.Dim, len(doc.W))
	}
	// positive_class names the class predicted when w·x + b > 0; default agentic
	// (classes sort to [chat, agentic], so the positive side is agentic).
	return &ldaRouter{w: doc.W, b: doc.B, dim: doc.Dim, posIsAgentic: doc.PositiveClass != "chat"}, nil
}
