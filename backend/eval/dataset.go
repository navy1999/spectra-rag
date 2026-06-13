// Package eval is a Phase-1 evaluation harness that measures what the spectra
// control-surface algorithms add to a small free-tier model. It runs the same
// model under three conditions (raw, plain RAG, full spectra pipeline) over a
// graph-grounded question set and scores answers with judge-free metrics:
// entity-spelling fidelity (the trie guard, A2) and repetition (the SVD
// redundancy penalty, A3), plus a groundedness proxy. The "spectra" condition
// reuses the real retrieval, synthesis, and trie packages, not reimplementations.
package eval

import (
	"encoding/json"
	"fmt"
	"os"
)

type Question struct {
	ID       string   `json:"id"`
	Text     string   `json:"text"`
	Entities []string `json:"entities"`
}

type Dataset struct {
	Model     string     `json:"model"`
	Questions []Question `json:"questions"`
}

func LoadDataset(path string) (*Dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read questions: %w", err)
	}
	var d Dataset
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("parse questions: %w", err)
	}
	if len(d.Questions) == 0 {
		return nil, fmt.Errorf("question set %q is empty", path)
	}
	return &d, nil
}
