// Package llmcaps reads each model's natively-supported sampling parameters from
// OpenRouter's /models endpoint ("supported_parameters"). The request builder
// gates the body on this so a user-picked model never receives a parameter its
// provider rejects with a 400 — and, crucially, so we send the *real* knob
// (e.g. frequency_penalty) when it exists instead of a heuristic stand-in.
//
// Capability support is per-model on OpenRouter: e.g. liquid/lfm-2.5-1.2b-instruct
// exposes temperature, top_p, top_k, min_p, frequency_penalty, presence_penalty,
// repetition_penalty, and seed — but not logit_bias or logprobs. So the trie (A2,
// logit_bias) and the vote evaluator (A4, logprobs) remain justified workarounds,
// while the SVD frequency penalty (A3) can defer to the native parameter.
package llmcaps

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Capabilities maps a model id to the set of sampling parameters it accepts.
type Capabilities struct {
	byModel map[string]map[string]bool
}

// Supports reports whether the model accepts the given sampling parameter.
// Unknown models (and a nil receiver) return false, so callers fall back to the
// safe minimum (temperature only) rather than risk a provider 400.
func (c *Capabilities) Supports(model, param string) bool {
	if c == nil || c.byModel == nil {
		return false
	}
	return c.byModel[model][param]
}

// Known reports whether capability data exists for a model at all.
func (c *Capabilities) Known(model string) bool {
	if c == nil || c.byModel == nil {
		return false
	}
	_, ok := c.byModel[model]
	return ok
}

// NewFromMap builds Capabilities from an explicit model→params map (tests and a
// static fallback table).
func NewFromMap(m map[string][]string) *Capabilities {
	byModel := make(map[string]map[string]bool, len(m))
	for model, params := range m {
		set := make(map[string]bool, len(params))
		for _, p := range params {
			set[p] = true
		}
		byModel[model] = set
	}
	return &Capabilities{byModel: byModel}
}

type modelsResponse struct {
	Data []struct {
		ID                  string   `json:"id"`
		SupportedParameters []string `json:"supported_parameters"`
	} `json:"data"`
}

// parse builds Capabilities from the raw /models JSON. Separated from Fetch so it
// is unit-testable without network.
func parse(raw []byte) (*Capabilities, error) {
	var mr modelsResponse
	if err := json.Unmarshal(raw, &mr); err != nil {
		return nil, fmt.Errorf("parse models: %w", err)
	}
	if len(mr.Data) == 0 {
		return nil, fmt.Errorf("models response had no entries")
	}
	byModel := make(map[string]map[string]bool, len(mr.Data))
	for _, m := range mr.Data {
		set := make(map[string]bool, len(m.SupportedParameters))
		for _, p := range m.SupportedParameters {
			set[p] = true
		}
		byModel[m.ID] = set
	}
	return &Capabilities{byModel: byModel}, nil
}

// Fetch pulls the capability table from OpenRouter. Best-effort: callers should
// treat an error as "no capability data" and fall back to temperature-only.
func Fetch(baseURL string) (*Capabilities, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(baseURL + "/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("models endpoint HTTP %d", resp.StatusCode)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return parse(raw)
}
