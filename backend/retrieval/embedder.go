package retrieval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

type Embedder struct {
	apiKey  string
	baseURL string
	model   string
	// task is an optional Jina v3 task adapter sent on every request. The query
	// embedding is used ONLY for routing (retrieval is graph traversal, not
	// vector search), and the supervised router is fitted on classification-task
	// embeddings, so the server must embed routing queries with the SAME task or
	// the projection silently degrades. Empty = no task field (general default).
	task  string
	cache sync.Map
}

func NewEmbedder(apiKey, baseURL, model string) *Embedder {
	return &Embedder{apiKey: apiKey, baseURL: baseURL, model: model}
}

// NewEmbedderWithTask is like NewEmbedder but pins a Jina v3 task adapter (e.g.
// "classification") on every request, to match the task the routing projection
// was fitted on.
func NewEmbedderWithTask(apiKey, baseURL, model, task string) *Embedder {
	return &Embedder{apiKey: apiKey, baseURL: baseURL, model: model, task: task}
}

// Mock reports whether the embedder has no API key configured and will return
// deterministic hash vectors. This is an offline/dev mode only — PCA routing is
// NOT semantically meaningful in it, and the server logs a warning at startup.
func (e *Embedder) Mock() bool { return e.apiKey == "" }

// Embed returns the embedding for text. With a key configured it calls the
// embeddings provider and returns a real error on failure (no silent fallback to
// a fake vector — a fake vector would make the router quietly meaningless). With
// no key it returns a deterministic mock so the app still boots offline.
func (e *Embedder) Embed(text string) ([]float32, error) {
	if v, ok := e.cache.Load(text); ok {
		return v.([]float32), nil
	}
	if e.apiKey == "" {
		return mockEmbedding(text), nil
	}

	payload := map[string]interface{}{
		"model": e.model,
		"input": []string{text},
	}
	if e.task != "" {
		payload["task"] = e.task
	}
	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("embeddings provider HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embeddings response: %w", err)
	}
	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("embeddings response contained no vector")
	}
	emb := result.Data[0].Embedding
	e.cache.Store(text, emb)
	return emb, nil
}

func mockEmbedding(text string) []float32 {
	// 384-dim deterministic hash vector. Offline/dev only; not semantic.
	emb := make([]float32, 384)
	for i := range emb {
		h := 0
		for j, c := range text {
			h = (h*31 + int(c) + i + j) & 0x7fffffff
		}
		emb[i] = float32(h%1000-500) / 1000.0
	}
	return emb
}
