package retrieval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

type Embedder struct {
	apiKey  string
	baseURL string
	model   string
	cache   sync.Map
}

func NewEmbedder(apiKey, baseURL string) *Embedder {
	return &Embedder{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   "text-embedding-ada-002",
	}
}

func (e *Embedder) Embed(text string) ([]float32, error) {
	if v, ok := e.cache.Load(text); ok {
		return v.([]float32), nil
	}
	if e.apiKey == "" {
		// Return deterministic mock embedding
		return mockEmbedding(text), nil
	}
	body, _ := json.Marshal(map[string]interface{}{
		"model": e.model,
		"input": text,
	})
	req, _ := http.NewRequest("POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return mockEmbedding(text), nil
	}
	defer resp.Body.Close()

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil || len(result.Data) == 0 {
		return nil, fmt.Errorf("embedder: %w", err)
	}
	emb := result.Data[0].Embedding
	e.cache.Store(text, emb)
	return emb, nil
}

func mockEmbedding(text string) []float32 {
	// 384-dim mock: deterministic hash-based
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
