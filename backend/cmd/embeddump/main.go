// Command embeddump embeds a set of texts via the SAME embeddings provider the
// backend uses at runtime and writes the raw vectors to JSON. It exists so the
// offline fitters (scripts/fit_lda.py, scripts/fit_pca.py) can run their linear
// algebra without making network calls themselves — the embeddings come from
// the one verified Go HTTP path, guaranteeing dimension/model consistency with
// the server (a mismatch silently collapses the router to a dev sketch).
//
// Input is a JSON file of {"questions":[{"id","text",...}]} or a flat list of
// {"id","text"}. Output mirrors the input ids and adds an "embedding" array.
//
// Needs EMBEDDINGS_API_KEY (+ optional EMBEDDINGS_BASE_URL / EMBEDDINGS_MODEL),
// exactly like the server and routeeval.
//
//	cd backend && EMBEDDINGS_API_KEY=... go run ./cmd/embeddump \
//	  -in ../data/routing_questions.json -out ../data/routing_embeddings.json
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/retrieval"
)

// embedWithTask calls the embeddings provider directly with a Jina v3 `task`
// adapter (e.g. "classification", "separation", "text-matching"). The production
// embedder deliberately does NOT send a task (it serves general retrieval), so
// this is an experiment-only path used to fit a task-tuned routing projection.
func embedWithTask(baseURL, model, apiKey, task, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"model": model,
		"task":  task,
		"input": []string{text},
	})
	req, _ := http.NewRequest("POST", baseURL+"/embeddings", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(b))
	}
	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no vector returned")
	}
	return result.Data[0].Embedding, nil
}

type item struct {
	ID    string `json:"id"`
	Text  string `json:"text"`
	Path  string `json:"path,omitempty"`
	Label string `json:"label,omitempty"`
}

type outItem struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Label     string    `json:"label,omitempty"`
	Embedding []float32 `json:"embedding"`
}

func loadItems(path string) ([]item, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Try {"questions":[...]} first, then a flat array.
	var wrapped struct {
		Questions []item `json:"questions"`
	}
	if err := json.Unmarshal(b, &wrapped); err == nil && len(wrapped.Questions) > 0 {
		return wrapped.Questions, nil
	}
	var flat []item
	if err := json.Unmarshal(b, &flat); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return flat, nil
}

func main() {
	in := flag.String("in", "../data/routing_questions.json", "input questions JSON")
	out := flag.String("out", "../data/routing_embeddings.json", "output embeddings JSON")
	task := flag.String("task", "", "optional Jina v3 task adapter (classification|separation|text-matching|retrieval.query); empty = default like the server")
	flag.Parse()

	cfg := config.Load()
	emb := retrieval.NewEmbedder(cfg.EmbeddingsAPIKey, cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel)
	if emb.Mock() {
		fmt.Fprintln(os.Stderr, "[embeddump] WARNING: no EMBEDDINGS_API_KEY — emitting MOCK (non-semantic) vectors. Set the key for a real fit.")
	}
	if *task != "" {
		fmt.Fprintf(os.Stderr, "[embeddump] using Jina task adapter %q (experiment path, not the server default)\n", *task)
	}

	items, err := loadItems(*in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}

	results := make([]outItem, 0, len(items))
	for _, it := range items {
		var vec []float32
		var err error
		if *task != "" && !emb.Mock() {
			vec, err = embedWithTask(cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel, cfg.EmbeddingsAPIKey, *task, it.Text)
		} else {
			vec, err = emb.Embed(it.Text)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "embed %s: %v\n", it.ID, err)
			os.Exit(1)
		}
		label := it.Label
		if label == "" {
			label = it.Path // routing_questions.json uses "path" for the class
		}
		results = append(results, outItem{ID: it.ID, Text: it.Text, Label: label, Embedding: vec})
	}

	dim := 0
	if len(results) > 0 {
		dim = len(results[0].Embedding)
	}
	fmt.Fprintf(os.Stderr, "[embeddump] embedded %d items, dim=%d, model=%s, mock=%v\n",
		len(results), dim, cfg.EmbeddingsModel, emb.Mock())

	b, _ := json.Marshal(struct {
		Model string    `json:"model"`
		Dim   int       `json:"dim"`
		Items []outItem `json:"items"`
	}{Model: cfg.EmbeddingsModel, Dim: dim, Items: results})
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "[embeddump] wrote %s\n", *out)
}
