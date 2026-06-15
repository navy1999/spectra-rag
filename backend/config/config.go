package config

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Config struct {
	OpenRouterAPIKey string
	DefaultModel     string
	// FallbackModels are tried in order when the primary model returns a 429
	// (free models are frequently rate-limited upstream). Set via FALLBACK_MODELS
	// (comma-separated); the built-in default rotates through common free models
	// so the live demo degrades gracefully instead of dying on a rate limit.
	FallbackModels     []string
	Port               string
	MaxHops            int
	RateLimitRPM       int
	MockLLM            bool
	Debug              bool
	GraphPath          string
	PCACentroidsPath   string
	PCAModelPath       string
	LDARouterPath      string
	NodeEmbeddingsPath string
	OpenRouterBaseURL  string
	IngestToken        string

	// Topic-driven ingestion (v3): build a graph on demand from an arXiv topic.
	ArxivBaseURL         string
	TopicIngestMaxPapers int
	TopicIngestEnabled   bool
	// Size-gated PCA compression of the ingested node index (v3.2): when a graph
	// has more than CompressThreshold nodes, embeddings are PCA-compressed to
	// CompressDim dims to fit free-tier RAM. Below the threshold the index stays
	// full-dim (best recall). Dim=0 disables compression entirely.
	NodeIndexCompressDim       int
	NodeIndexCompressThreshold int

	// Embeddings are a separate provider from the chat model. OpenRouter does
	// not serve embeddings, so the router needs a real embeddings endpoint
	// (default Jina) for its PCA projection to be semantically meaningful.
	EmbeddingsBaseURL string
	EmbeddingsModel   string
	EmbeddingsAPIKey  string
	// EmbeddingsTask is an optional Jina v3 task adapter sent on every embedding
	// request (default "classification"). The query embedding is used ONLY for
	// routing, and the supervised LDA router is fitted on classification-task
	// embeddings — pinning the same task here keeps the live router consistent
	// with the fitted projection. Set to empty to disable the task field.
	EmbeddingsTask string
}

func Load() *Config {
	return &Config{
		OpenRouterAPIKey:   getEnv("OPENROUTER_API_KEY", ""),
		DefaultModel:       getEnv("DEFAULT_MODEL", "liquid/lfm-2.5-1.2b-instruct:free"),
		FallbackModels:     getList("FALLBACK_MODELS", []string{"openai/gpt-oss-20b:free", "nvidia/nemotron-nano-9b-v2:free", "meta-llama/llama-3.2-3b-instruct:free"}),
		Port:               getEnv("PORT", "8080"),
		MaxHops:            getInt("MAX_HOPS", 3),
		RateLimitRPM:       getInt("RATE_LIMIT_RPM", 20),
		MockLLM:            getBool("MOCK_LLM", false),
		Debug:              getBool("DEBUG", false),
		GraphPath:          getEnv("GRAPH_PATH", resolveDataFile("graph.json")),
		PCACentroidsPath:   getEnv("PCA_CENTROIDS_PATH", resolveDataFile("pca_centroids.json")),
		PCAModelPath:       getEnv("PCA_MODEL_PATH", resolveDataFile("pca_model.json")),
		LDARouterPath:      getEnv("LDA_ROUTER_PATH", resolveDataFile("lda_router.json")),
		NodeEmbeddingsPath: getEnv("NODE_EMBEDDINGS_PATH", resolveDataFile("node_embeddings.json")),
		OpenRouterBaseURL:  getEnv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
		IngestToken:        getEnv("INGEST_TOKEN", ""),

		ArxivBaseURL:               getEnv("ARXIV_BASE_URL", "https://export.arxiv.org/api"),
		TopicIngestMaxPapers:       getInt("TOPIC_INGEST_MAX_PAPERS", 60),
		TopicIngestEnabled:         getBool("TOPIC_INGEST_ENABLED", true),
		NodeIndexCompressDim:       getInt("NODE_INDEX_COMPRESS_DIM", 128),
		NodeIndexCompressThreshold: getInt("NODE_INDEX_COMPRESS_THRESHOLD", 1500),
		EmbeddingsBaseURL:          getEnv("EMBEDDINGS_BASE_URL", "https://api.jina.ai/v1"),
		EmbeddingsModel:            getEnv("EMBEDDINGS_MODEL", "jina-embeddings-v3"),
		EmbeddingsAPIKey:           getEnv("EMBEDDINGS_API_KEY", ""),
		EmbeddingsTask:             getEnv("EMBEDDINGS_TASK", "classification"),
	}
}

func getEnv(key, def string) string {
	// Trim surrounding whitespace/newlines: a key pasted with a trailing newline
	// produces an invalid Authorization header ("invalid header field value"),
	// and none of our config values legitimately carry edge whitespace.
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func getInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getList(key string, def []string) []string {
	if v := os.Getenv(key); v != "" {
		var out []string
		for _, s := range strings.Split(v, ",") {
			if s = strings.TrimSpace(s); s != "" {
				out = append(out, s)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return def
}

func getBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		b, err := strconv.ParseBool(v)
		if err == nil {
			return b
		}
	}
	return def
}

// resolveDataFile returns the path to a file under data/, checking both the
// current working directory (e.g. Docker WORKDIR or `go run ./backend` from
// repo root) and one level up (e.g. `cd backend && go run .`). Falls back to
// the cwd-relative path if neither exists, preserving the existing graceful
// empty/default fallback in the graph loader and PCA router.
func resolveDataFile(filename string) string {
	candidates := []string{
		filepath.Join("data", filename),
		filepath.Join("..", "data", filename),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return candidates[0]
}
