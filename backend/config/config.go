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
	FallbackModels    []string
	Port              string
	MaxHops           int
	RateLimitRPM      int
	MockLLM           bool
	Debug             bool
	GraphPath         string
	PCACentroidsPath  string
	PCAModelPath      string
	OpenRouterBaseURL string
	IngestToken       string

	// Embeddings are a separate provider from the chat model. OpenRouter does
	// not serve embeddings, so the router needs a real embeddings endpoint
	// (default Jina) for its PCA projection to be semantically meaningful.
	EmbeddingsBaseURL string
	EmbeddingsModel   string
	EmbeddingsAPIKey  string
}

func Load() *Config {
	return &Config{
		OpenRouterAPIKey:  getEnv("OPENROUTER_API_KEY", ""),
		DefaultModel:      getEnv("DEFAULT_MODEL", "nex-agi/nex-n2-pro:free"),
		FallbackModels:    getList("FALLBACK_MODELS", []string{"openai/gpt-oss-20b:free", "nvidia/nemotron-nano-9b-v2:free", "meta-llama/llama-3.2-3b-instruct:free"}),
		Port:              getEnv("PORT", "8080"),
		MaxHops:           getInt("MAX_HOPS", 3),
		RateLimitRPM:      getInt("RATE_LIMIT_RPM", 20),
		MockLLM:           getBool("MOCK_LLM", false),
		Debug:             getBool("DEBUG", false),
		GraphPath:         getEnv("GRAPH_PATH", resolveDataFile("graph.json")),
		PCACentroidsPath:  getEnv("PCA_CENTROIDS_PATH", resolveDataFile("pca_centroids.json")),
		PCAModelPath:      getEnv("PCA_MODEL_PATH", resolveDataFile("pca_model.json")),
		OpenRouterBaseURL: getEnv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
		IngestToken:       getEnv("INGEST_TOKEN", ""),
		EmbeddingsBaseURL: getEnv("EMBEDDINGS_BASE_URL", "https://api.jina.ai/v1"),
		EmbeddingsModel:   getEnv("EMBEDDINGS_MODEL", "jina-embeddings-v3"),
		EmbeddingsAPIKey:  getEnv("EMBEDDINGS_API_KEY", ""),
	}
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
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
