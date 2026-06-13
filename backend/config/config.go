package config

import (
	"os"
	"path/filepath"
	"strconv"
)

type Config struct {
	OpenRouterAPIKey  string
	DefaultModel      string
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
}

func Load() *Config {
	return &Config{
		OpenRouterAPIKey:  getEnv("OPENROUTER_API_KEY", ""),
		DefaultModel:      getEnv("DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
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
