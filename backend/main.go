package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gin-gonic/gin"
	pcacgo "github.com/navy1999/spectra-rag/backend/cgo"
	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/handlers"
	"github.com/navy1999/spectra-rag/backend/middleware"
	"github.com/navy1999/spectra-rag/backend/retrieval"
)

func main() {
	cfg := config.Load()

	if cfg.MockLLM {
		log.Println("[spectra-rag] MOCK_LLM=true — responses will be synthetic")
	}

	// Load graph
	graph, err := retrieval.LoadGraph(cfg.GraphPath)
	if err != nil {
		log.Fatalf("failed to load graph: %v", err)
	}
	log.Printf("[spectra-rag] graph loaded: %d nodes", graph.NodeCount())

	// Load the PCA model into the C++ engine (no-op in the pure-Go build).
	if err := pcacgo.LoadModel(cfg.PCAModelPath); err != nil {
		log.Printf("[spectra-rag] PCA projection: %v", err)
	} else {
		log.Printf("[spectra-rag] PCA model loaded from %s (Eigen engine active)", cfg.PCAModelPath)
	}

	// Store holds the graph + derived Trie and supports atomic hot-swap via /ingest.
	store := retrieval.NewStore(graph)
	log.Printf("[spectra-rag] trie built")

	// Gin setup
	if !cfg.Debug {
		gin.SetMode(gin.ReleaseMode)
	}
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.CORS())
	r.Use(middleware.RateLimit(cfg.RateLimitRPM))

	// Routes
	h := handlers.New(cfg, store)
	r.POST("/query", h.Query)
	r.GET("/health", h.Health)
	r.GET("/graph", h.GraphInfo)
	r.POST("/ingest", h.Ingest)

	port := cfg.Port
	if port == "" {
		port = "8080"
	}
	log.Printf("[spectra-rag] listening on :%s", port)

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		if err := r.Run(":" + port); err != nil {
			log.Fatalf("server error: %v", err)
		}
	}()
	<-quit
	log.Println("[spectra-rag] shutting down")
}
